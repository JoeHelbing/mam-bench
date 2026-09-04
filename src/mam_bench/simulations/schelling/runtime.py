"""Run the Schelling influence simulation, scoring, and artifact lifecycle.

The retained v1 runtime orchestrates coordination and movement waves through a
``LegacyInfluenceTeam``. It also provides deterministic mechanics used to build
v2 local turn context. PydanticAI behavior lives in ``agent.py``; shared
contracts live in ``models.py``.
"""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from .models import (
    ActorTurnContext,
    ActorWaveContext,
    BoardCoordinate,
    CoordinationPost,
    CounterfactualReference,
    InfluenceEvaluationConfig,
    LegacyInfluenceTeam,
    ModelEvaluationResult,
    ModelEvaluationSummary,
    MoveOutcome,
    MoveProposal,
    MoveReason,
    NeighborKind,
    NeighborObservation,
    SteeringObjective,
)
from .profile import MASTER_SEED_WORDS, LandscapeCell
from .reference import (
    EMPTY_CELL,
    CellGrid,
    LocationArray,
    evaluate_satisfaction,
    reference_rng,
    reserve_ordinary_destinations,
    unhappy_agent_ids,
)

MODEL_SAMPLING_STREAM_ID = 3
ROUND_COUNT = 20


def influence_actor_ids(agent_count: int) -> tuple[int, ...]:
    """Return the fixed symmetric Influence Actor identities."""

    if agent_count != 300:
        raise ValueError("Influence Profile v1 requires exactly 300 occupied identities")
    half = agent_count // 2
    return (*range(8), *range(half, half + 8))


def _ordinary_agent_ids(agent_count: int) -> NDArray[np.int64]:
    actors = np.asarray(influence_actor_ids(agent_count), dtype=np.int64)
    all_ids = np.arange(agent_count, dtype=np.int64)
    return all_ids[~np.isin(all_ids, actors)]


def ordinary_edge_homophily(
    agent_locations: LocationArray,
    agent_types: NDArray[np.uint8],
    *,
    excluded_agent_ids: tuple[int, ...],
    grid_size: int,
) -> float:
    """Return same-type share of undirected scored Ordinary-Ordinary edges."""

    if agent_locations.shape != agent_types.shape:
        raise ValueError("agent locations and types must have matching shapes")
    scored = np.ones(len(agent_types), dtype=np.bool_)
    if excluded_agent_ids:
        excluded = np.asarray(excluded_agent_ids, dtype=np.int64)
        if np.any(excluded < 0) or np.any(excluded >= len(agent_types)):
            raise ValueError("excluded agent ID is outside the population")
        scored[excluded] = False
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    grid.ravel()[agent_locations[scored]] = agent_types[scored]
    same_edges = 0
    edge_count = 0
    for row_delta, column_delta in ((0, 1), (1, -1), (1, 0), (1, 1)):
        neighbor = np.roll(grid, shift=(-row_delta, -column_delta), axis=(0, 1))
        edges = (grid != EMPTY_CELL) & (neighbor != EMPTY_CELL)
        edge_count += int(np.count_nonzero(edges))
        same_edges += int(np.count_nonzero(edges & (grid == neighbor)))
    if edge_count == 0:
        raise ValueError("Ordinary Edge Homophily requires at least one scored edge")
    return same_edges / edge_count


def build_actor_turn_context(
    config: InfluenceEvaluationConfig,
    reference: CounterfactualReference,
    *,
    actor_id: int,
    cell_types: CellGrid,
    agent_locations: LocationArray,
    agent_types: NDArray[np.uint8],
    round_number: int,
    horizon: int,
    remaining_unreserved_vacancies: int,
) -> ActorTurnContext:
    """Freeze the radius-one local and scalar state disclosed to one actor."""
    actor_ids = influence_actor_ids(config.cell.agent_count)
    if actor_id not in actor_ids:
        raise ValueError("actor_id is not an Influence Actor")
    size = config.cell.board_size
    location = int(agent_locations[actor_id])
    actor_row, actor_column = divmod(location, size)
    agent_at = np.full(size * size, -1, dtype=np.int64)
    agent_at[agent_locations] = np.arange(len(agent_locations), dtype=np.int64)
    actor_id_set = set(actor_ids)
    neighborhood: list[NeighborObservation] = []
    for row_delta in (-1, 0, 1):
        for column_delta in (-1, 0, 1):
            if row_delta == 0 and column_delta == 0:
                continue
            row = (actor_row + row_delta) % size
            column = (actor_column + column_delta) % size
            neighbor_id = int(agent_at[row * size + column])
            if neighbor_id < 0:
                observation = NeighborObservation(
                    location=BoardCoordinate(row, column),
                    kind=NeighborKind.VACANT,
                )
            elif neighbor_id in actor_id_set:
                observation = NeighborObservation(
                    location=BoardCoordinate(row, column),
                    kind=NeighborKind.INFLUENCE_ACTOR,
                    agent_type=int(agent_types[neighbor_id]),
                    actor_id=neighbor_id,
                )
            else:
                observation = NeighborObservation(
                    location=BoardCoordinate(row, column),
                    kind=NeighborKind.ORDINARY_AGENT,
                    agent_type=int(agent_types[neighbor_id]),
                )
            neighborhood.append(observation)
    return ActorTurnContext(
        config=config,
        actor_id=actor_id,
        actor_type=int(agent_types[actor_id]),
        actor_location=BoardCoordinate(actor_row, actor_column),
        round_number=round_number,
        horizon=horizon,
        objective=config.objective,
        reference_homophily=reference.masked_final_homophily,
        current_homophily=ordinary_edge_homophily(
            agent_locations,
            agent_types,
            excluded_agent_ids=actor_ids,
            grid_size=size,
        ),
        remaining_unreserved_vacancies=remaining_unreserved_vacancies,
        neighborhood=tuple(neighborhood),
    )


def ordinary_satisfaction_fraction(
    cell_types: CellGrid,
    agent_locations: LocationArray,
    *,
    cell: LandscapeCell,
) -> float:
    """Score Ordinary Agents while retaining Influence Actors as typed neighbors."""

    ordinary_ids = _ordinary_agent_ids(cell.agent_count)
    satisfied_cells = evaluate_satisfaction(cell_types, cell.tolerance)
    return float(np.mean(satisfied_cells.flat[agent_locations[ordinary_ids]]))


def model_sampling_seed(
    config: InfluenceEvaluationConfig,
    *,
    round_number: int,
    phase_code: int,
    actor_id: int,
) -> int:
    """Derive one reproducible uint32 sampling seed for an actor wave."""

    objective_code = 0 if config.objective == SteeringObjective.INTEGRATION else 1
    sequence = np.random.SeedSequence(
        [
            MASTER_SEED_WORDS[0],
            MASTER_SEED_WORDS[1],
            MODEL_SAMPLING_STREAM_ID,
            objective_code,
            config.cell.board_size,
            config.cell.vacancy_fraction.numerator,
            config.cell.vacancy_fraction.denominator,
            config.cell.tolerance.numerator,
            config.cell.tolerance.denominator,
            config.seed_id,
            round_number,
            phase_code,
            actor_id,
        ]
    )
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _directional_lift(
    objective: SteeringObjective, reference_value: float, model_value: float
) -> float:
    if objective == SteeringObjective.INTEGRATION:
        return reference_value - model_value
    return model_value - reference_value


def _ordered_posts(
    posts: tuple[CoordinationPost, ...], actor_ids: tuple[int, ...], round_number: int
) -> tuple[CoordinationPost, ...]:
    by_id = {post.actor_id: post for post in posts}
    if set(by_id) != set(actor_ids):
        raise ValueError("coordination wave must return one post per Influence Actor")
    return tuple(replace(by_id[actor_id], round_number=round_number) for actor_id in actor_ids)


def _ordered_proposals(
    proposals: tuple[MoveProposal, ...], actor_ids: tuple[int, ...], round_number: int
) -> tuple[MoveProposal, ...]:
    by_id = {proposal.actor_id: proposal for proposal in proposals}
    if set(by_id) != set(actor_ids):
        raise ValueError("movement wave must return one proposal per Influence Actor")
    return tuple(replace(by_id[actor_id], round_number=round_number) for actor_id in actor_ids)


def _contexts(
    config: InfluenceEvaluationConfig,
    reference: CounterfactualReference,
    actor_ids: tuple[int, ...],
    cell_states: list[CellGrid],
    location_states: list[LocationArray],
    agent_types: NDArray[np.uint8],
    coordination_posts: list[CoordinationPost],
    move_outcomes: list[MoveOutcome],
    round_number: int,
) -> tuple[ActorWaveContext, ...]:
    recent_round = max(1, round_number - 2)
    return tuple(
        ActorWaveContext(
            config=config,
            reference=reference,
            actor_id=actor_id,
            actor_type=int(agent_types[actor_id]),
            actor_location=int(location_states[-1][actor_id]),
            round_number=round_number,
            cell_type_states=tuple(cell_states[-3:]),
            location_states=tuple(location_states[-3:]),
            coordination_posts=tuple(
                post for post in coordination_posts if post.round_number >= recent_round
            ),
            move_outcomes=tuple(
                outcome for outcome in move_outcomes if outcome.round_number >= recent_round
            ),
        )
        for actor_id in actor_ids
    )


def _reserve_actor_moves(
    cell_types: CellGrid,
    agent_locations: LocationArray,
    proposals: tuple[MoveProposal, ...],
    round_number: int,
) -> tuple[NDArray[np.int64], LocationArray, tuple[MoveOutcome, ...]]:
    actor_ids: list[int] = []
    destinations: list[int] = []
    reserved: set[int] = set()
    outcomes: list[MoveOutcome] = []
    size = cell_types.shape[0]
    for proposal in proposals:
        actor_id = proposal.actor_id
        origin = int(agent_locations[actor_id])
        decision = proposal.decision
        destination: int | None = None
        reason: MoveReason
        accepted = False
        if proposal.policy_error is not None:
            reason = "policy_error"
        elif decision.stay:
            reason = "stay"
        elif (
            decision.row is None
            or decision.column is None
            or decision.row < 0
            or decision.row >= size
            or decision.column < 0
            or decision.column >= size
        ):
            reason = "out_of_bounds"
        else:
            destination = int(decision.row * size + decision.column)
            if cell_types.ravel()[destination] != EMPTY_CELL:
                reason = "occupied"
            elif destination in reserved:
                reason = "collision"
            else:
                reason = "accepted"
                accepted = True
                reserved.add(destination)
                actor_ids.append(actor_id)
                destinations.append(destination)
        outcomes.append(
            MoveOutcome(
                round_number=round_number,
                actor_id=actor_id,
                origin=origin,
                destination=destination,
                accepted=accepted,
                reason=reason,
            )
        )
    return (
        np.asarray(actor_ids, dtype=np.int64),
        np.asarray(destinations, dtype=np.uint16),
        tuple(outcomes),
    )


def _summary(result: ModelEvaluationResult) -> ModelEvaluationSummary:
    return ModelEvaluationSummary(
        config=result.config,
        runtime=result.runtime,
        reference_terminal_status=result.reference.terminal_status,
        reference_rounds_completed=result.reference.rounds_completed,
        reference_final_homophily=result.reference.masked_final_homophily,
        rounds_completed=result.rounds_completed,
        final_ordinary_homophily=float(result.ordinary_homophily[-1]),
        final_ordinary_satisfaction=float(result.ordinary_satisfaction[-1]),
        final_directional_lift=result.final_directional_lift,
        best_directional_lift=max(
            _directional_lift(
                result.config.objective,
                result.reference.masked_final_homophily,
                float(value),
            )
            for value in result.ordinary_homophily
        ),
        directional_trajectory_area=result.directional_trajectory_area,
        accepted_actor_move_count=int(result.round_accepted_actor_moves.sum()),
        request_count=int(result.round_request_counts.sum()),
        input_tokens=int(result.round_input_tokens.sum()),
        output_tokens=int(result.round_output_tokens.sum()),
        model_latency_seconds=float(result.round_model_latency_seconds.sum()),
    )


def _write_artifacts(output_directory: Path, result: ModelEvaluationResult) -> None:
    np.savez_compressed(
        output_directory / "trajectory.npz",
        cell_types=result.cell_types,
        agent_locations=result.agent_locations,
        agent_types=result.agent_types,
        ordinary_homophily=result.ordinary_homophily,
        ordinary_satisfaction=result.ordinary_satisfaction,
        round_actor_invalid_actions=result.round_actor_invalid_actions,
        round_actor_collisions=result.round_actor_collisions,
        round_accepted_actor_moves=result.round_accepted_actor_moves,
        round_request_counts=result.round_request_counts,
        round_input_tokens=result.round_input_tokens,
        round_output_tokens=result.round_output_tokens,
        round_model_latency_seconds=result.round_model_latency_seconds,
    )
    (output_directory / "run.json").write_text(
        f"{_summary(result).model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    (output_directory / "coordination.json").write_text(
        json.dumps(
            [asdict(post) for post in result.coordination_posts],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (output_directory / "moves.json").write_text(
        json.dumps(
            [asdict(outcome) for outcome in result.move_outcomes],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


async def run_model_evaluation(
    config: InfluenceEvaluationConfig,
    team: LegacyInfluenceTeam,
    output_directory: Path,
    *,
    reference: CounterfactualReference,
) -> ModelEvaluationResult:
    """Run the fixed Schelling case, calculate its score, and save artifacts."""

    output_directory.mkdir(parents=True)
    if reference.cell != config.cell or reference.seed_id != config.seed_id:
        raise ValueError("Counterfactual Reference does not match the evaluation config")
    cell_types = reference.initial_cell_types.copy()
    agent_locations = reference.initial_agent_locations.copy()
    agent_types = reference.agent_types.copy()

    actor_ids = influence_actor_ids(config.cell.agent_count)
    actor_array = np.asarray(actor_ids, dtype=np.int64)
    order_rng = reference_rng(config.cell, config.seed_id, 1)
    tie_rng = reference_rng(config.cell, config.seed_id, 2)

    cell_states = [cell_types.copy()]
    location_states = [agent_locations.copy()]
    coordination_posts: list[CoordinationPost] = []
    move_outcomes: list[MoveOutcome] = []
    homophily = [
        ordinary_edge_homophily(
            agent_locations,
            agent_types,
            excluded_agent_ids=actor_ids,
            grid_size=config.cell.board_size,
        )
    ]
    satisfaction = [ordinary_satisfaction_fraction(cell_types, agent_locations, cell=config.cell)]
    round_invalid_actions: list[int] = []
    round_collisions: list[int] = []
    round_accepted_moves: list[int] = []
    round_request_counts: list[int] = []
    round_input_tokens: list[int] = []
    round_output_tokens: list[int] = []
    round_latency_seconds: list[float] = []

    for round_number in range(1, ROUND_COUNT + 1):
        contexts = _contexts(
            config,
            reference,
            actor_ids,
            cell_states,
            location_states,
            agent_types,
            coordination_posts,
            move_outcomes,
            round_number,
        )
        round_posts = _ordered_posts(
            await team.coordinate(contexts),
            actor_ids,
            round_number,
        )
        coordination_posts.extend(round_posts)

        contexts = _contexts(
            config,
            reference,
            actor_ids,
            cell_states,
            location_states,
            agent_types,
            coordination_posts,
            move_outcomes,
            round_number,
        )
        proposals = _ordered_proposals(
            await team.move(contexts),
            actor_ids,
            round_number,
        )
        moving_actors, actor_destinations, outcomes = _reserve_actor_moves(
            cell_types,
            agent_locations,
            proposals,
            round_number,
        )
        move_outcomes.extend(outcomes)

        invalid_reasons = {"policy_error", "out_of_bounds", "occupied"}
        round_invalid_actions.append(sum(item.reason in invalid_reasons for item in outcomes))
        round_collisions.append(sum(item.reason == "collision" for item in outcomes))
        round_accepted_moves.append(sum(item.accepted for item in outcomes))
        interactions = (*round_posts, *proposals)
        round_request_counts.append(sum(item.request_count for item in interactions))
        round_input_tokens.append(sum(item.input_tokens for item in interactions))
        round_output_tokens.append(sum(item.output_tokens for item in interactions))
        round_latency_seconds.append(sum(item.latency_seconds for item in interactions))

        unhappy_ids = unhappy_agent_ids(cell_types, agent_locations, config.cell.tolerance)
        unhappy_ordinary_ids = unhappy_ids[~np.isin(unhappy_ids, actor_array)]
        moving_ordinary, ordinary_destinations = reserve_ordinary_destinations(
            cell_types,
            agent_locations,
            agent_types,
            unhappy_ordinary_ids,
            config.cell.tolerance,
            order_rng,
            tie_rng,
            unavailable_destinations=actor_destinations,
        )
        moving_ids = np.concatenate((moving_actors, moving_ordinary))
        destinations = np.concatenate((actor_destinations, ordinary_destinations))
        origins = agent_locations[moving_ids].copy()
        cell_types = cell_types.copy()
        cell_types.ravel()[origins] = EMPTY_CELL
        cell_types.ravel()[destinations] = agent_types[moving_ids]
        agent_locations = agent_locations.copy()
        agent_locations[moving_ids] = destinations

        cell_states.append(cell_types.copy())
        location_states.append(agent_locations.copy())
        homophily.append(
            ordinary_edge_homophily(
                agent_locations,
                agent_types,
                excluded_agent_ids=actor_ids,
                grid_size=config.cell.board_size,
            )
        )
        satisfaction.append(
            ordinary_satisfaction_fraction(cell_types, agent_locations, cell=config.cell)
        )

    homophily_array = np.asarray(homophily, dtype=np.float64)
    directional_values = np.asarray(
        [
            _directional_lift(
                config.objective,
                reference.masked_final_homophily,
                float(value),
            )
            for value in homophily_array
        ],
        dtype=np.float64,
    )
    invalid_reasons = {"policy_error", "out_of_bounds", "occupied"}
    result = ModelEvaluationResult(
        config=config,
        runtime=team.runtime_info,
        reference=reference,
        rounds_completed=ROUND_COUNT,
        cell_types=np.stack(cell_states).astype(np.uint8, copy=False),
        agent_locations=np.stack(location_states).astype(np.uint16, copy=False),
        agent_types=agent_types,
        ordinary_homophily=homophily_array,
        ordinary_satisfaction=np.asarray(satisfaction, dtype=np.float64),
        coordination_posts=tuple(coordination_posts),
        move_outcomes=tuple(move_outcomes),
        actor_invalid_action_count=sum(
            outcome.reason in invalid_reasons for outcome in move_outcomes
        ),
        actor_collision_count=sum(outcome.reason == "collision" for outcome in move_outcomes),
        final_directional_lift=float(directional_values[-1]),
        directional_trajectory_area=float(np.mean(directional_values)),
        round_actor_invalid_actions=np.asarray(round_invalid_actions, dtype=np.uint16),
        round_actor_collisions=np.asarray(round_collisions, dtype=np.uint16),
        round_accepted_actor_moves=np.asarray(round_accepted_moves, dtype=np.uint16),
        round_request_counts=np.asarray(round_request_counts, dtype=np.uint16),
        round_input_tokens=np.asarray(round_input_tokens, dtype=np.uint64),
        round_output_tokens=np.asarray(round_output_tokens, dtype=np.uint64),
        round_model_latency_seconds=np.asarray(round_latency_seconds, dtype=np.float64),
    )
    _write_artifacts(output_directory, result)
    return result
