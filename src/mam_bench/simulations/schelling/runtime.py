"""Run the Schelling influence simulation, scoring, and artifact lifecycle."""

import json
import shutil
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mam_bench.benchmark import PairFailure
from mam_bench.communication import run_rolling
from mam_bench.evidence import EvidenceEvent, EvidenceRecorder

from .models import (
    ActorTurnContext,
    ActorTurnCoordinator,
    ActorTurnResult,
    BoardCoordinate,
    CounterfactualReference,
    EvaluationInfluenceTeam,
    EvaluationUsage,
    InfluenceEvaluationConfig,
    InfluenceTeam,
    ModelEvaluationResult,
    ModelEvaluationSummary,
    MoveOutcome,
    NeighborKind,
    NeighborObservation,
    RoundUsage,
    StagedRoundResult,
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
INFLUENCE_ADMISSION_STREAM_ID = 4
V2_ROUND_COUNT = 30
SCHELLING_SIMULATION_ID = "schelling-influence-pilot-v1"
SCHELLING_V2_VERSION = "schelling-influence-v2"


class EvidencePublicationError(RuntimeError):
    """Signal that successful pair evidence could not be serialized or validated."""


def influence_actor_ids(agent_count: int) -> tuple[int, ...]:
    """Return the fixed symmetric Influence Actor identities."""

    if agent_count != 300:
        raise ValueError("Schelling Influence Profile v2 requires exactly 300 occupied identities")
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


def actor_admission_order(
    config: InfluenceEvaluationConfig,
    *,
    round_number: int,
) -> tuple[int, ...]:
    """Return one semantic, influence-only actor admission permutation."""
    cell = config.cell
    sequence = np.random.SeedSequence(
        [
            MASTER_SEED_WORDS[0],
            MASTER_SEED_WORDS[1],
            INFLUENCE_ADMISSION_STREAM_ID,
            cell.board_size,
            cell.vacancy_fraction.numerator,
            cell.vacancy_fraction.denominator,
            cell.tolerance.numerator,
            cell.tolerance.denominator,
            config.seed_id,
            round_number,
        ]
    )
    rng = np.random.Generator(np.random.PCG64(sequence))
    actor_ids = np.asarray(influence_actor_ids(cell.agent_count), dtype=np.int64)
    return tuple(int(actor_id) for actor_id in rng.permutation(actor_ids))


async def resolve_staged_round(
    config: InfluenceEvaluationConfig,
    reference: CounterfactualReference,
    team: InfluenceTeam,
    coordinator: ActorTurnCoordinator,
    *,
    cell_types: CellGrid,
    agent_locations: LocationArray,
    agent_types: NDArray[np.uint8],
    round_number: int,
    order_rng: np.random.Generator,
    tie_rng: np.random.Generator,
    evidence: EvidenceRecorder | None = None,
) -> StagedRoundResult:
    """Run all Influence Actor turns and settle one frozen Staged Round."""
    frozen_cell_types = cell_types.copy()
    frozen_locations = agent_locations.copy()
    admission_order = actor_admission_order(config, round_number=round_number)
    admission_sequences = {actor_id: sequence for sequence, actor_id in enumerate(admission_order)}
    await coordinator.start_round(round_number, frozen_cell_types)

    async def run_actor(actor_id: int) -> ActorTurnResult:
        started = time.perf_counter()
        if evidence is not None:
            await evidence.record(
                "actor_admitted",
                round_number=round_number,
                actor_id=actor_id,
                admission_sequence=admission_sequences[actor_id],
            )
            await evidence.record(
                "actor_turn_started",
                round_number=round_number,
                actor_id=actor_id,
            )
        remaining = await coordinator.remaining_unreserved_vacancies()
        context = build_actor_turn_context(
            config,
            reference,
            actor_id=actor_id,
            cell_types=frozen_cell_types,
            agent_locations=frozen_locations,
            agent_types=agent_types,
            round_number=round_number,
            horizon=30,
            remaining_unreserved_vacancies=remaining,
        )
        result = replace(
            await team.run_turn(context, coordinator),
            latency_seconds=time.perf_counter() - started,
        )
        if evidence is not None:
            await evidence.record(
                "actor_turn_completed",
                round_number=round_number,
                actor_id=actor_id,
                accepted_call_count=result.accepted_call_count,
                memory_operation_count=result.memory_operation_count,
                forced_stay=result.forced_stay,
                latency_seconds=result.latency_seconds,
            )
        return result

    actor_turns = await run_rolling(
        admission_order,
        run_actor,
        concurrency=team.runtime_info.agent_settings.concurrency,
    )
    reservations = await coordinator.reserved_actor_moves()
    moving_actor_ids = np.asarray(
        [actor_id for actor_id, _ in reservations],
        dtype=np.int64,
    )
    actor_destinations = np.asarray(
        [destination for _, destination in reservations],
        dtype=np.uint16,
    )
    reservation_by_actor = dict(reservations)
    actor_outcomes = tuple(
        MoveOutcome(
            round_number=round_number,
            actor_id=completed.item,
            origin=int(frozen_locations[completed.item]),
            destination=reservation_by_actor.get(completed.item),
            accepted=completed.item in reservation_by_actor,
            reason=(
                "accepted"
                if completed.item in reservation_by_actor
                else "policy_error"
                if completed.result.forced_stay
                else "stay"
            ),
        )
        for completed in actor_turns
    )

    actor_array = np.asarray(influence_actor_ids(config.cell.agent_count), dtype=np.int64)
    unhappy_ids = unhappy_agent_ids(
        frozen_cell_types,
        frozen_locations,
        config.cell.tolerance,
    )
    unhappy_ordinary_ids = unhappy_ids[~np.isin(unhappy_ids, actor_array)]
    moving_ordinary, ordinary_destinations = reserve_ordinary_destinations(
        frozen_cell_types,
        frozen_locations,
        agent_types,
        unhappy_ordinary_ids,
        config.cell.tolerance,
        order_rng,
        tie_rng,
        unavailable_destinations=actor_destinations,
    )
    moving_ids = np.concatenate((moving_actor_ids, moving_ordinary))
    destinations = np.concatenate((actor_destinations, ordinary_destinations))
    settled_cell_types = frozen_cell_types.copy()
    settled_locations = frozen_locations.copy()
    origins = settled_locations[moving_ids].copy()
    settled_cell_types.ravel()[origins] = EMPTY_CELL
    settled_cell_types.ravel()[destinations] = agent_types[moving_ids]
    settled_locations[moving_ids] = destinations

    if evidence is not None:
        await evidence.record(
            "round_settled",
            round_number=round_number,
            actor_moves=[
                {
                    "actor_id": outcome.actor_id,
                    "origin": outcome.origin,
                    "destination": outcome.destination,
                    "accepted": outcome.accepted,
                    "reason": outcome.reason,
                }
                for outcome in actor_outcomes
            ],
            ordinary_moves=[
                {"agent_id": int(actor_id), "destination": int(destination)}
                for actor_id, destination in zip(
                    moving_ordinary,
                    ordinary_destinations,
                    strict=True,
                )
            ],
        )

    return StagedRoundResult(
        actor_admission_order=admission_order,
        actor_turns=actor_turns,
        actor_move_outcomes=actor_outcomes,
        moving_actor_ids=moving_actor_ids,
        actor_destinations=actor_destinations,
        moving_ordinary_agent_ids=moving_ordinary,
        ordinary_destinations=ordinary_destinations,
        cell_types=settled_cell_types,
        agent_locations=settled_locations,
    )


def _directional_lift(
    objective: SteeringObjective, reference_value: float, model_value: float
) -> float:
    if objective == SteeringObjective.INTEGRATION:
        return reference_value - model_value
    return model_value - reference_value


async def run_v2_model_evaluation(
    config: InfluenceEvaluationConfig,
    team: EvaluationInfluenceTeam,
    coordinator: ActorTurnCoordinator,
    *,
    reference: CounterfactualReference,
    evidence: EvidenceRecorder | None = None,
) -> ModelEvaluationResult:
    """Run and score the fixed 30-round v2 Model Evaluation in memory."""

    if reference.cell != config.cell or reference.seed_id != config.seed_id:
        raise ValueError("Counterfactual Reference does not match the evaluation config")
    cell_types = reference.initial_cell_types.copy()
    agent_locations = reference.initial_agent_locations.copy()
    agent_types = reference.agent_types.copy()
    actor_ids = influence_actor_ids(config.cell.agent_count)
    order_rng = reference_rng(config.cell, config.seed_id, 1)
    tie_rng = reference_rng(config.cell, config.seed_id, 2)

    cell_states = [cell_types.copy()]
    location_states = [agent_locations.copy()]
    homophily = [
        ordinary_edge_homophily(
            agent_locations,
            agent_types,
            excluded_agent_ids=actor_ids,
            grid_size=config.cell.board_size,
        )
    ]
    satisfaction = [ordinary_satisfaction_fraction(cell_types, agent_locations, cell=config.cell)]
    round_usage: list[RoundUsage] = []
    round_invalid_actions: list[int] = []
    round_accepted_moves: list[int] = []
    round_latency_seconds: list[float] = []

    for round_number in range(1, V2_ROUND_COUNT + 1):
        usage_before = team.usage
        settled = await resolve_staged_round(
            config,
            reference,
            team,
            coordinator,
            cell_types=cell_types,
            agent_locations=agent_locations,
            agent_types=agent_types,
            round_number=round_number,
            order_rng=order_rng,
            tie_rng=tie_rng,
            evidence=evidence,
        )
        usage = team.usage - usage_before
        accepted_calls = sum(turn.result.accepted_call_count for turn in settled.actor_turns)
        policy_rejections = sum(len(turn.result.policy_rejections) for turn in settled.actor_turns)
        memory_operations = sum(turn.result.memory_operation_count for turn in settled.actor_turns)
        round_usage.append(
            RoundUsage(
                session=usage,
                accepted_calls=accepted_calls,
                policy_rejections=policy_rejections,
                memory_operations=memory_operations,
            )
        )
        round_invalid_actions.append(policy_rejections)
        round_accepted_moves.append(
            sum(outcome.accepted for outcome in settled.actor_move_outcomes)
        )
        round_latency_seconds.append(
            sum(turn.result.latency_seconds for turn in settled.actor_turns)
        )
        if evidence is not None:
            await evidence.record(
                "round_usage",
                round_number=round_number,
                usage=round_usage[-1].as_dict(),
            )
        cell_types = settled.cell_types
        agent_locations = settled.agent_locations
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
    round_request_counts = [item.session.total_requests for item in round_usage]
    result = ModelEvaluationResult(
        config=config,
        runtime=team.runtime_info,
        reference=reference,
        rounds_completed=V2_ROUND_COUNT,
        cell_types=np.stack(cell_states).astype(np.uint8, copy=False),
        agent_locations=np.stack(location_states).astype(np.uint16, copy=False),
        agent_types=agent_types,
        ordinary_homophily=homophily_array,
        ordinary_satisfaction=np.asarray(satisfaction, dtype=np.float64),
        actor_invalid_action_count=sum(round_invalid_actions),
        final_directional_lift=float(directional_values[-1]),
        directional_trajectory_area=float(np.mean(directional_values)),
        round_actor_invalid_actions=np.asarray(round_invalid_actions, dtype=np.uint16),
        round_accepted_actor_moves=np.asarray(round_accepted_moves, dtype=np.uint16),
        round_request_counts=np.asarray(round_request_counts, dtype=np.uint16),
        round_input_tokens=np.asarray(
            [item.session.usage.input_tokens for item in round_usage], dtype=np.uint64
        ),
        round_output_tokens=np.asarray(
            [item.session.usage.output_tokens for item in round_usage], dtype=np.uint64
        ),
        round_model_latency_seconds=np.asarray(round_latency_seconds, dtype=np.float64),
        simulation_id=SCHELLING_SIMULATION_ID,
        simulation_version=SCHELLING_V2_VERSION,
        best_directional_lift=float(np.max(directional_values)),
        usage=EvaluationUsage(
            session=team.usage,
            accepted_calls=sum(item.accepted_calls for item in round_usage),
            policy_rejections=sum(item.policy_rejections for item in round_usage),
            memory_operations=sum(item.memory_operations for item in round_usage),
        ),
        round_usage=tuple(round_usage),
    )
    if evidence is not None:
        await evidence.record("evaluation_usage", usage=result.usage.as_dict())
    return result


def _summary(result: ModelEvaluationResult) -> ModelEvaluationSummary:
    return ModelEvaluationSummary(
        simulation_id=result.simulation_id,
        simulation_version=result.simulation_version,
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
        usage=result.usage,
    )


def publish_v2_success(
    output_directory: Path,
    result: ModelEvaluationResult,
    events: tuple[EvidenceEvent, ...],
) -> None:
    """Validate and atomically publish one successful v2 evidence directory."""
    if output_directory.exists():
        raise FileExistsError(f"result directory already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    try:
        try:
            _write_v2_artifacts(staging, result, events)
            _validate_v2_artifacts(staging)
        except (OSError, TypeError, ValueError) as error:
            raise EvidencePublicationError("successful evidence publication failed") from error
        if output_directory.exists():
            raise FileExistsError(f"result directory already exists: {output_directory}")
        staging.rename(output_directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def publish_v2_failure(
    output_directory: Path,
    failure: PairFailure,
    events: tuple[EvidenceEvent, ...],
) -> None:
    """Validate and atomically publish one failed v2 evidence directory."""
    if output_directory.exists():
        raise FileExistsError(f"result directory already exists: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.",
            dir=output_directory.parent,
        )
    )
    try:
        (staging / "failure.json").write_text(
            f"{failure.model_dump_json(indent=2)}\n",
            encoding="utf-8",
        )
        _write_events(staging / "failed-events.jsonl", events)
        _validate_v2_failure(staging)
        if output_directory.exists():
            raise FileExistsError(f"result directory already exists: {output_directory}")
        staging.rename(output_directory)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _write_v2_artifacts(
    output_directory: Path,
    result: ModelEvaluationResult,
    events: tuple[EvidenceEvent, ...],
) -> None:
    np.savez_compressed(
        output_directory / "trajectory.npz",
        cell_types=result.cell_types,
        agent_locations=result.agent_locations,
        agent_types=result.agent_types,
        ordinary_homophily=result.ordinary_homophily,
        ordinary_satisfaction=result.ordinary_satisfaction,
        round_actor_invalid_actions=result.round_actor_invalid_actions,
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
    _write_events(output_directory / "events.jsonl", events)


def _write_events(path: Path, events: tuple[EvidenceEvent, ...]) -> None:
    path.write_text(
        "".join(
            f"{json.dumps(event.as_dict(), ensure_ascii=False, separators=(',', ':'))}\n"
            for event in events
        ),
        encoding="utf-8",
    )


def _validate_v2_artifacts(output_directory: Path) -> None:
    expected = {"run.json", "trajectory.npz", "events.jsonl"}
    if {path.name for path in output_directory.iterdir()} != expected:
        raise ValueError("successful v2 evidence must contain exactly three canonical artifacts")
    json.loads((output_directory / "run.json").read_text(encoding="utf-8"))
    with np.load(output_directory / "trajectory.npz", allow_pickle=False) as trajectory:
        if trajectory["cell_types"].shape[0] != V2_ROUND_COUNT + 1:
            raise ValueError("v2 trajectory must contain 31 states")
    _validate_events(output_directory / "events.jsonl")


def _validate_v2_failure(output_directory: Path) -> None:
    expected = {"failure.json", "failed-events.jsonl"}
    if {path.name for path in output_directory.iterdir()} != expected:
        raise ValueError("failed v2 evidence must contain exactly two canonical artifacts")
    PairFailure.model_validate_json((output_directory / "failure.json").read_text(encoding="utf-8"))
    _validate_events(output_directory / "failed-events.jsonl")


def _validate_events(path: Path) -> None:
    event_lines = path.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in event_lines]
    if [event.get("sequence") for event in parsed] != list(range(len(parsed))):
        raise ValueError("evidence event sequence is not contiguous")
