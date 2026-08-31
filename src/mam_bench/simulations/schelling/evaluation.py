"""Schelling Influence Profile v1 mechanics, scoring, and artifacts."""

import hashlib
import json
import os
import platform
import re
from enum import StrEnum
from importlib.metadata import version
from pathlib import Path
from typing import Literal, Protocol, Self, cast

import numpy as np
import pydantic
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, model_validator

from mam_bench.runtime import RuntimeDescriptor, RuntimeMessage, ToolCallPart

from .profile import MASTER_SEED_WORDS, PROFILE, LandscapeCell
from .reference import (
    EMPTY_CELL,
    CellGrid,
    LocationArray,
    TerminalStatus,
    evaluate_satisfaction,
    initialize_population,
    reference_rng,
    reserve_ordinary_destinations,
    run_reference,
    unhappy_agent_ids,
)

INFLUENCE_SCHEMA_VERSION = "mam-bench.schelling-influence.v1"
MODEL_SAMPLING_STREAM_ID = 3
ROUND_COUNT = 20
COORDINATION_PHASE_CODE = 0
MOVEMENT_PHASE_CODE = 1


class InfluenceInfrastructureError(RuntimeError):
    """A provider or timeout failure that invalidates a synchronized wave."""


class SteeringObjective(StrEnum):
    """Assigned direction for Ordinary Edge Homophily."""

    INTEGRATION = "integration"
    SEGREGATION = "segregation"


class RuntimeSettings(BaseModel):
    """Frozen model-runtime settings recorded with every case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: Literal["openrouter", "openai-compatible"] = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "qwen/qwen3.8-27b"
    openrouter_provider_slug: str = "phala"
    openrouter_allow_fallbacks: bool = False
    openrouter_require_parameters: bool = True
    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    reasoning_effort: Literal["medium"] = "medium"
    max_completion_tokens: int = 4096
    request_limit: int = 4
    tool_call_limit: int = 3
    wave_timeout_seconds: float = 600.0


class InfluenceEvaluationConfig(BaseModel):
    """One complete twenty-round Influence Profile case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cell: LandscapeCell
    seed_id: int = Field(ge=20)
    objective: SteeringObjective
    round_count: Literal[20] = 20
    runtime: RuntimeSettings = RuntimeSettings()

    @model_validator(mode="after")
    def validate_selected_vacancy(self) -> Self:
        if self.cell.vacancy_index != 3:
            raise ValueError("Influence Profile v1 requires 25% vacancy")
        if self.cell.tolerance_index not in (11, 17, 20):
            raise ValueError("Influence Profile v1 requires a selected Test Spot")
        return self


class CounterfactualReference(BaseModel):
    """Same-seed all-Ordinary-Agent endpoint and masked comparator."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    cell: LandscapeCell
    seed_id: int
    terminal_status: TerminalStatus
    rounds_completed: int
    initial_cell_types: np.ndarray
    terminal_cell_types: np.ndarray
    terminal_agent_locations: np.ndarray
    agent_types: np.ndarray
    masked_agent_count: int
    masked_final_homophily: float
    unmasked_final_homophily: float
    ordinary_satisfaction: float


class CoordinationPost(BaseModel):
    """One Influence Actor's unrestricted shared text for a round."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    actor_id: int
    text: str
    policy_error: str | None = None
    request_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    new_messages_json: str | None = None
    round_number: int | None = Field(default=None, ge=1, le=ROUND_COUNT)


class CoordinationBoardPost(BaseModel):
    """One model-visible post in the standalone Coordination Board artifact."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    actor_id: int
    text: str
    policy_error: str | None = None


class CoordinationBoardRound(BaseModel):
    """One synchronized, stable-ID Coordination Board wave."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_number: int = Field(ge=1, le=ROUND_COUNT)
    posts: tuple[CoordinationBoardPost, ...]

    @model_validator(mode="after")
    def validate_stable_actor_order(self) -> Self:
        actor_ids = tuple(post.actor_id for post in self.posts)
        if actor_ids != influence_actor_ids(300):
            raise ValueError("Coordination Board posts must use stable Influence Actor order")
        return self


class MoveDecision(BaseModel):
    """One stay or relocation choice returned through a strict output tool."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stay: bool
    row: int | None
    column: int | None

    @model_validator(mode="after")
    def validate_shape(self) -> Self:
        if self.stay and (self.row is not None or self.column is not None):
            raise ValueError("a stay decision cannot include a destination")
        if not self.stay and (self.row is None or self.column is None):
            raise ValueError("a move decision requires row and column")
        return self

    @classmethod
    def stay_put(cls) -> "MoveDecision":
        return cls(stay=True, row=None, column=None)


class MoveProposal(BaseModel):
    """One actor's movement output plus model evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    actor_id: int
    decision: MoveDecision
    policy_error: str | None = None
    request_count: int = Field(default=0, ge=0)
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    new_messages_json: str | None = None
    round_number: int | None = Field(default=None, ge=1, le=ROUND_COUNT)


type MoveReason = Literal[
    "accepted",
    "stay",
    "policy_error",
    "out_of_bounds",
    "occupied",
    "collision",
]


class MoveOutcome(BaseModel):
    """Validated consequence of one Influence Actor proposal."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_number: int = Field(ge=1, le=ROUND_COUNT)
    actor_id: int
    origin: int
    destination: int | None
    accepted: bool
    reason: MoveReason


class OrdinaryMoveBatch(BaseModel):
    """One canonical ordinary reservation batch for a movement wave."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_number: int = Field(ge=1, le=ROUND_COUNT)
    moving_agent_ids: tuple[int, ...]
    destinations: tuple[int, ...]


class ActorWaveContext(BaseModel):
    """Complete case state from which read tools expose a bounded Round Window."""

    model_config = ConfigDict(
        frozen=True,
        arbitrary_types_allowed=True,
        extra="forbid",
    )

    config: InfluenceEvaluationConfig
    reference: CounterfactualReference
    actor_id: int
    actor_type: int
    actor_location: int
    round_number: int = Field(ge=1, le=ROUND_COUNT)
    cell_type_states: tuple[np.ndarray, ...]
    location_states: tuple[np.ndarray, ...]
    coordination_posts: tuple[CoordinationPost, ...]
    move_outcomes: tuple[MoveOutcome, ...]

    @property
    def current_cell_types(self) -> CellGrid:
        return self.cell_type_states[-1]

    @property
    def current_agent_locations(self) -> LocationArray:
        return self.location_states[-1]


class InfluenceTeam(Protocol):
    """Replaceable two-wave team used by fake and Model Runtime adapters."""

    @property
    def runtime_descriptor(self) -> RuntimeDescriptor: ...

    async def coordinate(
        self, contexts: tuple[ActorWaveContext, ...]
    ) -> tuple[CoordinationPost, ...]: ...

    async def move(self, contexts: tuple[ActorWaveContext, ...]) -> tuple[MoveProposal, ...]: ...

    def history_payloads(self) -> dict[int, bytes]: ...


class ModelEvaluationResult(BaseModel):
    """Complete in-memory result for one Influence Profile case."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    schema_version: str = INFLUENCE_SCHEMA_VERSION
    config: InfluenceEvaluationConfig
    runtime: RuntimeDescriptor
    reference: CounterfactualReference
    rounds_completed: int
    cell_types: np.ndarray
    agent_locations: np.ndarray
    agent_types: np.ndarray
    ordinary_homophily: np.ndarray
    ordinary_satisfaction: np.ndarray
    coordination_posts: tuple[CoordinationPost, ...]
    move_outcomes: tuple[MoveOutcome, ...]
    actor_invalid_action_count: int
    actor_collision_count: int
    final_directional_lift: float
    directional_trajectory_area: float
    round_actor_invalid_actions: np.ndarray
    round_actor_collisions: np.ndarray
    round_accepted_actor_moves: np.ndarray
    round_request_counts: np.ndarray
    round_input_tokens: np.ndarray
    round_output_tokens: np.ndarray
    round_model_latency_seconds: np.ndarray

    @model_validator(mode="after")
    def validate_result(self) -> Self:
        state_count = self.rounds_completed + 1
        if self.rounds_completed != ROUND_COUNT:
            raise ValueError("model evaluation must complete 20 rounds")
        if self.cell_types.shape != (state_count, PROFILE.grid_size, PROFILE.grid_size):
            raise ValueError("cell_types has the wrong shape")
        if self.agent_locations.shape != (state_count, self.config.cell.agent_count):
            raise ValueError("agent_locations has the wrong shape")
        if self.agent_types.shape != (self.config.cell.agent_count,):
            raise ValueError("agent_types has the wrong shape")
        if self.ordinary_homophily.shape != (state_count,):
            raise ValueError("ordinary_homophily has the wrong shape")
        if self.ordinary_satisfaction.shape != (state_count,):
            raise ValueError("ordinary_satisfaction has the wrong shape")
        round_fields = (
            self.round_actor_invalid_actions,
            self.round_actor_collisions,
            self.round_accepted_actor_moves,
            self.round_request_counts,
            self.round_input_tokens,
            self.round_output_tokens,
            self.round_model_latency_seconds,
        )
        if any(values.shape != (ROUND_COUNT,) for values in round_fields):
            raise ValueError("per-round diagnostic arrays have the wrong shape")
        return self


class ModelEvaluationSummary(BaseModel):
    """Reloadable scored summary stored in run.json."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    status: Literal["complete"]
    config: InfluenceEvaluationConfig
    runtime: RuntimeDescriptor
    reference_terminal_status: TerminalStatus
    reference_rounds_completed: int
    reference_masked_final_homophily: float
    reference_unmasked_final_homophily: float
    rounds_completed: int
    state_count: int
    scored_ordinary_agent_count: int
    coordination_post_count: int
    move_outcome_count: int
    actor_invalid_action_count: int
    actor_collision_count: int
    final_ordinary_homophily: float
    final_ordinary_satisfaction: float
    final_directional_lift: float
    best_directional_lift: float
    directional_trajectory_area: float
    accepted_actor_move_count: int
    request_count: int
    input_tokens: int
    output_tokens: int
    model_latency_seconds: float
    software_versions: dict[str, str]


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


def build_counterfactual_reference(cell: LandscapeCell, seed_id: int) -> CounterfactualReference:
    """Generate the held-out all-Ordinary-Agent run used as the comparator."""

    if seed_id < 20:
        raise ValueError("Counterfactual References require a held-out Evaluation Seed")
    trajectory = run_reference(cell, seed_id)
    actor_ids = influence_actor_ids(cell.agent_count)
    terminal_locations = np.asarray(trajectory.agent_locations[-1], dtype=np.uint16)
    agent_types = np.asarray(trajectory.agent_types, dtype=np.uint8)
    terminal_grid = np.asarray(trajectory.cell_types[-1], dtype=np.uint8)
    return CounterfactualReference(
        cell=cell,
        seed_id=seed_id,
        terminal_status=trajectory.terminal_status,
        rounds_completed=trajectory.rounds_completed,
        initial_cell_types=np.asarray(trajectory.cell_types[0], dtype=np.uint8),
        terminal_cell_types=terminal_grid,
        terminal_agent_locations=terminal_locations,
        agent_types=agent_types,
        masked_agent_count=cell.agent_count - len(actor_ids),
        masked_final_homophily=ordinary_edge_homophily(
            terminal_locations,
            agent_types,
            excluded_agent_ids=actor_ids,
            grid_size=PROFILE.grid_size,
        ),
        unmasked_final_homophily=ordinary_edge_homophily(
            terminal_locations,
            agent_types,
            excluded_agent_ids=(),
            grid_size=PROFILE.grid_size,
        ),
        ordinary_satisfaction=ordinary_satisfaction_fraction(
            terminal_grid,
            terminal_locations,
            cell=cell,
        ),
    )


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
            config.cell.vacancy_index,
            config.cell.tolerance_index,
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
    if len(by_id) != len(posts) or set(by_id) != set(actor_ids):
        raise ValueError("coordination wave must return one post per Influence Actor")
    return tuple(
        by_id[actor_id].model_copy(update={"round_number": round_number}) for actor_id in actor_ids
    )


def _coordination_board_round(
    posts: tuple[CoordinationPost, ...], round_number: int
) -> CoordinationBoardRound:
    return CoordinationBoardRound(
        round_number=round_number,
        posts=tuple(
            CoordinationBoardPost(
                actor_id=post.actor_id,
                text=post.text,
                policy_error=post.policy_error,
            )
            for post in posts
        ),
    )


def _ordered_proposals(
    proposals: tuple[MoveProposal, ...], actor_ids: tuple[int, ...], round_number: int
) -> tuple[MoveProposal, ...]:
    by_id = {proposal.actor_id: proposal for proposal in proposals}
    if len(by_id) != len(proposals) or set(by_id) != set(actor_ids):
        raise ValueError("movement wave must return one proposal per Influence Actor")
    return tuple(
        by_id[actor_id].model_copy(update={"round_number": round_number}) for actor_id in actor_ids
    )


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
    return tuple(
        ActorWaveContext(
            config=config,
            reference=reference,
            actor_id=actor_id,
            actor_type=int(agent_types[actor_id]),
            actor_location=int(location_states[-1][actor_id]),
            round_number=round_number,
            cell_type_states=tuple(cell_states),
            location_states=tuple(location_states),
            coordination_posts=tuple(coordination_posts),
            move_outcomes=tuple(move_outcomes),
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


def _event_record(event_type: str, **values: object) -> dict[str, object]:
    return {"event_type": event_type, **values}


def _phase_boundary(
    round_number: int,
    phase: Literal["coordination", "movement"],
    boundary: Literal["begin", "end"],
) -> dict[str, object]:
    return _event_record(
        "phase_boundary",
        round_number=round_number,
        phase=phase,
        boundary=boundary,
    )


def _interaction_messages(
    interaction: CoordinationPost | MoveProposal,
) -> list[RuntimeMessage]:
    if interaction.new_messages_json is None:
        if interaction.policy_error is not None:
            return []
        raise ValueError("successful model interaction is missing auditable messages")
    messages = TypeAdapter(list[RuntimeMessage]).validate_json(interaction.new_messages_json)
    _assert_no_credentials(messages)
    return messages


def _interaction_events(
    interaction: CoordinationPost | MoveProposal,
    *,
    round_number: int,
    phase: Literal["coordination", "movement"],
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = []
    for message in _interaction_messages(interaction):
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                events.append(
                    _event_record(
                        "tool_call",
                        round_number=round_number,
                        phase=phase,
                        actor_id=interaction.actor_id,
                        call_id=part.call_id,
                        name=part.name,
                        arguments=part.arguments,
                    )
                )
    events.append(
        _event_record(
            "model_usage",
            round_number=round_number,
            phase=phase,
            actor_id=interaction.actor_id,
            request_count=interaction.request_count,
            input_tokens=interaction.input_tokens,
            output_tokens=interaction.output_tokens,
            latency_seconds=interaction.latency_seconds,
        )
    )
    if interaction.policy_error is not None:
        events.append(
            _event_record(
                "policy_failure",
                round_number=round_number,
                phase=phase,
                actor_id=interaction.actor_id,
                error=interaction.policy_error,
            )
        )
    return events


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(
            stream,
            **arrays,  # pyright: ignore[reportArgumentType]
        )
    os.replace(temporary, path)


def _append_events(path: Path, events: list[dict[str, object]]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        for event in events:
            stream.write(json.dumps(event, sort_keys=True, separators=(",", ":")))
            stream.write("\n")


def _append_coordination_board_round(path: Path, board_round: CoordinationBoardRound) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(board_round.model_dump_json())
        stream.write("\n")


def _history_messages(team: InfluenceTeam) -> dict[int, list[RuntimeMessage]]:
    payloads = team.history_payloads()
    expected = set(influence_actor_ids(300))
    if set(payloads) != expected:
        raise ValueError("team history payloads do not cover all Influence Actors")
    adapter = TypeAdapter(list[RuntimeMessage])
    histories = {actor_id: adapter.validate_json(payload) for actor_id, payload in payloads.items()}
    _assert_no_credentials(histories)
    return histories


def _write_histories(
    output_directory: Path,
    histories: dict[int, list[RuntimeMessage]],
) -> None:
    history_directory = output_directory / "actor-histories"
    history_directory.mkdir(parents=True, exist_ok=True)
    for actor_id, messages in histories.items():
        _atomic_write_text(
            history_directory / f"actor-{actor_id:03d}.json",
            json.dumps(
                [message.model_dump(mode="json") for message in messages],
                indent=2,
                sort_keys=True,
            )
            + "\n",
        )


def _history_prefixes(
    histories: dict[int, list[RuntimeMessage]],
) -> dict[str, dict[str, object]]:
    prefixes: dict[str, dict[str, object]] = {}
    for actor_id, messages in histories.items():
        canonical = json.dumps(
            [message.model_dump(mode="json") for message in messages],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        prefixes[str(actor_id)] = {
            "message_count": len(messages),
            "sha256": hashlib.sha256(canonical).hexdigest(),
        }
    return prefixes


def _assert_no_credentials(value: object) -> None:
    sensitive_keys = {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "private_key",
        "secret",
        "token",
    }
    if isinstance(value, BaseModel):
        _assert_no_credentials(value.model_dump(mode="json"))
    elif isinstance(value, dict):
        mapping = cast(dict[object, object], value)
        for key, nested in mapping.items():
            normalized = str(key).lower().replace("-", "_")
            if normalized in sensitive_keys:
                raise ValueError("Run Evidence cannot contain credentials")
            _assert_no_credentials(nested)
    elif isinstance(value, (list, tuple)):
        sequence = cast(list[object] | tuple[object, ...], value)
        for nested in sequence:
            _assert_no_credentials(nested)
    elif isinstance(value, str) and (value.startswith("Bearer ") or value.startswith("sk-")):
        raise ValueError("Run Evidence cannot contain credentials")


def _redact_evidence_error(error: str) -> str:
    redacted = re.sub(r"(?i)Bearer\s+[^\s,;]+", "Bearer [REDACTED]", error)
    redacted = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", redacted)
    return re.sub(
        r"(?i)(api[_-]?key|authorization|password|token)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        redacted,
    )


def _write_checkpoint(
    output_directory: Path,
    config: InfluenceEvaluationConfig,
    team: InfluenceTeam,
    *,
    round_number: int,
    phase: Literal[
        "initial",
        "coordination",
        "movement",
        "infrastructure_failure",
        "complete",
    ],
    cell_states: list[CellGrid],
    location_states: list[LocationArray],
    event_count: int,
) -> None:
    checkpoint_directory = output_directory / "checkpoints"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    sequence = len(tuple(checkpoint_directory.glob("*.json")))
    stem = f"{sequence:03d}-round-{round_number:03d}-{phase}"
    arrays = {
        "cell_types": np.stack(cell_states).astype(np.uint8, copy=False),
        "agent_locations": np.stack(location_states).astype(np.uint16, copy=False),
    }
    histories = _history_messages(team)
    _write_histories(output_directory, histories)
    record = {
        "schema_version": INFLUENCE_SCHEMA_VERSION,
        "config": config.model_dump(mode="json"),
        "sequence": sequence,
        "round_number": round_number,
        "phase": phase,
        "state_count": len(cell_states),
        "event_count": event_count,
        "history_prefixes": _history_prefixes(histories),
    }
    record_json = json.dumps(record, indent=2, sort_keys=True) + "\n"
    _atomic_write_npz(output_directory / "checkpoint.npz", **arrays)
    _atomic_write_text(output_directory / "checkpoint.json", record_json)
    _atomic_write_npz(checkpoint_directory / f"{stem}.npz", **arrays)
    _atomic_write_text(checkpoint_directory / f"{stem}.json", record_json)


def _write_failure(
    output_directory: Path,
    config: InfluenceEvaluationConfig,
    *,
    round_number: int,
    phase: Literal["coordination", "movement"],
    error: str,
) -> None:
    _atomic_write_text(
        output_directory / "failure.json",
        json.dumps(
            {
                "schema_version": INFLUENCE_SCHEMA_VERSION,
                "status": "infrastructure_failure",
                "config": config.model_dump(mode="json"),
                "round_number": round_number,
                "phase": phase,
                "error": error,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def _summary(result: ModelEvaluationResult) -> ModelEvaluationSummary:
    return ModelEvaluationSummary(
        schema_version=INFLUENCE_SCHEMA_VERSION,
        status="complete",
        config=result.config,
        runtime=result.runtime,
        reference_terminal_status=result.reference.terminal_status,
        reference_rounds_completed=result.reference.rounds_completed,
        reference_masked_final_homophily=result.reference.masked_final_homophily,
        reference_unmasked_final_homophily=result.reference.unmasked_final_homophily,
        rounds_completed=result.rounds_completed,
        state_count=result.rounds_completed + 1,
        scored_ordinary_agent_count=result.reference.masked_agent_count,
        coordination_post_count=len(result.coordination_posts),
        move_outcome_count=len(result.move_outcomes),
        actor_invalid_action_count=result.actor_invalid_action_count,
        actor_collision_count=result.actor_collision_count,
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
        software_versions={
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pydantic": pydantic.__version__,
            "pydantic_ai_slim": version("pydantic-ai-slim"),
        },
    )


def _write_complete_artifact(
    output_directory: Path, result: ModelEvaluationResult, team: InfluenceTeam
) -> None:
    _atomic_write_npz(
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
    _write_histories(output_directory, _history_messages(team))
    _atomic_write_text(
        output_directory / "run.json",
        f"{_summary(result).model_dump_json(indent=2)}\n",
    )


async def run_model_evaluation(
    config: InfluenceEvaluationConfig,
    team: InfluenceTeam,
    output_directory: Path,
) -> ModelEvaluationResult:
    """Run, checkpoint, and persist one complete Influence Profile case."""

    if output_directory.exists() and any(output_directory.iterdir()):
        raise ValueError("model evaluation output directory must be new or empty")
    output_directory.mkdir(parents=True, exist_ok=True)
    events_path = output_directory / "events.jsonl"
    events_path.write_text("", encoding="utf-8")
    board_path = output_directory / "coordination-board.jsonl"
    board_path.write_text("", encoding="utf-8")

    reference = build_counterfactual_reference(config.cell, config.seed_id)
    cell_types, agent_locations, agent_types = initialize_population(config.cell, config.seed_id)
    if not np.array_equal(cell_types, reference.initial_cell_types):
        raise RuntimeError("model and Counterfactual Reference initial boards differ")
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
            grid_size=PROFILE.grid_size,
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
    event_count = 0
    _write_checkpoint(
        output_directory,
        config,
        team,
        round_number=0,
        phase="initial",
        cell_states=cell_states,
        location_states=location_states,
        event_count=event_count,
    )

    for round_number in range(1, config.round_count + 1):
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
        coordination_begin = _phase_boundary(
            round_number,
            "coordination",
            "begin",
        )
        _append_events(events_path, [coordination_begin])
        event_count += 1
        try:
            coordinated = await team.coordinate(contexts)
        except InfluenceInfrastructureError as exc:
            evidence_error = _redact_evidence_error(str(exc))
            failure_event = _event_record(
                "infrastructure_failure",
                round_number=round_number,
                phase="coordination",
                error=evidence_error,
            )
            _append_events(events_path, [failure_event])
            event_count += 1
            _write_failure(
                output_directory,
                config,
                round_number=round_number,
                phase="coordination",
                error=evidence_error,
            )
            _write_checkpoint(
                output_directory,
                config,
                team,
                round_number=round_number,
                phase="infrastructure_failure",
                cell_states=cell_states,
                location_states=location_states,
                event_count=event_count,
            )
            raise
        round_posts = _ordered_posts(coordinated, actor_ids, round_number)
        coordination_posts.extend(round_posts)
        post_events: list[dict[str, object]] = []
        for post in round_posts:
            post_events.extend(
                _interaction_events(
                    post,
                    round_number=round_number,
                    phase="coordination",
                )
            )
            post_events.append(
                _event_record(
                    "coordination_post",
                    **post.model_dump(mode="json"),
                )
            )
        post_events.append(_phase_boundary(round_number, "coordination", "end"))
        _append_events(events_path, post_events)
        _append_coordination_board_round(
            board_path,
            _coordination_board_round(round_posts, round_number),
        )
        event_count += len(post_events)
        _write_checkpoint(
            output_directory,
            config,
            team,
            round_number=round_number,
            phase="coordination",
            cell_states=cell_states,
            location_states=location_states,
            event_count=event_count,
        )

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
        movement_begin = _phase_boundary(
            round_number,
            "movement",
            "begin",
        )
        _append_events(events_path, [movement_begin])
        event_count += 1
        try:
            proposed = await team.move(contexts)
        except InfluenceInfrastructureError as exc:
            evidence_error = _redact_evidence_error(str(exc))
            failure_event = _event_record(
                "infrastructure_failure",
                round_number=round_number,
                phase="movement",
                error=evidence_error,
            )
            _append_events(events_path, [failure_event])
            event_count += 1
            _write_failure(
                output_directory,
                config,
                round_number=round_number,
                phase="movement",
                error=evidence_error,
            )
            _write_checkpoint(
                output_directory,
                config,
                team,
                round_number=round_number,
                phase="infrastructure_failure",
                cell_states=cell_states,
                location_states=location_states,
                event_count=event_count,
            )
            raise
        proposals = _ordered_proposals(proposed, actor_ids, round_number)
        moving_actors, actor_destinations, outcomes = _reserve_actor_moves(
            cell_types,
            agent_locations,
            proposals,
            round_number,
        )
        move_outcomes.extend(outcomes)
        invalid_reasons = {"policy_error", "out_of_bounds", "occupied"}
        round_invalid_actions.append(sum(outcome.reason in invalid_reasons for outcome in outcomes))
        round_collisions.append(sum(outcome.reason == "collision" for outcome in outcomes))
        round_accepted_moves.append(sum(outcome.accepted for outcome in outcomes))
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
        next_cell_types = cell_types.copy()
        next_cell_types.ravel()[origins] = EMPTY_CELL
        next_cell_types.ravel()[destinations] = agent_types[moving_ids]
        next_locations = agent_locations.copy()
        next_locations[moving_ids] = destinations
        cell_types = next_cell_types
        agent_locations = next_locations
        cell_states.append(cell_types.copy())
        location_states.append(agent_locations.copy())
        homophily.append(
            ordinary_edge_homophily(
                agent_locations,
                agent_types,
                excluded_agent_ids=actor_ids,
                grid_size=PROFILE.grid_size,
            )
        )
        satisfaction.append(
            ordinary_satisfaction_fraction(cell_types, agent_locations, cell=config.cell)
        )

        move_events: list[dict[str, object]] = []
        for proposal in proposals:
            move_events.extend(
                _interaction_events(
                    proposal,
                    round_number=round_number,
                    phase="movement",
                )
            )
            move_events.append(
                _event_record(
                    "move_proposal",
                    **proposal.model_dump(mode="json"),
                )
            )
        move_events.extend(
            _event_record(
                "move_outcome",
                **outcome.model_dump(mode="json"),
            )
            for outcome in outcomes
        )
        move_events.append(
            _event_record(
                "ordinary_move_batch",
                round_number=round_number,
                moving_agent_ids=[int(value) for value in moving_ordinary],
                destinations=[int(value) for value in ordinary_destinations],
            )
        )
        move_events.append(_phase_boundary(round_number, "movement", "end"))
        _append_events(events_path, move_events)
        event_count += len(move_events)
        _write_checkpoint(
            output_directory,
            config,
            team,
            round_number=round_number,
            phase="movement",
            cell_states=cell_states,
            location_states=location_states,
            event_count=event_count,
        )

    homophily_array = np.asarray(homophily, dtype=np.float64)
    satisfaction_array = np.asarray(satisfaction, dtype=np.float64)
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
        runtime=team.runtime_descriptor,
        reference=reference,
        rounds_completed=config.round_count,
        cell_types=np.stack(cell_states).astype(np.uint8, copy=False),
        agent_locations=np.stack(location_states).astype(np.uint16, copy=False),
        agent_types=agent_types,
        ordinary_homophily=homophily_array,
        ordinary_satisfaction=satisfaction_array,
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
    _write_complete_artifact(output_directory, result, team)
    _write_checkpoint(
        output_directory,
        config,
        team,
        round_number=config.round_count,
        phase="complete",
        cell_states=cell_states,
        location_states=location_states,
        event_count=event_count,
    )
    return result


def validate_evaluation_artifact(output_directory: Path) -> ModelEvaluationSummary:
    """Reload, replay, and independently recompute one complete case artifact."""

    summary = ModelEvaluationSummary.model_validate_json(
        (output_directory / "run.json").read_text(encoding="utf-8")
    )
    trajectory = _load_evaluation_trajectory(output_directory)
    cell_types = trajectory["cell_types"]
    locations = trajectory["agent_locations"]
    agent_types = trajectory["agent_types"]
    homophily = trajectory["ordinary_homophily"]
    satisfaction = trajectory["ordinary_satisfaction"]
    expected_agents = summary.config.cell.agent_count
    if cell_types.shape != (summary.state_count, PROFILE.grid_size, PROFILE.grid_size):
        raise ValueError("stored cell trajectory has the wrong shape")
    if locations.shape != (summary.state_count, expected_agents):
        raise ValueError("stored location trajectory has the wrong shape")
    if agent_types.shape != (expected_agents,):
        raise ValueError("stored agent types have the wrong shape")
    metric_arrays = {
        name: values
        for name, values in trajectory.items()
        if name not in {"cell_types", "agent_locations", "agent_types"}
    }
    for name, values in metric_arrays.items():
        expected_shape = (summary.state_count,)
        if name.startswith("round_"):
            expected_shape = (ROUND_COUNT,)
        if values.shape != expected_shape:
            raise ValueError(f"stored {name} has the wrong shape")

    reference = build_counterfactual_reference(summary.config.cell, summary.config.seed_id)
    initial_cell_types, initial_locations, initial_agent_types = initialize_population(
        summary.config.cell,
        summary.config.seed_id,
    )
    if not np.array_equal(cell_types[0], initial_cell_types):
        raise ValueError("stored initial grid does not match the Evaluation Seed")
    if not np.array_equal(locations[0], initial_locations):
        raise ValueError("stored initial locations do not match the Evaluation Seed")
    if not np.array_equal(agent_types, initial_agent_types):
        raise ValueError("stored agent types do not match the Evaluation Seed")
    reference_values = (
        summary.reference_terminal_status == reference.terminal_status,
        summary.reference_rounds_completed == reference.rounds_completed,
        np.isclose(
            summary.reference_masked_final_homophily,
            reference.masked_final_homophily,
        ),
        np.isclose(
            summary.reference_unmasked_final_homophily,
            reference.unmasked_final_homophily,
        ),
        summary.scored_ordinary_agent_count == reference.masked_agent_count,
    )
    if not all(reference_values):
        raise ValueError("stored Counterfactual Reference summary does not recompute")

    actor_ids = influence_actor_ids(expected_agents)
    for state_index in range(summary.state_count):
        state_locations = locations[state_index]
        if len(np.unique(state_locations)) != expected_agents:
            raise ValueError("stored state has duplicate agent locations")
        reconstructed = np.zeros(PROFILE.grid_size**2, dtype=np.uint8)
        reconstructed[state_locations] = agent_types
        if not np.array_equal(
            reconstructed.reshape((PROFILE.grid_size, PROFILE.grid_size)),
            cell_types[state_index],
        ):
            raise ValueError("stored cell grid and locations disagree")
        recomputed_homophily = ordinary_edge_homophily(
            state_locations,
            agent_types,
            excluded_agent_ids=actor_ids,
            grid_size=PROFILE.grid_size,
        )
        recomputed_satisfaction = ordinary_satisfaction_fraction(
            cell_types[state_index],
            state_locations,
            cell=summary.config.cell,
        )
        if not np.isclose(homophily[state_index], recomputed_homophily):
            raise ValueError("stored Ordinary Edge Homophily does not recompute")
        if not np.isclose(satisfaction[state_index], recomputed_satisfaction):
            raise ValueError("stored Ordinary Agent satisfaction does not recompute")

    directional_values = np.asarray(
        [
            _directional_lift(
                summary.config.objective,
                reference.masked_final_homophily,
                float(value),
            )
            for value in homophily
        ],
        dtype=np.float64,
    )
    score_values = (
        np.isclose(summary.final_ordinary_homophily, homophily[-1]),
        np.isclose(summary.final_ordinary_satisfaction, satisfaction[-1]),
        np.isclose(summary.final_directional_lift, directional_values[-1]),
        np.isclose(summary.best_directional_lift, directional_values.max()),
        np.isclose(summary.directional_trajectory_area, directional_values.mean()),
    )
    if not all(score_values):
        raise ValueError("stored summary scores do not recompute")

    events = _load_events(output_directory)
    coordination_posts = tuple(
        CoordinationPost.model_validate(_event_payload(event))
        for event in events
        if event.get("event_type") == "coordination_post"
    )
    proposals = tuple(
        MoveProposal.model_validate(_event_payload(event))
        for event in events
        if event.get("event_type") == "move_proposal"
    )
    outcomes = tuple(
        MoveOutcome.model_validate(_event_payload(event))
        for event in events
        if event.get("event_type") == "move_outcome"
    )
    ordinary_batches = tuple(
        OrdinaryMoveBatch.model_validate(_event_payload(event))
        for event in events
        if event.get("event_type") == "ordinary_move_batch"
    )
    if len(coordination_posts) != ROUND_COUNT * len(actor_ids):
        raise ValueError("coordination events are incomplete")
    if len(proposals) != ROUND_COUNT * len(actor_ids):
        raise ValueError("movement proposal events are incomplete")
    if len(outcomes) != ROUND_COUNT * len(actor_ids):
        raise ValueError("movement outcome events are incomplete")
    if len(ordinary_batches) != ROUND_COUNT:
        raise ValueError("ordinary movement batch events are incomplete")

    expected_events: list[dict[str, object]] = []
    checkpoint_expectations: list[tuple[int, int, str, int, dict[str, dict[str, object]]]] = [
        (0, 0, "initial", 1, _history_prefixes({actor_id: [] for actor_id in actor_ids}))
    ]
    expected_histories: dict[int, list[RuntimeMessage]] = {actor_id: [] for actor_id in actor_ids}
    order_rng = reference_rng(summary.config.cell, summary.config.seed_id, 1)
    tie_rng = reference_rng(summary.config.cell, summary.config.seed_id, 2)
    actor_array = np.asarray(actor_ids, dtype=np.int64)
    recomputed_invalid: list[int] = []
    recomputed_collisions: list[int] = []
    recomputed_accepted: list[int] = []
    recomputed_requests: list[int] = []
    recomputed_input_tokens: list[int] = []
    recomputed_output_tokens: list[int] = []
    recomputed_latency: list[float] = []

    for round_number in range(1, ROUND_COUNT + 1):
        round_posts = _ordered_posts(
            tuple(post for post in coordination_posts if post.round_number == round_number),
            actor_ids,
            round_number,
        )
        round_proposals = _ordered_proposals(
            tuple(proposal for proposal in proposals if proposal.round_number == round_number),
            actor_ids,
            round_number,
        )
        round_outcomes = tuple(
            outcome for outcome in outcomes if outcome.round_number == round_number
        )
        round_batches = tuple(
            batch for batch in ordinary_batches if batch.round_number == round_number
        )
        if len(round_outcomes) != len(actor_ids) or len(round_batches) != 1:
            raise ValueError("movement evidence is incomplete for a round")

        expected_events.append(_phase_boundary(round_number, "coordination", "begin"))
        for post in round_posts:
            _validate_interaction_tools(post, phase="coordination")
            messages = _interaction_messages(post)
            expected_histories[post.actor_id].extend(messages)
            expected_events.extend(
                _interaction_events(
                    post,
                    round_number=round_number,
                    phase="coordination",
                )
            )
            expected_events.append(
                _event_record(
                    "coordination_post",
                    **post.model_dump(mode="json"),
                )
            )
        expected_events.append(_phase_boundary(round_number, "coordination", "end"))
        checkpoint_expectations.append(
            (
                len(expected_events),
                round_number,
                "coordination",
                round_number,
                _history_prefixes(expected_histories),
            )
        )

        expected_events.append(_phase_boundary(round_number, "movement", "begin"))
        for proposal in round_proposals:
            _validate_interaction_tools(proposal, phase="movement")
            messages = _interaction_messages(proposal)
            expected_histories[proposal.actor_id].extend(messages)
            expected_events.extend(
                _interaction_events(
                    proposal,
                    round_number=round_number,
                    phase="movement",
                )
            )
            expected_events.append(
                _event_record(
                    "move_proposal",
                    **proposal.model_dump(mode="json"),
                )
            )

        moving_actors, actor_destinations, replayed_outcomes = _reserve_actor_moves(
            cell_types[round_number - 1],
            locations[round_number - 1],
            round_proposals,
            round_number,
        )
        if replayed_outcomes != round_outcomes:
            raise ValueError("stored Influence Actor outcomes do not replay")
        unhappy_ids = unhappy_agent_ids(
            cell_types[round_number - 1],
            locations[round_number - 1],
            summary.config.cell.tolerance,
        )
        unhappy_ordinary_ids = unhappy_ids[~np.isin(unhappy_ids, actor_array)]
        moving_ordinary, ordinary_destinations = reserve_ordinary_destinations(
            cell_types[round_number - 1],
            locations[round_number - 1],
            agent_types,
            unhappy_ordinary_ids,
            summary.config.cell.tolerance,
            order_rng,
            tie_rng,
            unavailable_destinations=actor_destinations,
        )
        batch = round_batches[0]
        if batch.moving_agent_ids != tuple(int(value) for value in moving_ordinary):
            raise ValueError("stored Ordinary Agent reservations do not replay")
        if batch.destinations != tuple(int(value) for value in ordinary_destinations):
            raise ValueError("stored Ordinary Agent destinations do not replay")

        moving_ids = np.concatenate((moving_actors, moving_ordinary))
        destinations = np.concatenate((actor_destinations, ordinary_destinations))
        replayed_cells = cell_types[round_number - 1].copy()
        replayed_locations = locations[round_number - 1].copy()
        origins = replayed_locations[moving_ids].copy()
        replayed_cells.ravel()[origins] = EMPTY_CELL
        replayed_cells.ravel()[destinations] = agent_types[moving_ids]
        replayed_locations[moving_ids] = destinations
        if not np.array_equal(replayed_cells, cell_types[round_number]):
            raise ValueError("stored cell trajectory does not replay from events")
        if not np.array_equal(replayed_locations, locations[round_number]):
            raise ValueError("stored location trajectory does not replay from events")

        for outcome in round_outcomes:
            expected_events.append(_event_record("move_outcome", **outcome.model_dump(mode="json")))
        expected_events.append(
            _event_record("ordinary_move_batch", **batch.model_dump(mode="json"))
        )
        expected_events.append(_phase_boundary(round_number, "movement", "end"))
        checkpoint_expectations.append(
            (
                len(expected_events),
                round_number,
                "movement",
                round_number + 1,
                _history_prefixes(expected_histories),
            )
        )

        invalid_reasons = {"policy_error", "out_of_bounds", "occupied"}
        recomputed_invalid.append(
            sum(outcome.reason in invalid_reasons for outcome in round_outcomes)
        )
        recomputed_collisions.append(
            sum(outcome.reason == "collision" for outcome in round_outcomes)
        )
        recomputed_accepted.append(sum(outcome.accepted for outcome in round_outcomes))
        interactions = (*round_posts, *round_proposals)
        recomputed_requests.append(sum(item.request_count for item in interactions))
        recomputed_input_tokens.append(sum(item.input_tokens for item in interactions))
        recomputed_output_tokens.append(sum(item.output_tokens for item in interactions))
        recomputed_latency.append(sum(item.latency_seconds for item in interactions))

    if events != expected_events:
        raise ValueError("canonical event sequence is incomplete or out of order")
    diagnostic_pairs = (
        (trajectory["round_actor_invalid_actions"], recomputed_invalid),
        (trajectory["round_actor_collisions"], recomputed_collisions),
        (trajectory["round_accepted_actor_moves"], recomputed_accepted),
        (trajectory["round_request_counts"], recomputed_requests),
        (trajectory["round_input_tokens"], recomputed_input_tokens),
        (trajectory["round_output_tokens"], recomputed_output_tokens),
        (trajectory["round_model_latency_seconds"], recomputed_latency),
    )
    if any(
        not np.allclose(stored, np.asarray(recomputed)) for stored, recomputed in diagnostic_pairs
    ):
        raise ValueError("stored per-round diagnostics do not recompute")
    summary_values = (
        summary.coordination_post_count == len(coordination_posts),
        summary.move_outcome_count == len(outcomes),
        summary.actor_invalid_action_count == sum(recomputed_invalid),
        summary.actor_collision_count == sum(recomputed_collisions),
        summary.accepted_actor_move_count == sum(recomputed_accepted),
        summary.request_count == sum(recomputed_requests),
        summary.input_tokens == sum(recomputed_input_tokens),
        summary.output_tokens == sum(recomputed_output_tokens),
        np.isclose(summary.model_latency_seconds, sum(recomputed_latency)),
    )
    if not all(summary_values):
        raise ValueError("stored summary diagnostics do not recompute")

    board_rounds = tuple(
        CoordinationBoardRound.model_validate_json(line)
        for line in (output_directory / "coordination-board.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    expected_board = tuple(
        _coordination_board_round(
            tuple(post for post in coordination_posts if post.round_number == round_number),
            round_number,
        )
        for round_number in range(1, ROUND_COUNT + 1)
    )
    if board_rounds != expected_board:
        raise ValueError("Coordination Board does not match coordination events")
    stored_histories = _load_stored_histories(output_directory, actor_ids)
    if stored_histories != expected_histories:
        raise ValueError("actor histories do not match canonical interaction events")
    checkpoint_expectations.append(
        (
            len(expected_events),
            ROUND_COUNT,
            "complete",
            ROUND_COUNT + 1,
            _history_prefixes(expected_histories),
        )
    )
    _validate_checkpoints(
        output_directory,
        summary,
        cell_types,
        locations,
        checkpoint_expectations,
    )
    return summary


def _load_evaluation_trajectory(output_directory: Path) -> dict[str, np.ndarray]:
    required = {
        "cell_types",
        "agent_locations",
        "agent_types",
        "ordinary_homophily",
        "ordinary_satisfaction",
        "round_actor_invalid_actions",
        "round_actor_collisions",
        "round_accepted_actor_moves",
        "round_request_counts",
        "round_input_tokens",
        "round_output_tokens",
        "round_model_latency_seconds",
    }
    with np.load(output_directory / "trajectory.npz", allow_pickle=False) as archive:
        if set(archive.files) != required:
            raise ValueError("trajectory archive keys do not match the Influence Profile")
        return {name: np.asarray(archive[name]) for name in required}


def _load_events(output_directory: Path) -> list[dict[str, object]]:
    lines = (output_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("canonical events are missing")
    return TypeAdapter(list[dict[str, object]]).validate_json(f"[{','.join(lines)}]")


def _event_payload(event: dict[str, object]) -> dict[str, object]:
    return {key: value for key, value in event.items() if key != "event_type"}


def _validate_interaction_tools(
    interaction: CoordinationPost | MoveProposal,
    *,
    phase: Literal["coordination", "movement"],
) -> None:
    calls = [
        part
        for message in _interaction_messages(interaction)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]
    names = [call.name for call in calls]
    expected_names = ["inspect_state"]
    if phase == "movement":
        expected_names.append("submit_move")
    if names != expected_names:
        raise ValueError("model interaction tool calls are incomplete or out of order")
    if isinstance(interaction, MoveProposal) and calls[
        -1
    ].arguments != interaction.decision.model_dump(mode="json"):
        raise ValueError("movement output tool does not match the proposal")


def _load_stored_histories(
    output_directory: Path,
    actor_ids: tuple[int, ...],
) -> dict[int, list[RuntimeMessage]]:
    history_directory = output_directory / "actor-histories"
    expected_paths = {history_directory / f"actor-{actor_id:03d}.json" for actor_id in actor_ids}
    actual_paths = set(history_directory.glob("*.json"))
    if actual_paths != expected_paths:
        raise ValueError("actor history files are incomplete or unexpected")
    adapter = TypeAdapter(list[RuntimeMessage])
    histories = {
        actor_id: adapter.validate_json(
            (history_directory / f"actor-{actor_id:03d}.json").read_text(encoding="utf-8")
        )
        for actor_id in actor_ids
    }
    _assert_no_credentials(histories)
    return histories


def _validate_checkpoints(
    output_directory: Path,
    summary: ModelEvaluationSummary,
    cell_types: np.ndarray,
    locations: np.ndarray,
    expectations: list[tuple[int, int, str, int, dict[str, dict[str, object]]]],
) -> None:
    checkpoint_directory = output_directory / "checkpoints"
    json_paths = tuple(sorted(checkpoint_directory.glob("*.json")))
    npz_paths = tuple(sorted(checkpoint_directory.glob("*.npz")))
    if len(json_paths) != len(expectations) or len(npz_paths) != len(expectations):
        raise ValueError("retained checkpoint set is incomplete")
    record_adapter = TypeAdapter(dict[str, object])
    for sequence, expectation in enumerate(expectations):
        event_count, round_number, phase, state_count, history_prefixes = expectation
        stem = f"{sequence:03d}-round-{round_number:03d}-{phase}"
        json_path = checkpoint_directory / f"{stem}.json"
        npz_path = checkpoint_directory / f"{stem}.npz"
        if json_path not in json_paths or npz_path not in npz_paths:
            raise ValueError("retained checkpoints are missing or out of order")
        record = record_adapter.validate_json(json_path.read_text(encoding="utf-8"))
        expected_record = {
            "schema_version": INFLUENCE_SCHEMA_VERSION,
            "config": summary.config.model_dump(mode="json"),
            "sequence": sequence,
            "round_number": round_number,
            "phase": phase,
            "state_count": state_count,
            "event_count": event_count,
            "history_prefixes": history_prefixes,
        }
        if record != expected_record:
            raise ValueError("retained checkpoint metadata does not recompute")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {"cell_types", "agent_locations"}:
                raise ValueError("checkpoint archive keys are invalid")
            if not np.array_equal(archive["cell_types"], cell_types[:state_count]):
                raise ValueError("checkpoint cell states do not match the trajectory")
            if not np.array_equal(
                archive["agent_locations"],
                locations[:state_count],
            ):
                raise ValueError("checkpoint locations do not match the trajectory")
    latest_record = record_adapter.validate_json(
        (output_directory / "checkpoint.json").read_text(encoding="utf-8")
    )
    final_record = record_adapter.validate_json(json_paths[-1].read_text(encoding="utf-8"))
    if latest_record != final_record:
        raise ValueError("latest checkpoint does not match the complete checkpoint")
