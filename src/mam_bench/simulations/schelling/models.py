"""Data contracts for the Schelling influence benchmark.

This module contains the immutable simulation state, actor inputs and outputs,
runtime protocol, and serialized result models shared by the Schelling runtime,
fixture loader, and developer utilities. Execution belongs in ``runtime.py``.
"""

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Annotated, Literal, Protocol

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator
from pydantic_ai_harness.memory import MemoryToolset

from mam_bench.benchmark import RuntimeInfo
from mam_bench.communication import CompletedWork
from mam_bench.usage import AgentSessionUsage, session_usage_data

from .profile import LandscapeCell
from .reference import CellGrid, LocationArray, TerminalStatus


class SteeringObjective(StrEnum):
    """Assigned direction for Ordinary Edge Homophily."""

    INTEGRATION = "integration"
    SEGREGATION = "segregation"


@dataclass(frozen=True)
class InfluenceEvaluationConfig:
    """The fixed Schelling case and steering objective for one model run."""

    cell: LandscapeCell
    seed_id: int
    objective: SteeringObjective


@dataclass(frozen=True)
class CounterfactualReference:
    """Same-seed all-Ordinary-Agent endpoint and masked comparator."""

    cell: LandscapeCell
    seed_id: int
    terminal_status: TerminalStatus
    rounds_completed: int
    initial_cell_types: np.ndarray
    initial_agent_locations: np.ndarray
    terminal_cell_types: np.ndarray
    terminal_agent_locations: np.ndarray
    agent_types: np.ndarray
    masked_agent_count: int
    masked_final_homophily: float
    unmasked_final_homophily: float
    ordinary_satisfaction: float


@dataclass(frozen=True)
class BoardCoordinate:
    """One absolute row and column on the toroidal board."""

    row: int
    column: int


class NeighborKind(StrEnum):
    """Actor-visible classification of one neighboring cell."""

    VACANT = "vacant"
    ORDINARY_AGENT = "ordinary-agent"
    INFLUENCE_ACTOR = "influence-actor"


@dataclass(frozen=True)
class NeighborObservation:
    """One cell in a frozen radius-one actor observation."""

    location: BoardCoordinate
    kind: NeighborKind
    agent_type: int | None = None
    actor_id: int | None = None


@dataclass(frozen=True)
class ActorTurnContext:
    """The complete Schelling state disclosed at the start of one actor turn."""

    config: InfluenceEvaluationConfig
    actor_id: int
    actor_type: int
    actor_location: BoardCoordinate
    round_number: int
    horizon: int
    objective: SteeringObjective
    reference_homophily: float
    current_homophily: float
    remaining_unreserved_vacancies: int
    neighborhood: tuple[NeighborObservation, ...]


class AppendMemory(BaseModel):
    """Append content to one Harness-permitted actor notebook file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["append"]
    content: str = Field(min_length=1)
    file: str = "MEMORY.md"

    @field_validator("content")
    @classmethod
    def require_nonblank_content(cls, content: str) -> str:
        if not content.strip():
            raise ValueError("memory append content must not be blank")
        return content


class ReplaceMemory(BaseModel):
    """Replace one unique passage in an actor notebook file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["replace"]
    old_text: str = Field(min_length=1)
    content: str
    file: str = "MEMORY.md"


class DeleteMemory(BaseModel):
    """Delete one permitted non-main actor notebook file."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    operation: Literal["delete"]
    file: str


type ActorMemoryMutation = Annotated[
    AppendMemory | ReplaceMemory | DeleteMemory,
    Field(discriminator="operation"),
]


class SubmitMove(BaseModel):
    """One terminal v2 request to move the calling Influence Actor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    row: int = Field(strict=True, ge=0, lt=20)
    column: int = Field(strict=True, ge=0, lt=20)
    memory: ActorMemoryMutation | None = None


class Stay(BaseModel):
    """One terminal v2 request to leave the calling Influence Actor in place."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    memory: ActorMemoryMutation | None = None


type ActorTerminalAction = SubmitMove | Stay


@dataclass(frozen=True)
class ActorTurnResult:
    """One Influence Actor's terminal result from the turn-oriented team seam."""

    actor_id: int
    action: ActorTerminalAction
    accepted_call_count: int = 0
    policy_rejections: tuple[str, ...] = ()
    memory_operation_count: int = 0
    forced_stay: bool = False
    latency_seconds: float = 0.0


@dataclass(frozen=True)
class UnverifiedPost:
    """Free-form actor-authored content in the Public Document."""

    text: str


@dataclass(frozen=True)
class AuthoritativeRecord:
    """Runtime-authored fact in the Public Document."""

    text: str


type PublicDocumentRecord = UnverifiedPost | AuthoritativeRecord


class DocumentRead(BaseModel):
    """Rendered unread Public Document records returned to one actor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str
    more_available: bool


class PostReceipt(BaseModel):
    """Authoritative acknowledgement of one appended actor post."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sequence: int


class ActorTurnCoordinator(Protocol):
    """Schelling-owned public operations available during one actor turn."""

    async def read_document(self, actor_id: int) -> DocumentRead: ...

    async def post_message(
        self,
        actor_id: int,
        text: str,
        memory: ActorMemoryMutation | None = None,
        notebook: ActorNotebook | None = None,
    ) -> PostReceipt: ...

    async def commit_terminal(
        self,
        actor_id: int,
        action: ActorTerminalAction,
        notebook: ActorNotebook,
    ) -> ActorTerminalAction: ...

    async def start_round(self, round_number: int, cell_types: CellGrid) -> None: ...

    async def remaining_unreserved_vacancies(self) -> int: ...

    async def reserved_actor_moves(self) -> tuple[tuple[int, int], ...]: ...


class ActorNotebook(Protocol):
    """Minimal action-bound notebook interface used by Schelling commits."""

    async def write(
        self,
        content: str,
        *,
        file: str = "MEMORY.md",
        old_text: str | None = None,
    ) -> object: ...

    async def delete(self, file: str) -> object: ...


@dataclass
class ActorTurnDependencies:
    """Typed Schelling dependencies available throughout one v2 actor turn."""

    context: ActorTurnContext
    coordinator: ActorTurnCoordinator
    notebook: MemoryToolset[ActorTurnDependencies]


type MoveReason = Literal[
    "accepted",
    "stay",
    "policy_error",
]


@dataclass(frozen=True)
class MoveOutcome:
    """The simulation's authoritative result for one actor proposal."""

    round_number: int
    actor_id: int
    origin: int
    destination: int | None
    accepted: bool
    reason: MoveReason


@dataclass(frozen=True)
class StagedRoundResult:
    """One settled v2 Staged Round in authoritative completion order."""

    actor_admission_order: tuple[int, ...]
    actor_turns: tuple[CompletedWork[int, ActorTurnResult], ...]
    actor_move_outcomes: tuple[MoveOutcome, ...]
    moving_actor_ids: NDArray[np.int64]
    actor_destinations: LocationArray
    moving_ordinary_agent_ids: NDArray[np.int64]
    ordinary_destinations: LocationArray
    cell_types: CellGrid
    agent_locations: LocationArray


class InfluenceTeam(Protocol):
    """Turn-oriented team seam shared by scripted and PydanticAI v2 actors."""

    @property
    def runtime_info(self) -> RuntimeInfo: ...

    async def run_turn(
        self,
        context: ActorTurnContext,
        coordinator: ActorTurnCoordinator,
    ) -> ActorTurnResult: ...


class EvaluationInfluenceTeam(InfluenceTeam, Protocol):
    """Turn-oriented team that exposes aggregate usage for a full evaluation."""

    @property
    def usage(self) -> AgentSessionUsage: ...


@dataclass(frozen=True)
class EvaluationUsage:
    """Actor calls and policy retries paired with normalized session usage."""

    session: AgentSessionUsage = field(default_factory=AgentSessionUsage)
    accepted_calls: int = 0
    policy_rejections: int = 0
    memory_operations: int = 0

    def as_dict(self) -> dict[str, object]:
        """Keep the flat artifact schema independent of native usage internals."""
        return {
            **session_usage_data(self.session, details_as_pairs=True),
            "accepted_calls": self.accepted_calls,
            "policy_rejections": self.policy_rejections,
            "memory_operations": self.memory_operations,
        }


@dataclass(frozen=True)
class RoundUsage(EvaluationUsage):
    """Evaluation usage accumulated during one round."""


@dataclass(frozen=True)
class ModelEvaluationResult:
    """In-memory result of one Schelling benchmark run."""

    config: InfluenceEvaluationConfig
    runtime: RuntimeInfo
    reference: CounterfactualReference
    rounds_completed: int
    cell_types: np.ndarray
    agent_locations: np.ndarray
    agent_types: np.ndarray
    ordinary_homophily: np.ndarray
    ordinary_satisfaction: np.ndarray
    actor_invalid_action_count: int
    final_directional_lift: float
    directional_trajectory_area: float
    round_actor_invalid_actions: np.ndarray
    round_accepted_actor_moves: np.ndarray
    round_request_counts: np.ndarray
    round_input_tokens: np.ndarray
    round_output_tokens: np.ndarray
    round_model_latency_seconds: np.ndarray
    simulation_id: str = "schelling-influence-pilot-v1"
    simulation_version: str = "schelling-influence-v2"
    best_directional_lift: float = 0.0
    usage: EvaluationUsage = field(default_factory=EvaluationUsage)
    round_usage: tuple[RoundUsage, ...] = ()


class ModelEvaluationSummary(BaseModel):
    """Small JSON summary saved as ``run.json``."""

    simulation_id: str
    simulation_version: str
    config: InfluenceEvaluationConfig
    runtime: RuntimeInfo
    reference_terminal_status: TerminalStatus
    reference_rounds_completed: int
    reference_final_homophily: float
    rounds_completed: int
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
    usage: EvaluationUsage
    stochastic_trial_count: int = 1
    confidence_interval: None = None

    @field_serializer("usage")
    def serialize_usage(self, usage: EvaluationUsage) -> dict[str, object]:
        return usage.as_dict()
