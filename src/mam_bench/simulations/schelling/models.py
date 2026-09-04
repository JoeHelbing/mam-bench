"""Data contracts for the Schelling influence benchmark.

This module contains the immutable simulation state, actor inputs and outputs,
runtime protocol, and serialized result models shared by the Schelling runtime,
fixture loader, and developer utilities. Execution belongs in ``runtime.py``.
"""

from dataclasses import dataclass
from enum import StrEnum
from typing import Literal, Protocol, Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from mam_bench.benchmark import RuntimeInfo

from .profile import LandscapeCell
from .reference import CellGrid, LocationArray, TerminalStatus

COORDINATION_PHASE_CODE = 0
MOVEMENT_PHASE_CODE = 1


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
class CoordinationPost:
    """One Influence Actor's public message for a coordination wave."""

    actor_id: int
    text: str
    policy_error: str | None = None
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    round_number: int = 0


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
    def stay_put(cls) -> MoveDecision:
        return cls(stay=True, row=None, column=None)


@dataclass(frozen=True)
class MoveProposal:
    """One Influence Actor's requested movement and model-call accounting."""

    actor_id: int
    decision: MoveDecision
    policy_error: str | None = None
    request_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_seconds: float = 0.0
    round_number: int = 0


type MoveReason = Literal[
    "accepted",
    "stay",
    "policy_error",
    "out_of_bounds",
    "occupied",
    "collision",
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
class ActorWaveContext:
    """One actor's bounded view of the current evaluation round."""

    config: InfluenceEvaluationConfig
    reference: CounterfactualReference
    actor_id: int
    actor_type: int
    actor_location: int
    round_number: int
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
    """Replaceable two-wave team used by fake and PydanticAI runtimes."""

    @property
    def runtime_info(self) -> RuntimeInfo: ...

    async def coordinate(
        self, contexts: tuple[ActorWaveContext, ...]
    ) -> tuple[CoordinationPost, ...]: ...

    async def move(self, contexts: tuple[ActorWaveContext, ...]) -> tuple[MoveProposal, ...]: ...


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


class ModelEvaluationSummary(BaseModel):
    """Small JSON summary saved as ``run.json``."""

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


class StateRequest(BaseModel):
    """One bounded global-state selection passed to ``inspect_state``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    board_round_offset: Literal[0, 1, 2]
    coordination_round_offset: Literal[0, 1, 2]
    include_reference: bool


@dataclass
class ActorRunDependencies:
    """Typed state available to agent instructions, tools, and validators."""

    context: ActorWaveContext
    phase_code: int
    inspection_count: int = 0
