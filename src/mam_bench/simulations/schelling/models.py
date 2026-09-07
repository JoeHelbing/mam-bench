"""Scientific records and tool arguments; behavior lives in the domain classes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from mam_bench.model import RuntimeInfo

from .profile import LandscapeCell
from .reference import BoardCoordinate, CellGrid, LocationArray, TerminalStatus


class SteeringObjective(StrEnum):
    INTEGRATION = "integration"
    SEGREGATION = "segregation"


@dataclass(frozen=True)
class EvaluationConfig:
    cell: LandscapeCell
    seed_id: int
    objective: SteeringObjective = SteeringObjective.INTEGRATION
    max_transitions: int = 30
    vision_radius: int = 3
    controlled_agent_count: int = 16


@dataclass(frozen=True)
class Trajectory:
    """Initial state and every settled state, with stable identity ordering."""

    cell: LandscapeCell
    seed_id: int
    cell_types: CellGrid
    agent_locations: LocationArray
    agent_types: CellGrid
    terminal_status: TerminalStatus
    rounds_completed: int

    @property
    def trajectory_length(self) -> int:
        return self.rounds_completed + 1


class NeighborKind(StrEnum):
    VACANT = "vacant"
    ORDINARY_AGENT = "ordinary-agent"
    MODEL_CONTROLLED_AGENT = "model-controlled-agent"


@dataclass(frozen=True)
class NeighborObservation:
    location: BoardCoordinate
    kind: NeighborKind
    agent_type: int | None = None
    agent_id: int | None = None


@dataclass(frozen=True)
class Observation:
    agent_id: int
    agent_type: int
    location: BoardCoordinate
    round_number: int
    horizon: int
    objective: SteeringObjective
    reference_homophily: float
    current_homophily: float
    remaining_vacancies: int
    neighborhood: tuple[NeighborObservation, ...]


class Move(BaseModel):
    """Request one vacancy; SchellingSim validates and reserves it."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    row: int = Field(strict=True, ge=0)
    column: int = Field(strict=True, ge=0)


class Stay(BaseModel):
    """Finish the turn without moving."""

    model_config = ConfigDict(frozen=True, extra="forbid")


type Action = Move | Stay


@dataclass(frozen=True)
class EvaluationResult:
    config: EvaluationConfig
    runtime: RuntimeInfo
    ordinary: Trajectory
    model_controlled: Trajectory
    controlled_agent_ids: tuple[int, ...]
    reference_homophily: float
    ordinary_homophily: tuple[float, ...]
    ordinary_satisfaction: tuple[float, ...]
    final_directional_lift: float
    best_directional_lift: float
    directional_trajectory_area: float


@dataclass(frozen=True)
class RoundResult:
    """Accepted destinations in reservation order and detached settled state."""

    admission_order: tuple[int, ...]
    controlled_moves: tuple[tuple[int, int], ...]
    ordinary_moves: tuple[tuple[int, int], ...]
    cell_types: CellGrid
    agent_locations: LocationArray
