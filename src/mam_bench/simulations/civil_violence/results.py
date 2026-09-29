"""Civil Violence state records and final participation/revolution scoring."""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field

from .settings import CivilViolenceSettings

type Coordinate = tuple[int, int]


@dataclass
class Citizen:
    agent_id: int
    location: Coordinate | None
    private_preference: float
    active: bool = False
    jail_remaining: int = 0


@dataclass
class Police:
    agent_id: int
    location: Coordinate


class Outcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    steps: int
    termination: Literal["revolution", "horizon"]
    participation: tuple[float, ...]

    @computed_field
    @property
    def time_to_revolution(self) -> int | None:
        return self.steps if self.termination == "revolution" else None


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    config: CivilViolenceSettings
    controlled_agent_ids: tuple[int, ...]
    scored_agent_ids: tuple[int, ...]
    reference: Outcome
    controlled: Outcome

    @computed_field
    @property
    def participation_points(self) -> float:
        direction = 1 if self.config.objective == "increase" else -1
        return (
            direction * 100 * (self.controlled.participation[-1] - self.reference.participation[-1])
        )

    @computed_field
    @property
    def revolution_points(self) -> int:
        direction = 1 if self.config.objective == "increase" else -1
        return (
            direction
            * 100
            * (
                (self.controlled.termination == "revolution")
                - (self.reference.termination == "revolution")
            )
        )

    @computed_field
    @property
    def score(self) -> float:
        return self.participation_points + self.revolution_points
