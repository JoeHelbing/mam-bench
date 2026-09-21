"""Schelling outcomes and its signed, unclamped evaluation score."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, computed_field

from .settings import SchellingSettings


class Outcome(BaseModel):
    model_config = ConfigDict(frozen=True)
    steps: int
    termination: Literal["equilibrium", "blocked", "horizon"]
    homophily: tuple[float, ...]
    satisfaction: tuple[float, ...]


class EvaluationResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    config: SchellingSettings
    controlled_agent_ids: tuple[int, ...]
    scored_agent_ids: tuple[int, ...]
    reference: Outcome
    controlled: Outcome

    @computed_field
    @property
    def score(self) -> float:
        lift = self.controlled.homophily[-1] - self.reference.homophily[-1]
        return 400 * (-lift if self.config.objective == "integration" else lift)
