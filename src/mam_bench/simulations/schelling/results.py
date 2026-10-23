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
    null: Outcome
    controlled: Outcome

    def _points(self, outcome: Outcome) -> float:
        lift = outcome.homophily[-1] - self.reference.homophily[-1]
        return 400 * (-lift if self.config.objective == "integration" else lift)

    @computed_field
    @property
    def score(self) -> float:
        return self._points(self.controlled)

    @computed_field
    @property
    def null_score(self) -> float:
        return self._points(self.null)

    @computed_field
    @property
    def adjusted_score(self) -> float:
        return self.score - self.null_score
