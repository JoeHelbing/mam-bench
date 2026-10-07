"""The complete configuration of one paired Schelling evaluation."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class SchellingSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    simulation: Literal["schelling"]
    board_size: int = Field(strict=True, ge=3, le=255)
    tolerance: float = Field(strict=True, ge=0, le=1)
    vacancy_fraction: float = Field(strict=True, ge=0, le=1)
    seed: int = Field(strict=True, ge=0)
    max_steps: Literal[30]
    vision_radius: int = Field(strict=True, ge=1)
    controlled_agent_count: int = Field(strict=True, ge=2)
    objective: Literal["integration", "segregation"]

    @property
    def vacancy_count(self) -> int:
        return round(self.board_size**2 * self.vacancy_fraction)

    @property
    def agent_count(self) -> int:
        return self.board_size**2 - self.vacancy_count

    @property
    def controlled_agent_ids(self) -> tuple[int, ...]:
        count = self.controlled_agent_count // 2
        half = self.agent_count // 2
        return (*range(count), *range(half, half + count))

    @model_validator(mode="after")
    def population_and_vision(self) -> Self:
        if not 0 < self.vacancy_count < self.board_size**2:
            raise ValueError("the board must have vacancies and agents")
        if self.agent_count % 2 or self.controlled_agent_count % 2:
            raise ValueError("population and controlled count must be even for equal type counts")
        if self.agent_count - self.controlled_agent_count <= self.board_size**2 // 4:
            raise ValueError("ordinary agents must occupy over one quarter of the board")
        if 2 * self.vision_radius + 1 > self.board_size:
            raise ValueError("vision diameter must not exceed board_size")
        return self
