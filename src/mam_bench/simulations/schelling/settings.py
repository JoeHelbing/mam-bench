"""Validated parameters for a configured Schelling trial."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .profile import LandscapeCell, Rational


class SchellingSettings(BaseModel):
    """Simulation settings shared by the controlled and paired ordinary worlds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    board_size: int = Field(default=20, strict=True, ge=3, le=255)
    tolerance: str = "3/4"
    vacancy_fraction: str = "1/4"
    seed_id: int = Field(default=50, strict=True, ge=0)
    max_transitions: int = Field(default=30, strict=True, ge=1)
    vision_radius: int = Field(default=3, strict=True, ge=1)
    controlled_agent_count: int = Field(default=16, strict=True, ge=2)
    objective: Literal["integration", "segregation"] = "integration"

    @field_validator("tolerance", "vacancy_fraction")
    @classmethod
    def require_unit_fraction(cls, value: str) -> str:
        """Retain the exact representation because it participates in RNG seeding."""
        try:
            fraction = Rational.parse(value)
        except (ValueError, TypeError) as error:
            raise ValueError("must be an exact rational string such as '3/4'") from error
        if fraction.denominator <= 0 or not 0 <= fraction.numerator <= fraction.denominator:
            raise ValueError("must be a fraction between 0 and 1 with a positive denominator")
        if fraction.denominator > 1_000_000_000:
            raise ValueError("fraction denominator must be at most 1000000000 for exact arithmetic")
        return value

    @model_validator(mode="after")
    def validate_population_and_vision(self) -> Self:
        cell = self.cell
        if cell.vacancy_fraction.numerator == cell.vacancy_fraction.denominator:
            raise ValueError("vacancy_fraction must be less than 1")
        if cell.vacancy_count < 1:
            raise ValueError("the board must contain at least one vacancy")
        if cell.agent_count % 2:
            raise ValueError("the population must be even for equal type counts")
        if self.controlled_agent_count % 2:
            raise ValueError("controlled_agent_count must be even for equal type counts")
        if cell.agent_count < self.controlled_agent_count + 2:
            raise ValueError("the population must include at least two ordinary agents")
        # Each ordinary identity occurs in four wrapped 2x2 windows. More than
        # size**2 / 4 identities forces a shared window, hence a scored edge.
        if cell.agent_count - self.controlled_agent_count <= self.board_size**2 // 4:
            raise ValueError(
                "ordinary agents must occupy more than one quarter of the board "
                "to guarantee a defined homophily score"
            )
        if 2 * self.vision_radius + 1 > self.board_size:
            raise ValueError("vision diameter must not exceed board_size")
        return self

    @property
    def cell(self) -> LandscapeCell:
        return LandscapeCell(
            board_size=self.board_size,
            tolerance=Rational.parse(self.tolerance),
            vacancy_fraction=Rational.parse(self.vacancy_fraction),
        )

    @property
    def controlled_agent_ids(self) -> tuple[int, ...]:
        per_type = self.controlled_agent_count // 2
        second_type = self.cell.agent_count // 2
        return (*range(per_type), *range(second_type, second_type + per_type))
