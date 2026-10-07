"""The complete configuration of one paired binary Cascade evaluation."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CivilViolenceSettings(BaseModel):
    model_config = ConfigDict(
        frozen=True, extra="forbid", allow_inf_nan=False, hide_input_in_errors=True
    )

    simulation: Literal["civil-violence"]
    controlled_role: Literal["citizen", "police"]
    objective: Literal["increase", "decrease"]
    board_size: int = Field(strict=True, ge=3, le=255)
    citizen_density: float = Field(ge=0, le=1)
    police_density: float = Field(ge=0, le=1)
    vision_radius: int = Field(strict=True, ge=1)
    threshold: float
    private_preference_mean: float
    private_preference_std: float = Field(ge=0)
    max_jail_term: int = Field(strict=True, ge=0)
    seed: int = Field(strict=True, ge=0)
    max_steps: int = Field(strict=True, ge=1)
    controlled_agent_count: Literal[16]

    @property
    def citizen_count(self) -> int:
        return round(self.board_size**2 * self.citizen_density)

    @property
    def police_count(self) -> int:
        return round(self.board_size**2 * self.police_density)

    @model_validator(mode="after")
    def populations(self) -> Self:
        if self.citizen_density + self.police_density > 1:
            raise ValueError("citizen and police densities exceed board capacity")
        if self.citizen_count + self.police_count > self.board_size**2:
            raise ValueError("rounded populations exceed board capacity")
        population = self.citizen_count if self.controlled_role == "citizen" else self.police_count
        if population < self.controlled_agent_count:
            raise ValueError("selected role must contain at least 16 agents")
        excluded = self.controlled_agent_count if self.controlled_role == "citizen" else 0
        if self.citizen_count <= excluded:
            raise ValueError("at least one ordinary scored citizen must remain")
        return self
