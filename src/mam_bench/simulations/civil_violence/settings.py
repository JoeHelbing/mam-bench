"""The selected binary Cascade profile's scientific parameters."""

from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CivilViolenceSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    board_size: int = Field(default=20, strict=True, ge=3, le=255)
    citizen_density: float = Field(default=0.7, ge=0, le=1)
    police_density: float = Field(default=0.074, ge=0, le=1)
    citizen_vision: int = Field(default=3, strict=True, ge=1)
    police_vision: int = Field(default=3, strict=True, ge=1)
    threshold: float = 3.66356
    private_preference_mean: float = 0.0
    private_preference_std: float = Field(default=1.0, ge=0)
    max_jail_term: int = Field(default=100, strict=True, ge=0)
    seed_id: int = Field(default=50, strict=True, ge=0)
    max_transitions: int = Field(default=30, strict=True, ge=1)
    controlled_agent_count: int = Field(default=16, strict=True, ge=1)

    @property
    def citizen_count(self) -> int:
        return round(self.board_size**2 * self.citizen_density)

    @property
    def police_count(self) -> int:
        return round(self.board_size**2 * self.police_density)

    @model_validator(mode="after")
    def validate_populations(self) -> Self:
        if self.citizen_density + self.police_density > 1:
            raise ValueError("citizen and police densities exceed board capacity")
        if self.citizen_count + self.police_count > self.board_size**2:
            raise ValueError("rounded citizen and police populations exceed board capacity")
        if self.citizen_count == 0:
            raise ValueError("at least one scored citizen is required")
        return self

    def validate_controlled_role(self, role: Literal["citizen", "police"]) -> None:
        population = self.citizen_count if role == "citizen" else self.police_count
        if self.controlled_agent_count > population:
            raise ValueError("controlled count exceeds the selected role population")
        if role == "citizen" and self.controlled_agent_count >= self.citizen_count:
            raise ValueError("at least one ordinary scored citizen must remain")
