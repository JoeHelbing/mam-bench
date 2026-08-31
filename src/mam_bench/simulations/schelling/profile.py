"""Frozen scientific contract for the first Schelling Reference Landscape."""

import math
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Rational(BaseModel):
    """A normalized exact fraction between zero and one."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    numerator: int = Field(ge=0)
    denominator: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_normalized_fraction(self) -> Self:
        if self.numerator > self.denominator:
            raise ValueError("fraction must be between zero and one")
        if math.gcd(self.numerator, self.denominator) != 1:
            raise ValueError("fraction must be normalized")
        return self

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"


TOLERANCES: tuple[Rational, ...] = tuple(
    Rational(numerator=numerator, denominator=denominator)
    for numerator, denominator in (
        (0, 1),
        (1, 8),
        (1, 7),
        (1, 6),
        (1, 5),
        (1, 4),
        (2, 7),
        (1, 3),
        (3, 8),
        (2, 5),
        (3, 7),
        (1, 2),
        (4, 7),
        (3, 5),
        (5, 8),
        (2, 3),
        (5, 7),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 8),
        (1, 1),
    )
)

VACANCY_FRACTIONS: tuple[Rational, ...] = tuple(
    Rational(numerator=numerator, denominator=denominator)
    for numerator, denominator in (
        (1, 10),
        (3, 20),
        (1, 5),
        (1, 4),
        (3, 10),
        (7, 20),
        (2, 5),
    )
)
VACANCY_COUNTS: tuple[int, ...] = (40, 60, 80, 100, 120, 140, 160)
LANDSCAPE_SEED_IDS: tuple[int, ...] = tuple(range(20))
LANDSCAPE_CELL_COUNT = len(TOLERANCES) * len(VACANCY_COUNTS)
MASTER_SEED_WORDS: tuple[int, int] = (0x4D414D42, 0x454E4348)


class LandscapeCell(BaseModel):
    """One exact tolerance-vacancy coordinate in the Reference Landscape."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tolerance_index: int = Field(ge=0, lt=len(TOLERANCES))
    vacancy_index: int = Field(ge=0, lt=len(VACANCY_COUNTS))
    tolerance: Rational
    vacancy_fraction: Rational
    vacancy_count: int

    @model_validator(mode="after")
    def validate_coordinate(self) -> Self:
        if self.tolerance != TOLERANCES[self.tolerance_index]:
            raise ValueError("tolerance does not match tolerance_index")
        if self.vacancy_fraction != VACANCY_FRACTIONS[self.vacancy_index]:
            raise ValueError("vacancy_fraction does not match vacancy_index")
        if self.vacancy_count != VACANCY_COUNTS[self.vacancy_index]:
            raise ValueError("vacancy_count does not match vacancy_index")
        return self

    @property
    def agent_count(self) -> int:
        return 400 - self.vacancy_count

    @property
    def cell_id(self) -> str:
        return f"t{self.tolerance_index:02d}-v{self.vacancy_index:02d}"


class ReferenceProfile(BaseModel):
    """Versioned fixed mechanics and coordinates for generation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_version: str = "schelling-reference-v1"
    grid_size: int = 20
    max_transitions: int = 500
    group_count: int = 2
    neighborhood_radius: int = 1
    topology: str = "toroidal"
    update_rule: str = "staged-sequential-reservation"
    distance: str = "toroidal-chebyshev"
    rng: str = "numpy-pcg64-seedsequence"
    tie_draw_rule: str = "draw-only-for-multiple-nearest"
    master_seed_words: tuple[int, int] = MASTER_SEED_WORDS
    tolerances: tuple[Rational, ...] = TOLERANCES
    vacancy_fractions: tuple[Rational, ...] = VACANCY_FRACTIONS
    vacancy_counts: tuple[int, ...] = VACANCY_COUNTS
    landscape_seed_ids: tuple[int, ...] = LANDSCAPE_SEED_IDS


PROFILE = ReferenceProfile()


def landscape_cell(tolerance_index: int, vacancy_index: int) -> LandscapeCell:
    """Build and validate one canonical landscape coordinate."""

    return LandscapeCell(
        tolerance_index=tolerance_index,
        vacancy_index=vacancy_index,
        tolerance=TOLERANCES[tolerance_index],
        vacancy_fraction=VACANCY_FRACTIONS[vacancy_index],
        vacancy_count=VACANCY_COUNTS[vacancy_index],
    )


def landscape_cells() -> tuple[LandscapeCell, ...]:
    """Return all cells in tolerance-major, vacancy-minor order."""

    return tuple(
        landscape_cell(tolerance_index, vacancy_index)
        for tolerance_index in range(len(TOLERANCES))
        for vacancy_index in range(len(VACANCY_COUNTS))
    )
