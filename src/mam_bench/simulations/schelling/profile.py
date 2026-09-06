"""Fixed coordinates and constants for Schelling Reference Dataset v2."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Rational:
    """An exact fraction used by the simulation."""

    numerator: int
    denominator: int

    def __str__(self) -> str:
        return f"{self.numerator}/{self.denominator}"

    @classmethod
    def parse(cls, value: str) -> Rational:
        numerator, denominator = value.split("/")
        return cls(int(numerator), int(denominator))

    @property
    def filename_component(self) -> str:
        return f"{self.numerator}-of-{self.denominator}"


MASTER_SEED_WORDS = (0x4D414D42, 0x454E4348)
MAX_TRANSITIONS = 30
PROFILE_VERSION = "schelling-reference-v2"


@dataclass(frozen=True)
class LandscapeCell:
    """One board-size, tolerance, and vacancy coordinate."""

    board_size: int
    tolerance: Rational
    vacancy_fraction: Rational

    @property
    def vacancy_count(self) -> int:
        return (
            self.board_size**2
            * self.vacancy_fraction.numerator
            // self.vacancy_fraction.denominator
        )

    @property
    def agent_count(self) -> int:
        return self.board_size**2 - self.vacancy_count

    @property
    def cell_id(self) -> str:
        return (
            f"board-{self.board_size:03d}"
            f"__tolerance-{self.tolerance.filename_component}"
            f"__vacancy-{self.vacancy_fraction.filename_component}"
        )
