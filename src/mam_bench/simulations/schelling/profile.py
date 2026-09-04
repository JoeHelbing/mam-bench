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


TOLERANCES = tuple(
    Rational(numerator, denominator)
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
BOARD_SIZES = (20, 60, 100)
VACANCY_FRACTIONS = tuple(
    Rational(numerator, denominator)
    for numerator, denominator in (
        (1, 100),
        (1, 40),
        (1, 20),
        (1, 10),
        (1, 5),
        (1, 4),
        (3, 10),
        (2, 5),
        (1, 2),
    )
)
LANDSCAPE_SEED_IDS = tuple(range(50))
LANDSCAPE_CELL_COUNT = len(BOARD_SIZES) * len(TOLERANCES) * len(VACANCY_FRACTIONS)
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


def landscape_cells() -> tuple[LandscapeCell, ...]:
    return tuple(
        LandscapeCell(board_size, tolerance, vacancy)
        for board_size in BOARD_SIZES
        for tolerance in TOLERANCES
        for vacancy in VACANCY_FRACTIONS
    )
