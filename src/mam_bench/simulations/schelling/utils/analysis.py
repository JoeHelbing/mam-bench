"""Reduce full Schelling trajectories to the packaged landscape summary.

This module validates the complete developer-only reference dataset, reads each
run's terminal state, and calculates satisfaction and termination aggregates for
every Landscape Cell. It writes one JSONL record per cell; intermediate grids
and per-seed trajectories remain in ``.scratch`` and are not packaged.
"""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from ..profile import Rational
from ..reference import TerminalStatus, evaluate_satisfaction
from .dataset import validate_dataset


class LandscapePoint(BaseModel):
    """Seed-aggregated outcome for one Landscape Cell."""

    model_config = ConfigDict(frozen=True)

    board_size: int
    tolerance_fraction: str
    fractional_preference: float
    vacancy_fraction: str
    empty_percentage: float
    mean_final_satisfaction: float
    final_satisfaction_standard_deviation: float
    equilibrium_rate: float
    blocked_rate: float
    horizon_exhausted_rate: float
    median_trajectory_length: float


def final_satisfaction(cell_types: np.ndarray, tolerance: Rational) -> float:
    """Return the satisfied fraction among occupied cells in one terminal grid."""

    occupied = np.count_nonzero(cell_types)
    if not occupied:
        raise ValueError("final grid has no agents")
    return float(np.count_nonzero(evaluate_satisfaction(cell_types, tolerance)) / occupied)


def analyze_landscape(dataset_root: Path) -> tuple[LandscapePoint, ...]:
    """Validate the full sweep and aggregate its terminal outcomes."""

    points: list[LandscapePoint] = []
    for record in validate_dataset(dataset_root):
        cell = record.cell
        with np.load(dataset_root / record.path, allow_pickle=False) as arrays:
            cell_types: NDArray[np.uint8] = arrays["cell_types"]
            lengths: NDArray[np.uint16] = arrays["trajectory_lengths"]
            statuses: NDArray[np.uint8] = arrays["terminal_status"]
            satisfaction = [
                final_satisfaction(cell_types[index, int(length) - 1], cell.tolerance)
                for index, length in enumerate(lengths)
            ]
        points.append(
            LandscapePoint(
                board_size=cell.board_size,
                tolerance_fraction=str(cell.tolerance),
                fractional_preference=cell.tolerance.numerator / cell.tolerance.denominator,
                vacancy_fraction=str(cell.vacancy_fraction),
                empty_percentage=(
                    100 * cell.vacancy_fraction.numerator / cell.vacancy_fraction.denominator
                ),
                mean_final_satisfaction=float(np.mean(satisfaction)),
                final_satisfaction_standard_deviation=float(np.std(satisfaction, ddof=1)),
                equilibrium_rate=(
                    int(np.count_nonzero(statuses == int(TerminalStatus.EQUILIBRIUM)))
                    / len(statuses)
                ),
                blocked_rate=(
                    int(np.count_nonzero(statuses == int(TerminalStatus.BLOCKED))) / len(statuses)
                ),
                horizon_exhausted_rate=(
                    int(np.count_nonzero(statuses == int(TerminalStatus.HORIZON_EXHAUSTED)))
                    / len(statuses)
                ),
                median_trajectory_length=float(np.median(lengths)),
            )
        )
    return tuple(points)


def write_landscape(points: tuple[LandscapePoint, ...], path: Path) -> None:
    """Write one aggregate JSON object per Landscape Cell."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(f"{point.model_dump_json()}\n" for point in points),
        encoding="utf-8",
    )
