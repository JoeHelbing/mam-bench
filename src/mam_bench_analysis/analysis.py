"""Machine-readable final-satisfaction analysis for Reference Landscape v1."""

import csv
from enum import StrEnum
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from mam_bench.simulations.schelling.dataset import validate_dataset
from mam_bench.simulations.schelling.profile import TOLERANCES, VACANCY_FRACTIONS, Rational
from mam_bench.simulations.schelling.reference import TerminalStatus, evaluate_satisfaction


class ManifoldPoint(BaseModel):
    """Seed-aggregated final outcome for one Landscape Cell."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tolerance_index: int = Field(ge=0, lt=23)
    tolerance_fraction: str
    fractional_preference: float = Field(ge=0.0, le=1.0)
    vacancy_index: int = Field(ge=0, lt=7)
    empty_percentage: float = Field(ge=0.0, le=100.0)
    mean_final_satisfaction: float = Field(ge=0.0, le=1.0)
    final_satisfaction_standard_deviation: float = Field(ge=0.0)
    equilibrium_rate: float = Field(ge=0.0, le=1.0)
    blocked_rate: float = Field(ge=0.0, le=1.0)
    horizon_exhausted_rate: float = Field(ge=0.0, le=1.0)
    median_trajectory_length: float = Field(ge=1.0, le=501.0)


class SpotRole(StrEnum):
    """Purpose of one selected test coordinate."""

    CLASSIC_ANCHOR = "classic_anchor"
    MID_SURFACE = "mid_surface"
    BLOCKED_LOW = "blocked_low"


class TestSpot(BaseModel):
    """One selected Landscape Cell and why it belongs in the test set."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: SpotRole
    label: str
    rationale: str
    point: ManifoldPoint


class ManifoldAnalysis(BaseModel):
    """Complete 23x7 satisfaction surface and its three selected cells."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "mam-bench.final-satisfaction-manifold.v1"
    metric: str = "mean final satisfied-agent fraction across 20 Landscape Seeds"
    points: tuple[ManifoldPoint, ...]
    selected_spots: tuple[TestSpot, ...]


def write_reference_landscape_analysis(
    analysis: ManifoldAnalysis,
    output_directory: Path,
) -> tuple[Path, Path]:
    """Write JSON and CSV forms of the complete derived surface."""

    output_directory.mkdir(parents=True, exist_ok=True)
    json_path = output_directory / "final-satisfaction-manifold.json"
    csv_path = output_directory / "final-satisfaction-manifold.csv"
    json_path.write_text(f"{analysis.model_dump_json(indent=2)}\n", encoding="utf-8")

    point_records = [point.model_dump(mode="json") for point in analysis.points]
    if not point_records:
        raise ValueError("analysis has no manifold points")
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(point_records[0]))
        writer.writeheader()
        writer.writerows(point_records)
    return json_path, csv_path


def final_satisfaction(cell_types: np.ndarray, tolerance: Rational) -> float:
    """Return the satisfied fraction among occupied cells in one terminal grid."""

    occupied_count = int(np.count_nonzero(cell_types))
    if occupied_count == 0:
        raise ValueError("final grid has no agents")
    satisfied = evaluate_satisfaction(cell_types, tolerance)
    return float(np.count_nonzero(satisfied)) / occupied_count


def analyze_final_satisfaction(dataset_root: Path) -> ManifoldAnalysis:
    """Validate the raw dataset, then derive all 161 aggregate surface points."""

    manifest = validate_dataset(dataset_root)
    points: list[ManifoldPoint] = []
    records = {
        (record.tolerance_index, record.vacancy_index): record for record in manifest.artifacts
    }
    for tolerance_index, tolerance in enumerate(TOLERANCES):
        for vacancy_index, vacancy_fraction in enumerate(VACANCY_FRACTIONS):
            record = records[(tolerance_index, vacancy_index)]
            with np.load(dataset_root / record.path, allow_pickle=False) as arrays:
                cell_types: NDArray[np.uint8] = np.asarray(arrays["cell_types"], dtype=np.uint8)
                trajectory_lengths: NDArray[np.uint16] = np.asarray(
                    arrays["trajectory_lengths"], dtype=np.uint16
                )
                terminal_status: NDArray[np.uint8] = np.asarray(
                    arrays["terminal_status"], dtype=np.uint8
                )
                satisfaction = np.asarray(
                    [
                        final_satisfaction(
                            cell_types[run_index, int(length) - 1],
                            tolerance,
                        )
                        for run_index, length in enumerate(trajectory_lengths)
                    ],
                    dtype=np.float64,
                )
                run_count = terminal_status.size
                equilibrium_rate = (
                    int(np.count_nonzero(terminal_status == int(TerminalStatus.EQUILIBRIUM)))
                    / run_count
                )
                blocked_rate = (
                    int(np.count_nonzero(terminal_status == int(TerminalStatus.BLOCKED)))
                    / run_count
                )
                horizon_rate = (
                    int(np.count_nonzero(terminal_status == int(TerminalStatus.HORIZON_EXHAUSTED)))
                    / run_count
                )
                points.append(
                    ManifoldPoint(
                        tolerance_index=tolerance_index,
                        tolerance_fraction=str(tolerance),
                        fractional_preference=(tolerance.numerator / tolerance.denominator),
                        vacancy_index=vacancy_index,
                        empty_percentage=(
                            100.0 * vacancy_fraction.numerator / vacancy_fraction.denominator
                        ),
                        mean_final_satisfaction=float(np.mean(satisfaction)),
                        final_satisfaction_standard_deviation=float(np.std(satisfaction, ddof=1)),
                        equilibrium_rate=equilibrium_rate,
                        blocked_rate=blocked_rate,
                        horizon_exhausted_rate=horizon_rate,
                        median_trajectory_length=float(np.median(trajectory_lengths)),
                    )
                )
    point_tuple = tuple(points)
    return ManifoldAnalysis(
        points=point_tuple,
        selected_spots=select_test_spots(point_tuple),
    )


def select_test_spots(points: tuple[ManifoldPoint, ...]) -> tuple[TestSpot, ...]:
    """Select three preference regimes while holding vacancy at 25 percent."""

    by_coordinate = {(point.tolerance_index, point.vacancy_index): point for point in points}
    coordinates = {
        "classic anchor": (11, 3),
        "transition": (17, 3),
        "blocked low": (20, 3),
    }
    try:
        anchor = by_coordinate[coordinates["classic anchor"]]
        transition = by_coordinate[coordinates["transition"]]
        blocked_low = by_coordinate[coordinates["blocked low"]]
    except KeyError as exc:
        raise ValueError(f"surface lacks selected fixed-vacancy coordinate {exc.args[0]}") from exc

    return (
        TestSpot(
            role=SpotRole.CLASSIC_ANCHOR,
            label="Classic anchor",
            rationale=(
                "Keeps the recognizable 1/2 preference and 25% vacancy condition as "
                "a stable all-satisfied reference."
            ),
            point=anchor,
        ),
        TestSpot(
            role=SpotRole.MID_SURFACE,
            label="Fixed-vacancy transition",
            rationale=(
                "Holds vacancy at 25% while raising preference to 3/4; the even "
                "equilibrium/blocked split exposes decision sensitivity."
            ),
            point=transition,
        ),
        TestSpot(
            role=SpotRole.BLOCKED_LOW,
            label="Blocked low-satisfaction regime",
            rationale=(
                "Holds vacancy at 25% while raising preference to 6/7, producing "
                "a stable deadlock stress test without using the 1/1 boundary."
            ),
            point=blocked_low,
        ),
    )
