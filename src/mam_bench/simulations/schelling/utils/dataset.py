"""Build and validate the full Schelling Reference Dataset v2 sweep.

A dataset contains one compressed NPZ archive for each board-size, tolerance,
and vacancy combination. Each archive stores 50 deterministic trajectories plus
the metadata needed for analysis. This module defines that archive format and
checks the completed manifest, hashes, array shapes, seeds, padding, populations,
and tolerance-paired initial boards before compact data is published.

The full dataset is developer-only material under ``.scratch``. Ordinary
benchmark runs use the much smaller packaged fixture and never call this module.
"""

import hashlib
import io
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from ..profile import (
    LANDSCAPE_CELL_COUNT,
    LANDSCAPE_SEED_IDS,
    MAX_TRANSITIONS,
    LandscapeCell,
    Rational,
    landscape_cells,
)
from ..reference import PAD_CELL, PAD_LOCATION, TerminalStatus, run_reference

MAX_STATES = MAX_TRANSITIONS + 1
EXPECTED_ARRAY_KEYS = frozenset(
    {
        "cell_types",
        "agent_locations",
        "agent_types",
        "trajectory_lengths",
        "terminal_status",
        "landscape_seed_ids",
        "board_size",
        "tolerance_numerator",
        "tolerance_denominator",
        "vacancy_numerator",
        "vacancy_denominator",
        "vacancy_count",
    }
)


class ArtifactRecord(BaseModel):
    """One independently verifiable Reference Dataset archive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    board_size: int
    tolerance: str
    vacancy_fraction: str
    vacancy_count: int = Field(gt=0)
    run_count: int = Field(gt=0)
    path: str
    byte_size: int = Field(gt=0)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value

    @property
    def cell(self) -> LandscapeCell:
        return LandscapeCell(
            self.board_size,
            Rational.parse(self.tolerance),
            Rational.parse(self.vacancy_fraction),
        )

    @model_validator(mode="after")
    def validate_cell_metadata(self) -> Self:
        cell = self.cell
        if (
            self.vacancy_count != cell.vacancy_count
            or self.run_count != len(LANDSCAPE_SEED_IDS)
            or self.path != cell_filename(cell)
        ):
            raise ValueError("artifact metadata does not match its Landscape Cell")
        return self


def cell_filename(cell: LandscapeCell) -> str:
    """Return the canonical dataset-relative path for one cell."""

    return f"cells/{cell.cell_id}.npz"


def build_cell_archive(cell: LandscapeCell) -> bytes:
    """Generate all Landscape Seeds for one cell and return a compressed NPZ."""

    run_count = len(LANDSCAPE_SEED_IDS)
    cell_types = np.full(
        (run_count, MAX_STATES, cell.board_size, cell.board_size),
        PAD_CELL,
        dtype=np.uint8,
    )
    agent_locations = np.full(
        (run_count, MAX_STATES, cell.agent_count),
        PAD_LOCATION,
        dtype=np.uint16,
    )
    trajectory_lengths = np.empty(run_count, dtype=np.uint16)
    terminal_status = np.empty(run_count, dtype=np.uint8)
    agent_types: np.ndarray | None = None

    for run_index, seed_id in enumerate(LANDSCAPE_SEED_IDS):
        trajectory = run_reference(cell, seed_id)
        length = trajectory.trajectory_length
        cell_types[run_index, :length] = trajectory.cell_types
        agent_locations[run_index, :length] = trajectory.agent_locations
        trajectory_lengths[run_index] = length
        terminal_status[run_index] = int(trajectory.terminal_status)
        if agent_types is None:
            agent_types = trajectory.agent_types.copy()
        elif not np.array_equal(agent_types, trajectory.agent_types):
            raise RuntimeError("agent type metadata changed across seeds")

    if agent_types is None:
        raise RuntimeError("cell generation produced no runs")

    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        cell_types=cell_types,
        agent_locations=agent_locations,
        agent_types=agent_types,
        trajectory_lengths=trajectory_lengths,
        terminal_status=terminal_status,
        landscape_seed_ids=np.asarray(LANDSCAPE_SEED_IDS, dtype=np.uint16),
        board_size=np.asarray(cell.board_size, dtype=np.uint16),
        tolerance_numerator=np.asarray(cell.tolerance.numerator, dtype=np.uint8),
        tolerance_denominator=np.asarray(cell.tolerance.denominator, dtype=np.uint8),
        vacancy_numerator=np.asarray(cell.vacancy_fraction.numerator, dtype=np.uint8),
        vacancy_denominator=np.asarray(cell.vacancy_fraction.denominator, dtype=np.uint8),
        vacancy_count=np.asarray(cell.vacancy_count, dtype=np.uint16),
    )
    return stream.getvalue()


def _load_archive(archive: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(archive), allow_pickle=False) as loaded:
        if frozenset(loaded.files) != EXPECTED_ARRAY_KEYS:
            raise ValueError("archive arrays do not match the dataset format")
        return {name: loaded[name].copy() for name in loaded.files}


def _scalar(arrays: dict[str, np.ndarray], name: str) -> int:
    value = arrays[name]
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar array")
    return int(value.item())


def validate_cell_archive(archive: bytes, expected_cell: LandscapeCell) -> dict[str, np.ndarray]:
    """Fail closed unless an archive exactly matches its scientific contract."""

    arrays = _load_archive(archive)
    expected_scalars = {
        "board_size": expected_cell.board_size,
        "tolerance_numerator": expected_cell.tolerance.numerator,
        "tolerance_denominator": expected_cell.tolerance.denominator,
        "vacancy_numerator": expected_cell.vacancy_fraction.numerator,
        "vacancy_denominator": expected_cell.vacancy_fraction.denominator,
        "vacancy_count": expected_cell.vacancy_count,
    }
    for name, expected in expected_scalars.items():
        if _scalar(arrays, name) != expected:
            raise ValueError(f"{name} does not match expected Landscape Cell")

    run_count = len(LANDSCAPE_SEED_IDS)
    expected_shapes_and_dtypes = {
        "cell_types": (
            (run_count, MAX_STATES, expected_cell.board_size, expected_cell.board_size),
            np.dtype(np.uint8),
        ),
        "agent_locations": (
            (run_count, MAX_STATES, expected_cell.agent_count),
            np.dtype(np.uint16),
        ),
        "agent_types": ((expected_cell.agent_count,), np.dtype(np.uint8)),
        "trajectory_lengths": ((run_count,), np.dtype(np.uint16)),
        "terminal_status": ((run_count,), np.dtype(np.uint8)),
        "landscape_seed_ids": ((run_count,), np.dtype(np.uint16)),
    }
    for name, (shape, dtype) in expected_shapes_and_dtypes.items():
        value = arrays[name]
        if value.shape != shape or value.dtype != dtype:
            raise ValueError(
                f"{name} expected shape={shape}, dtype={dtype}; "
                f"received shape={value.shape}, dtype={value.dtype}"
            )

    if not np.array_equal(
        arrays["landscape_seed_ids"], np.asarray(LANDSCAPE_SEED_IDS, dtype=np.uint16)
    ):
        raise ValueError("landscape_seed_ids are not canonical")
    half = expected_cell.agent_count // 2
    expected_agent_types = np.concatenate(
        (np.full(half, 1, dtype=np.uint8), np.full(half, 2, dtype=np.uint8))
    )
    if not np.array_equal(arrays["agent_types"], expected_agent_types):
        raise ValueError("agent_types are not the canonical equal split")

    lengths = arrays["trajectory_lengths"]
    statuses = arrays["terminal_status"]
    valid_statuses = {int(status) for status in TerminalStatus}
    for run_index in range(run_count):
        length = int(lengths[run_index])
        if not 1 <= length <= MAX_STATES:
            raise ValueError(f"trajectory length is outside 1..{MAX_STATES}")
        if int(statuses[run_index]) not in valid_statuses:
            raise ValueError("terminal status code is unknown")
        if not np.all(arrays["cell_types"][run_index, length:] == PAD_CELL):
            raise ValueError("cell_types has non-padding states after trajectory length")
        if not np.all(arrays["agent_locations"][run_index, length:] == PAD_LOCATION):
            raise ValueError("agent_locations has non-padding states after trajectory length")
    return arrays


def _ordered_complete_records(records: list[ArtifactRecord]) -> tuple[ArtifactRecord, ...]:
    expected = {cell_filename(cell): cell for cell in landscape_cells()}
    actual = {record.path for record in records}
    if len(records) != LANDSCAPE_CELL_COUNT or actual != set(expected):
        raise ValueError("manifest must contain exactly one record for every Landscape Cell")
    order = {path: index for index, path in enumerate(expected)}
    return tuple(sorted(records, key=lambda record: order[record.path]))


def read_manifest(path: Path) -> tuple[ArtifactRecord, ...]:
    """Read and validate the artifact-only JSONL manifest."""

    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or any(not line.strip() for line in lines):
        raise ValueError("manifest must contain one JSON object per non-empty line")
    return _ordered_complete_records([ArtifactRecord.model_validate_json(line) for line in lines])


def _validate_artifact_records(output_root: Path, records: tuple[ArtifactRecord, ...]) -> None:
    paired_initial: dict[tuple[int, str, int], np.ndarray] = {}
    for record in records:
        archive = (output_root / record.path).read_bytes()
        if len(archive) != record.byte_size:
            raise ValueError(f"byte size differs for {record.path}")
        if hashlib.sha256(archive).hexdigest() != record.sha256:
            raise ValueError(f"SHA-256 differs for {record.path}")
        cell = record.cell
        arrays = validate_cell_archive(archive, cell)
        for run_index, seed_id in enumerate(LANDSCAPE_SEED_IDS):
            key = (cell.board_size, str(cell.vacancy_fraction), seed_id)
            initial = arrays["cell_types"][run_index, 0]
            expected = paired_initial.setdefault(key, initial.copy())
            if not np.array_equal(initial, expected):
                raise ValueError(
                    "initial grids are not paired across tolerance at "
                    f"board={cell.board_size}, vacancy={cell.vacancy_fraction}, seed={seed_id}"
                )


def validate_dataset(output_root: Path) -> tuple[ArtifactRecord, ...]:
    """Validate the artifact inventory, hashes, archives, and paired initial grids."""

    records = read_manifest(output_root / "manifest.jsonl")
    expected_paths = {record.path for record in records}
    cells_directory = output_root / "cells"
    actual_paths = {
        path.relative_to(output_root).as_posix()
        for path in cells_directory.rglob("*")
        if path.is_file()
    }
    if actual_paths != expected_paths:
        missing = sorted(expected_paths - actual_paths)
        extra = sorted(actual_paths - expected_paths)
        raise ValueError(f"Reference Dataset cells differ: missing={missing}, extra={extra}")
    _validate_artifact_records(output_root, records)
    return records
