"""Artifact packaging and validation for Reference Landscape v1."""

import hashlib
import io
import json
import os
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

import numpy as np
import pydantic
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .profile import (
    LANDSCAPE_CELL_COUNT,
    LANDSCAPE_SEED_IDS,
    PROFILE,
    LandscapeCell,
    ReferenceProfile,
    landscape_cell,
)
from .reference import PAD_CELL, PAD_LOCATION, TerminalStatus, run_reference

SCHEMA_VERSION = "mam-bench.reference-landscape.v1"
MAX_STATES = PROFILE.max_transitions + 1
EXPECTED_ARRAY_KEYS = frozenset(
    {
        "cell_types",
        "agent_locations",
        "agent_types",
        "trajectory_lengths",
        "terminal_status",
        "landscape_seed_ids",
        "tolerance_index",
        "tolerance_numerator",
        "tolerance_denominator",
        "vacancy_index",
        "vacancy_numerator",
        "vacancy_denominator",
        "vacancy_count",
    }
)


class ArtifactRecord(BaseModel):
    """Manifest entry for one independently verifiable cell archive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tolerance_index: int = Field(ge=0, lt=23)
    vacancy_index: int = Field(ge=0, lt=7)
    path: str
    byte_size: int = Field(gt=0)
    sha256: str

    @field_validator("sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("sha256 must be 64 lowercase hexadecimal characters")
        return value

    @model_validator(mode="after")
    def validate_canonical_path(self) -> Self:
        expected = f"cells/t{self.tolerance_index:02d}-v{self.vacancy_index:02d}.npz"
        if self.path != expected:
            raise ValueError("artifact path does not match its Landscape Cell")
        return self


class PartialSweep(BaseModel):
    """Resumable local receipt written as Modal cell results arrive."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "mam-bench.reference-landscape.partial.v1"
    modal_client_version: str
    generator_versions: dict[str, str]
    artifacts: tuple[ArtifactRecord, ...]


class ArraySpecification(BaseModel):
    """Machine-readable interpretation of one NPZ array."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dtype: str
    shape: str
    description: str
    padding: int | None = None


STATE_CODES = {"empty": 0, "type_a": 1, "type_b": 2, "padding": 255}
TERMINAL_STATUS_CODES = {
    "equilibrium": int(TerminalStatus.EQUILIBRIUM),
    "blocked": int(TerminalStatus.BLOCKED),
    "horizon_exhausted": int(TerminalStatus.HORIZON_EXHAUSTED),
}
RNG_STREAM_IDS = {"initialization": 0, "reservation_order": 1, "tie_break": 2}


def _array_specifications() -> dict[str, ArraySpecification]:
    return {
        "cell_types": ArraySpecification(
            dtype="uint8",
            shape="[20,501,20,20]",
            description="Cell-type grid at every retained state.",
            padding=int(PAD_CELL),
        ),
        "agent_locations": ArraySpecification(
            dtype="uint16",
            shape="[20,501,n_agents]",
            description="Flattened cell index for each stable agent ID.",
            padding=int(PAD_LOCATION),
        ),
        "agent_types": ArraySpecification(
            dtype="uint8",
            shape="[n_agents]",
            description="Static type indexed by stable agent ID.",
        ),
        "trajectory_lengths": ArraySpecification(
            dtype="uint16",
            shape="[20]",
            description="Number of retained states, including initialization.",
        ),
        "terminal_status": ArraySpecification(
            dtype="uint8",
            shape="[20]",
            description="Terminal outcome code for each Landscape Seed.",
        ),
    }


class DatasetManifest(BaseModel):
    """Complete provenance and integrity contract for the raw landscape."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str
    dataset_version: str
    created_at: datetime
    complete: bool
    expected_cell_count: int
    expected_run_count: int
    profile: ReferenceProfile
    state_codes: dict[str, int]
    terminal_status_codes: dict[str, int]
    arrays: dict[str, ArraySpecification]
    rng_stream_ids: dict[str, int]
    software_versions: dict[str, str]
    artifacts: tuple[ArtifactRecord, ...]

    @model_validator(mode="after")
    def validate_canonical_contract(self) -> Self:
        expected_coordinates = {
            (tolerance, vacancy) for tolerance in range(23) for vacancy in range(7)
        }
        actual_coordinates = {
            (artifact.tolerance_index, artifact.vacancy_index) for artifact in self.artifacts
        }
        errors: list[str] = []
        if self.schema_version != SCHEMA_VERSION:
            errors.append("schema_version")
        if self.dataset_version != "reference-landscape-v1":
            errors.append("dataset_version")
        if not self.complete:
            errors.append("complete")
        if self.expected_cell_count != LANDSCAPE_CELL_COUNT:
            errors.append("expected_cell_count")
        if self.expected_run_count != LANDSCAPE_CELL_COUNT * len(LANDSCAPE_SEED_IDS):
            errors.append("expected_run_count")
        if self.profile != PROFILE:
            errors.append("profile")
        if self.state_codes != STATE_CODES:
            errors.append("state_codes")
        if self.terminal_status_codes != TERMINAL_STATUS_CODES:
            errors.append("terminal_status_codes")
        if self.arrays != _array_specifications():
            errors.append("arrays")
        if self.rng_stream_ids != RNG_STREAM_IDS:
            errors.append("rng_stream_ids")
        if len(self.artifacts) != LANDSCAPE_CELL_COUNT:
            errors.append("artifacts.length")
        if actual_coordinates != expected_coordinates:
            errors.append("artifacts.coordinates")
        required_versions = {
            "validator_python",
            "validator_numpy",
            "validator_pydantic",
            "generator_python",
            "generator_numpy",
            "generator_pydantic",
        }
        if not required_versions.issubset(self.software_versions) or any(
            not self.software_versions[name] for name in required_versions
        ):
            errors.append("software_versions")
        if errors:
            raise ValueError(f"manifest differs from canonical v1 contract: {errors}")
        return self

    @classmethod
    def build_complete(
        cls,
        *,
        artifacts: list[ArtifactRecord],
        modal_client_version: str | None = None,
        generator_versions: dict[str, str] | None = None,
    ) -> "DatasetManifest":
        ordered = tuple(
            sorted(artifacts, key=lambda item: (item.tolerance_index, item.vacancy_index))
        )
        validator_versions = _software_versions()
        software_versions = {
            f"validator_{name}": value for name, value in validator_versions.items()
        }
        source_versions = generator_versions or validator_versions
        software_versions.update(
            {f"generator_{name}": value for name, value in source_versions.items()}
        )
        if modal_client_version is not None:
            software_versions["modal_client"] = modal_client_version
        return cls(
            schema_version=SCHEMA_VERSION,
            dataset_version="reference-landscape-v1",
            created_at=datetime.now(UTC),
            complete=True,
            expected_cell_count=LANDSCAPE_CELL_COUNT,
            expected_run_count=LANDSCAPE_CELL_COUNT * len(LANDSCAPE_SEED_IDS),
            profile=PROFILE,
            state_codes=STATE_CODES,
            terminal_status_codes=TERMINAL_STATUS_CODES,
            arrays=_array_specifications(),
            rng_stream_ids=RNG_STREAM_IDS,
            software_versions=software_versions,
            artifacts=ordered,
        )


def _software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pydantic": pydantic.__version__,
    }


def cell_filename(cell: LandscapeCell) -> str:
    """Return the canonical dataset-relative path for one cell."""

    return f"cells/{cell.cell_id}.npz"


def build_cell_archive(cell: LandscapeCell) -> bytes:
    """Generate all Landscape Seeds for one cell and return a compressed NPZ."""

    run_count = len(LANDSCAPE_SEED_IDS)
    cell_types = np.full(
        (run_count, MAX_STATES, PROFILE.grid_size, PROFILE.grid_size),
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
        tolerance_index=np.asarray(cell.tolerance_index, dtype=np.uint8),
        tolerance_numerator=np.asarray(cell.tolerance.numerator, dtype=np.uint8),
        tolerance_denominator=np.asarray(cell.tolerance.denominator, dtype=np.uint8),
        vacancy_index=np.asarray(cell.vacancy_index, dtype=np.uint8),
        vacancy_numerator=np.asarray(cell.vacancy_fraction.numerator, dtype=np.uint8),
        vacancy_denominator=np.asarray(cell.vacancy_fraction.denominator, dtype=np.uint8),
        vacancy_count=np.asarray(cell.vacancy_count, dtype=np.uint16),
    )
    archive = stream.getvalue()
    validate_cell_archive(archive, cell)
    return archive


def _load_archive(archive: bytes) -> dict[str, np.ndarray]:
    try:
        with np.load(io.BytesIO(archive), allow_pickle=False) as loaded:
            if frozenset(loaded.files) != EXPECTED_ARRAY_KEYS:
                missing = sorted(EXPECTED_ARRAY_KEYS - frozenset(loaded.files))
                extra = sorted(frozenset(loaded.files) - EXPECTED_ARRAY_KEYS)
                raise ValueError(f"archive keys differ: missing={missing}, extra={extra}")
            return {name: loaded[name].copy() for name in loaded.files}
    except (OSError, ValueError) as exc:
        if isinstance(exc, ValueError) and str(exc).startswith("archive keys differ"):
            raise
        raise ValueError(f"invalid NPZ archive: {exc}") from exc


def _scalar(arrays: dict[str, np.ndarray], name: str) -> int:
    value = arrays[name]
    if value.shape != ():
        raise ValueError(f"{name} must be a scalar array")
    return int(value.item())


def validate_cell_archive(archive: bytes, expected_cell: LandscapeCell) -> dict[str, np.ndarray]:
    """Fail closed unless an archive exactly matches its scientific contract."""

    arrays = _load_archive(archive)
    expected_scalars = {
        "tolerance_index": expected_cell.tolerance_index,
        "tolerance_numerator": expected_cell.tolerance.numerator,
        "tolerance_denominator": expected_cell.tolerance.denominator,
        "vacancy_index": expected_cell.vacancy_index,
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
            (run_count, MAX_STATES, PROFILE.grid_size, PROFILE.grid_size),
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
        (
            np.full(half, 1, dtype=np.uint8),
            np.full(half, 2, dtype=np.uint8),
        )
    )
    if not np.array_equal(arrays["agent_types"], expected_agent_types):
        raise ValueError("agent_types are not the canonical equal split")

    lengths = arrays["trajectory_lengths"]
    statuses = arrays["terminal_status"]
    valid_statuses = {int(status) for status in TerminalStatus}
    for run_index in range(run_count):
        length = int(lengths[run_index])
        if not 1 <= length <= MAX_STATES:
            raise ValueError("trajectory length is outside 1..501")
        if int(statuses[run_index]) not in valid_statuses:
            raise ValueError("terminal status code is unknown")
        if not np.all(arrays["cell_types"][run_index, length:] == PAD_CELL):
            raise ValueError("cell_types has non-padding states after trajectory length")
        if not np.all(arrays["agent_locations"][run_index, length:] == PAD_LOCATION):
            raise ValueError("agent_locations has non-padding states after trajectory length")
        _validate_run_states(arrays, run_index, length, expected_cell)
    return arrays


def _validate_run_states(
    arrays: dict[str, np.ndarray],
    run_index: int,
    length: int,
    cell: LandscapeCell,
) -> None:
    agent_types = arrays["agent_types"]
    prior_locations: np.ndarray | None = None
    for state_index in range(length):
        grid = np.asarray(arrays["cell_types"][run_index, state_index], dtype=np.uint8)
        locations = np.asarray(arrays["agent_locations"][run_index, state_index], dtype=np.uint16)
        if np.any(locations >= PROFILE.grid_size * PROFILE.grid_size):
            raise ValueError("retained agent location is outside the grid")
        if len(np.unique(locations)) != cell.agent_count:
            raise ValueError("retained agent locations are not unique")
        reconstructed = np.zeros(PROFILE.grid_size * PROFILE.grid_size, dtype=np.uint8)
        reconstructed[locations] = agent_types
        expected_grid = reconstructed.reshape((PROFILE.grid_size, PROFILE.grid_size))
        if not np.array_equal(expected_grid, grid):
            raise ValueError("cell grid and agent locations disagree")
        if np.count_nonzero(grid == 0) != cell.vacancy_count:
            raise ValueError("state does not preserve exact vacancy count")
        if prior_locations is not None:
            moved = prior_locations != locations
            prior_occupied = np.zeros(PROFILE.grid_size * PROFILE.grid_size, dtype=np.bool_)
            prior_occupied[prior_locations] = True
            if np.any(prior_occupied[locations[moved]]):
                raise ValueError("a move used a cell that was occupied at round start")
        prior_locations = locations


def write_artifact_atomically(
    output_root: Path, cell: LandscapeCell, archive: bytes
) -> ArtifactRecord:
    """Validate, atomically persist, and describe one cell artifact."""

    validate_cell_archive(archive, cell)
    relative_path = Path(cell_filename(cell))
    destination = output_root / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_bytes(archive)
    os.replace(temporary, destination)
    return ArtifactRecord(
        tolerance_index=cell.tolerance_index,
        vacancy_index=cell.vacancy_index,
        path=relative_path.as_posix(),
        byte_size=len(archive),
        sha256=hashlib.sha256(archive).hexdigest(),
    )


def write_manifest_atomically(output_root: Path, manifest: DatasetManifest) -> Path:
    """Write the final complete manifest only after model validation."""

    destination = output_root / "manifest.json"
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    payload = manifest.model_dump_json(indent=2)
    json.loads(payload)
    temporary.write_text(f"{payload}\n", encoding="utf-8")
    os.replace(temporary, destination)
    return destination


def _validate_artifact_records(output_root: Path, records: tuple[ArtifactRecord, ...]) -> None:
    paired_initial: dict[tuple[int, int], np.ndarray] = {}
    for record in sorted(records, key=lambda item: (item.tolerance_index, item.vacancy_index)):
        path = output_root / record.path
        archive = path.read_bytes()
        if len(archive) != record.byte_size:
            raise ValueError(f"byte size differs for {record.path}")
        if hashlib.sha256(archive).hexdigest() != record.sha256:
            raise ValueError(f"SHA-256 differs for {record.path}")
        cell = landscape_cell(record.tolerance_index, record.vacancy_index)
        arrays = validate_cell_archive(archive, cell)
        for run_index, seed_id in enumerate(LANDSCAPE_SEED_IDS):
            key = (record.vacancy_index, seed_id)
            initial = arrays["cell_types"][run_index, 0]
            expected = paired_initial.setdefault(key, initial.copy())
            if not np.array_equal(initial, expected):
                raise ValueError(
                    "initial grids are not paired across tolerance at "
                    f"vacancy={record.vacancy_index}, seed={seed_id}"
                )


def finalize_dataset(
    output_root: Path,
    artifacts: list[ArtifactRecord],
    *,
    modal_client_version: str | None = None,
    generator_versions: dict[str, str] | None = None,
) -> DatasetManifest:
    """Validate every raw artifact before atomically publishing a complete manifest."""

    candidate = DatasetManifest.build_complete(
        artifacts=artifacts,
        modal_client_version=modal_client_version,
        generator_versions=generator_versions,
    )
    _validate_artifact_records(output_root, candidate.artifacts)
    write_manifest_atomically(output_root, candidate)
    return candidate


def validate_dataset(output_root: Path) -> DatasetManifest:
    """Validate manifest, exact cells, hashes, archives, and paired initial grids."""

    manifest_path = output_root / "manifest.json"
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    expected_paths = {record.path for record in manifest.artifacts}
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
    _validate_artifact_records(output_root, manifest.artifacts)
    return manifest
