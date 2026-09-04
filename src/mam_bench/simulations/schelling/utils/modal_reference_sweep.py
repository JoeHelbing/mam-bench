# pyright: reportFunctionMemberAccess=false, reportMissingImports=false
# pyright: reportUnknownArgumentType=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUntypedFunctionDecorator=false
"""Generate the full Schelling Reference Dataset v2 sweep on Modal.

The local entry point submits one remote task per Landscape Cell, downloads each
generated NPZ archive, and records completed artifacts in an atomic resumable
receipt. A finished 621-cell run writes ``manifest.jsonl`` beside the full
trajectories under ``.scratch``. This adapter is explicitly invoked by a
developer; it is neither imported nor required by normal benchmark execution.
"""

import hashlib
import json
import os
from pathlib import Path
from typing import TypedDict

import modal  # pyright: ignore[reportMissingImports]

from mam_bench.simulations.schelling.profile import (
    LANDSCAPE_CELL_COUNT,
    LandscapeCell,
    Rational,
    landscape_cells,
)
from mam_bench.simulations.schelling.utils.dataset import build_cell_archive, cell_filename

APP_NAME = "mam-bench-schelling-reference-v2"


class ArtifactPayload(TypedDict):
    board_size: int
    tolerance: str
    vacancy_fraction: str
    vacancy_count: int
    run_count: int
    path: str
    byte_size: int
    sha256: str


app = modal.App(APP_NAME)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("numpy==2.5.2", "pydantic==2.13.5")
    .add_local_dir("src/mam_bench", remote_path="/root/mam_bench", copy=True)
)


@app.function(
    image=image,
    cpu=1.0,
    memory=2048,
    timeout=3600,
    retries=2,
    max_containers=32,
)
def generate_cell(
    coordinate: tuple[int, str, str],
) -> tuple[ArtifactPayload, bytes]:
    """Generate and validate one 50-seed Landscape Cell remotely."""

    board_size, tolerance, vacancy_fraction = coordinate
    cell = LandscapeCell(
        board_size,
        Rational.parse(tolerance),
        Rational.parse(vacancy_fraction),
    )
    archive = build_cell_archive(cell)
    record: ArtifactPayload = {
        "board_size": board_size,
        "tolerance": tolerance,
        "vacancy_fraction": vacancy_fraction,
        "vacancy_count": cell.vacancy_count,
        "run_count": 50,
        "path": f"cells/{cell.cell_id}.npz",
        "byte_size": len(archive),
        "sha256": hashlib.sha256(archive).hexdigest(),
    }
    return record, archive


def _coordinates() -> tuple[tuple[int, str, str], ...]:
    return tuple(
        (cell.board_size, str(cell.tolerance), str(cell.vacancy_fraction))
        for cell in landscape_cells()
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    _atomic_write_bytes(path, f"{json.dumps(payload, indent=2, sort_keys=True)}\n".encode())


def _load_receipt(output_root: Path, receipt_path: Path) -> dict[str, ArtifactPayload]:
    if not receipt_path.exists():
        return {}
    records: dict[str, ArtifactPayload] = {}
    for raw_record in json.loads(receipt_path.read_text(encoding="utf-8"))["artifacts"]:
        record = ArtifactPayload(**raw_record)
        path = output_root / record["path"]
        if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == record["sha256"]:
            records[record["path"]] = record
    return records


def _write_receipt(receipt_path: Path, records: dict[str, ArtifactPayload]) -> None:
    _atomic_write_json(
        receipt_path,
        {"artifacts": [records[path] for path in sorted(records)]},
    )


def _write_manifest(output_root: Path, records: dict[str, ArtifactPayload]) -> None:
    paths = (_path(coordinate) for coordinate in _coordinates())
    payload = "".join(f"{json.dumps(records[path], separators=(',', ':'))}\n" for path in paths)
    _atomic_write_bytes(output_root / "manifest.jsonl", payload.encode())


@app.local_entrypoint()
def main(
    output_dir: str = ".scratch/schelling-reference-v2-full",
    receipt_path: str = ".scratch/schelling-reference-v2-receipt.json",
    max_cells: int = 0,
) -> None:
    """Run missing cells in parallel and publish the manifest when complete."""

    if max_cells < 0:
        raise ValueError("max_cells cannot be negative")
    output_root = Path(output_dir).expanduser().resolve()
    receipt = Path(receipt_path).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records = _load_receipt(output_root, receipt)
    coordinates = _coordinates()
    missing = [coordinate for coordinate in coordinates if _path(coordinate) not in records]
    selected = missing[:max_cells] if max_cells else missing
    print(
        f"Schelling Reference v2: {len(records)} complete, {len(missing)} missing, "
        f"submitting {len(selected)}"
    )

    for record, archive in generate_cell.map(selected, order_outputs=False):
        _atomic_write_bytes(output_root / record["path"], archive)
        records[record["path"]] = record
        _write_receipt(receipt, records)
        print(
            f"[{len(records):3d}/{LANDSCAPE_CELL_COUNT}] {record['path']} "
            f"({len(archive) / (1024 * 1024):.2f} MiB)"
        )

    if len(records) != LANDSCAPE_CELL_COUNT:
        if max_cells:
            print(f"Bounded run ended with {len(records)} of {LANDSCAPE_CELL_COUNT} artifacts.")
            return
        raise RuntimeError(f"sweep ended with {len(records)} of {LANDSCAPE_CELL_COUNT} artifacts")
    _write_manifest(output_root, records)
    print(f"Published {output_root / 'manifest.jsonl'}; run publish_reference_data.py locally.")


def _path(coordinate: tuple[int, str, str]) -> str:
    board_size, tolerance, vacancy_fraction = coordinate
    return cell_filename(
        LandscapeCell(
            board_size,
            Rational.parse(tolerance),
            Rational.parse(vacancy_fraction),
        )
    )
