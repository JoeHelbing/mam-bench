"""Modal adapter for the complete Schelling Reference Landscape v1 sweep."""

import hashlib
import json
import os
from importlib.metadata import version
from pathlib import Path

import modal

APP_NAME = "mam-bench-reference-landscape-v1"
PARTIAL_SCHEMA_VERSION = "mam-bench.reference-landscape.partial.v1"
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
    coordinate: tuple[int, int],
) -> tuple[int, int, bytes, str, dict[str, str]]:
    """Generate and validate one 20-seed Landscape Cell remotely."""

    import platform

    import numpy as np
    import pydantic

    from mam_bench.simulations.schelling.dataset import build_cell_archive
    from mam_bench.simulations.schelling.profile import landscape_cell

    tolerance_index, vacancy_index = coordinate
    archive = build_cell_archive(landscape_cell(tolerance_index, vacancy_index))
    runtime_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pydantic": pydantic.__version__,
    }
    return (
        tolerance_index,
        vacancy_index,
        archive,
        hashlib.sha256(archive).hexdigest(),
        runtime_versions,
    )


def _atomic_write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


def _atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    encoded = f"{json.dumps(payload, indent=2, sort_keys=True)}\n".encode()
    _atomic_write_bytes(path, encoded)


def _load_valid_receipts(
    output_root: Path,
) -> tuple[dict[tuple[int, int], dict[str, object]], dict[str, str]]:
    partial_path = output_root / ".partial-artifacts.json"
    if not partial_path.exists():
        return {}, {}
    payload = json.loads(partial_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != PARTIAL_SCHEMA_VERSION:
        raise ValueError("partial receipt has an unsupported schema version")
    records: dict[tuple[int, int], dict[str, object]] = {}
    for raw_record in payload.get("artifacts", []):
        if not isinstance(raw_record, dict):
            continue
        tolerance_index = int(raw_record["tolerance_index"])
        vacancy_index = int(raw_record["vacancy_index"])
        expected_path = f"cells/t{tolerance_index:02d}-v{vacancy_index:02d}.npz"
        if raw_record.get("path") != expected_path:
            continue
        artifact_path = output_root / expected_path
        if not artifact_path.is_file():
            continue
        content = artifact_path.read_bytes()
        if len(content) != int(raw_record["byte_size"]):
            continue
        if hashlib.sha256(content).hexdigest() != raw_record.get("sha256"):
            continue
        records[(tolerance_index, vacancy_index)] = raw_record
    raw_versions = payload.get("generator_versions", {})
    if not isinstance(raw_versions, dict):
        raise ValueError("partial receipt has invalid generator versions")
    generator_versions = {str(name): str(value) for name, value in raw_versions.items()}
    return records, generator_versions


def _write_partial(
    output_root: Path,
    records: dict[tuple[int, int], dict[str, object]],
    generator_versions: dict[str, str],
) -> None:
    payload: dict[str, object] = {
        "schema_version": PARTIAL_SCHEMA_VERSION,
        "modal_client_version": version("modal"),
        "generator_versions": generator_versions,
        "artifacts": [records[key] for key in sorted(records)],
    }
    _atomic_write_json(output_root / ".partial-artifacts.json", payload)


@app.local_entrypoint()
def main(
    output_dir: str = "data/reference-landscape/v1",
    max_cells: int = 0,
) -> None:
    """Run missing cells in parallel and retain a resumable local receipt."""

    if max_cells < 0:
        raise ValueError("max_cells cannot be negative")
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    records, generator_versions = _load_valid_receipts(output_root)
    coordinates = [(tolerance, vacancy) for tolerance in range(23) for vacancy in range(7)]
    missing = [coordinate for coordinate in coordinates if coordinate not in records]
    selected = missing[:max_cells] if max_cells else missing
    print(
        f"Reference Landscape: {len(records)} complete, {len(missing)} missing, "
        f"submitting {len(selected)}"
    )

    for (
        tolerance_index,
        vacancy_index,
        archive,
        remote_sha256,
        runtime_versions,
    ) in generate_cell.map(selected, order_outputs=False):
        local_sha256 = hashlib.sha256(archive).hexdigest()
        if local_sha256 != remote_sha256:
            raise RuntimeError(
                f"result hash changed in transit for t{tolerance_index:02d}-v{vacancy_index:02d}"
            )
        if generator_versions and runtime_versions != generator_versions:
            raise RuntimeError("Modal generator software versions changed during the sweep")
        generator_versions = runtime_versions
        relative_path = f"cells/t{tolerance_index:02d}-v{vacancy_index:02d}.npz"
        _atomic_write_bytes(output_root / relative_path, archive)
        record: dict[str, object] = {
            "tolerance_index": tolerance_index,
            "vacancy_index": vacancy_index,
            "path": relative_path,
            "byte_size": len(archive),
            "sha256": local_sha256,
        }
        records[(tolerance_index, vacancy_index)] = record
        _write_partial(output_root, records, generator_versions)
        print(
            f"[{len(records):3d}/161] {relative_path} "
            f"({len(archive) / (1024 * 1024):.2f} MiB)"
        )

    if len(records) != 161:
        if max_cells:
            print(f"Smoke/bounded run ended with {len(records)} of 161 artifacts.")
            return
        raise RuntimeError(f"sweep ended with {len(records)} of 161 artifacts")
    print("All raw artifacts received.")
    print(
        "Validate and finalize the dataset with "
        "mam_bench.simulations.schelling.dataset APIs: "
        f"{output_root}"
    )
