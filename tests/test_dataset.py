import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from mam_bench.simulations.schelling.dataset import (
    ArtifactRecord,
    DatasetManifest,
    build_cell_archive,
    cell_filename,
    validate_cell_archive,
    write_artifact_atomically,
)
from mam_bench.simulations.schelling.profile import LANDSCAPE_CELL_COUNT, landscape_cell
from mam_bench.simulations.schelling.reference import TerminalStatus


class CellArchiveTests(unittest.TestCase):
    def test_tolerance_zero_cell_round_trips_with_safe_typed_arrays(self) -> None:
        cell = landscape_cell(tolerance_index=0, vacancy_index=3)

        archive = build_cell_archive(cell)
        arrays = validate_cell_archive(archive, cell)

        self.assertEqual(arrays["cell_types"].dtype, np.dtype(np.uint8))
        self.assertEqual(arrays["cell_types"].shape, (20, 501, 20, 20))
        self.assertEqual(arrays["agent_locations"].dtype, np.dtype(np.uint16))
        self.assertEqual(arrays["agent_locations"].shape, (20, 501, 300))
        self.assertTrue(np.all(arrays["trajectory_lengths"] == 1))
        self.assertTrue(
            np.all(arrays["terminal_status"] == int(TerminalStatus.EQUILIBRIUM))
        )
        self.assertTrue(np.all(arrays["cell_types"][:, 1:] == 255))
        self.assertTrue(np.all(arrays["agent_locations"][:, 1:] == 65535))

    def test_validation_rejects_wrong_cell_metadata(self) -> None:
        archive = build_cell_archive(landscape_cell(tolerance_index=0, vacancy_index=3))

        with self.assertRaises(ValueError):
            validate_cell_archive(
                archive,
                landscape_cell(tolerance_index=1, vacancy_index=3),
            )

    def test_atomic_write_returns_hashed_artifact_record(self) -> None:
        cell = landscape_cell(tolerance_index=0, vacancy_index=3)
        archive = build_cell_archive(cell)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            record = write_artifact_atomically(root, cell, archive)

            path = root / cell_filename(cell)
            self.assertTrue(path.is_file())
            self.assertEqual(record.byte_size, len(archive))
            self.assertEqual(record.sha256, hashlib.sha256(archive).hexdigest())
            self.assertEqual(path.read_bytes(), archive)


class ManifestTests(unittest.TestCase):
    def test_complete_manifest_requires_every_landscape_cell(self) -> None:
        record = ArtifactRecord(
            tolerance_index=0,
            vacancy_index=0,
            path="cells/t00-v00.npz",
            byte_size=10,
            sha256="0" * 64,
        )

        with self.assertRaises(ValidationError):
            DatasetManifest.build_complete(artifacts=[record])

    @staticmethod
    def _records() -> list[ArtifactRecord]:
        return [
            ArtifactRecord(
                tolerance_index=tolerance_index,
                vacancy_index=vacancy_index,
                path=f"cells/t{tolerance_index:02d}-v{vacancy_index:02d}.npz",
                byte_size=10,
                sha256=f"{tolerance_index * 7 + vacancy_index:064x}",
            )
            for tolerance_index in range(23)
            for vacancy_index in range(7)
        ]

    def test_complete_manifest_serializes_as_json_without_custom_encoder(self) -> None:
        manifest = DatasetManifest.build_complete(artifacts=self._records())
        payload = json.loads(manifest.model_dump_json())

        self.assertEqual(len(payload["artifacts"]), LANDSCAPE_CELL_COUNT)
        self.assertTrue(payload["complete"])
        self.assertEqual(payload["schema_version"], "mam-bench.reference-landscape.v1")

    def test_manifest_rejects_incomplete_status(self) -> None:
        payload = DatasetManifest.build_complete(artifacts=self._records()).model_dump(
            mode="json"
        )
        payload["complete"] = False

        with self.assertRaises(ValidationError):
            DatasetManifest.model_validate(payload)

    def test_manifest_rejects_changed_scientific_profile(self) -> None:
        payload = DatasetManifest.build_complete(artifacts=self._records()).model_dump(
            mode="json"
        )
        payload["profile"]["topology"] = "bounded"

        with self.assertRaises(ValidationError):
            DatasetManifest.model_validate(payload)

    def test_manifest_rejects_changed_rng_stream_mapping(self) -> None:
        payload = DatasetManifest.build_complete(artifacts=self._records()).model_dump(
            mode="json"
        )
        payload["rng_stream_ids"]["tie_break"] = 9

        with self.assertRaises(ValidationError):
            DatasetManifest.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
