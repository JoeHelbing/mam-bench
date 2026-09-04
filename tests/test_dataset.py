import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydantic import ValidationError

from mam_bench.simulations.schelling.profile import (
    LANDSCAPE_CELL_COUNT,
    LandscapeCell,
    Rational,
    landscape_cells,
)
from mam_bench.simulations.schelling.reference import TerminalStatus
from mam_bench.simulations.schelling.utils.dataset import (
    ArtifactRecord,
    build_cell_archive,
    cell_filename,
    read_manifest,
    validate_cell_archive,
)


class CellArchiveTests(unittest.TestCase):
    def test_tolerance_zero_cell_round_trips_with_safe_typed_arrays(self) -> None:
        cell = LandscapeCell(20, Rational(0, 1), Rational(1, 4))

        archive = build_cell_archive(cell)
        arrays = validate_cell_archive(archive, cell)

        self.assertEqual(arrays["cell_types"].dtype, np.dtype(np.uint8))
        self.assertEqual(arrays["cell_types"].shape, (50, 31, 20, 20))
        self.assertEqual(arrays["agent_locations"].dtype, np.dtype(np.uint16))
        self.assertEqual(arrays["agent_locations"].shape, (50, 31, 300))
        self.assertTrue(np.all(arrays["trajectory_lengths"] == 1))
        self.assertTrue(np.all(arrays["terminal_status"] == int(TerminalStatus.EQUILIBRIUM)))
        self.assertTrue(np.all(arrays["cell_types"][:, 1:] == 255))
        self.assertTrue(np.all(arrays["agent_locations"][:, 1:] == 65535))

    def test_validation_rejects_wrong_cell_metadata(self) -> None:
        archive = build_cell_archive(LandscapeCell(20, Rational(0, 1), Rational(1, 4)))

        with self.assertRaises(ValueError):
            validate_cell_archive(archive, LandscapeCell(20, Rational(1, 8), Rational(1, 4)))


class ManifestTests(unittest.TestCase):
    @staticmethod
    def _record(index: int) -> ArtifactRecord:
        cell = landscape_cells()[index]
        return ArtifactRecord(
            board_size=cell.board_size,
            tolerance=str(cell.tolerance),
            vacancy_fraction=str(cell.vacancy_fraction),
            vacancy_count=cell.vacancy_count,
            run_count=50,
            path=cell_filename(cell),
            byte_size=10,
            sha256=f"{index:064x}",
        )

    @classmethod
    def _records(cls) -> list[ArtifactRecord]:
        return [cls._record(index) for index in range(LANDSCAPE_CELL_COUNT)]

    def test_reads_one_complete_artifact_record_per_line(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manifest.jsonl"
            path.write_text(
                "".join(f"{record.model_dump_json()}\n" for record in self._records()),
                encoding="utf-8",
            )

            self.assertEqual(read_manifest(path), tuple(self._records()))

    def test_artifact_rejects_metadata_that_disagrees_with_its_coordinates(self) -> None:
        payload = self._record(0).model_dump()
        payload["tolerance"] = "1/2"

        with self.assertRaises(ValidationError):
            ArtifactRecord.model_validate(payload)


if __name__ == "__main__":
    unittest.main()
