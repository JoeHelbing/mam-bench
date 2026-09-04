import tempfile
import unittest
from pathlib import Path

import numpy as np

from mam_bench.simulations.schelling.profile import TOLERANCES
from mam_bench.simulations.schelling.utils.analysis import (
    LandscapePoint,
    final_satisfaction,
    write_landscape,
)


class SchellingAnalysisTests(unittest.TestCase):
    def test_final_satisfaction_counts_only_occupied_cells(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1

        self.assertEqual(final_satisfaction(grid, TOLERANCES[11]), 1.0)

    def test_final_satisfaction_uses_the_cells_exact_tolerance(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1
        grid[1, 2] = 1
        grid[2, 1] = 2

        self.assertEqual(final_satisfaction(grid, TOLERANCES[11]), 2 / 3)
        self.assertEqual(final_satisfaction(grid, TOLERANCES[12]), 0.0)

    def test_writes_one_json_object_per_landscape_point(self) -> None:
        point = LandscapePoint(
            board_size=20,
            tolerance_fraction="3/4",
            fractional_preference=0.75,
            vacancy_fraction="1/4",
            empty_percentage=25.0,
            mean_final_satisfaction=0.7,
            final_satisfaction_standard_deviation=0.1,
            equilibrium_rate=0.5,
            blocked_rate=0.5,
            horizon_exhausted_rate=0.0,
            median_trajectory_length=15.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "landscape.jsonl"

            write_landscape((point,), path)

            self.assertEqual(path.read_text(encoding="utf-8").count("\n"), 1)


if __name__ == "__main__":
    unittest.main()
