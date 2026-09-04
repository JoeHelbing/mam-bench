import tempfile
import unittest
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np

from mam_bench.simulations.schelling.fixture import load_evaluation_reference
from mam_bench.simulations.schelling.profile import LandscapeCell, Rational
from mam_bench.simulations.schelling.utils.reference_data import (
    build_evaluation_reference,
    write_evaluation_reference,
)


class EvaluationReferenceFixtureTests(unittest.TestCase):
    def test_packaged_fixture_loads_the_fixed_case(self) -> None:
        resource = files("mam_bench.data").joinpath("schelling-reference-v2")

        with as_file(resource) as root:
            reference = load_evaluation_reference(root)

        self.assertEqual(reference.cell, LandscapeCell(20, Rational(3, 4), Rational(1, 4)))
        self.assertEqual(reference.seed_id, 50)
        self.assertEqual(reference.rounds_completed, 20)
        self.assertEqual(reference.initial_agent_locations.shape, (300,))
        self.assertEqual(reference.terminal_agent_locations.shape, (300,))
        generated = build_evaluation_reference(reference.cell, reference.seed_id)
        np.testing.assert_array_equal(
            reference.initial_agent_locations, generated.initial_agent_locations
        )
        np.testing.assert_array_equal(reference.terminal_cell_types, generated.terminal_cell_types)
        self.assertEqual(reference.masked_final_homophily, generated.masked_final_homophily)

    def test_fixture_round_trips_and_rejects_changed_archive(self) -> None:
        reference = build_evaluation_reference(
            LandscapeCell(20, Rational(3, 4), Rational(1, 4)), 50
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_evaluation_reference(root, reference)

            self.assertEqual(load_evaluation_reference(root).cell, reference.cell)
            archive = root / "evaluation-reference.npz"
            archive.write_bytes(archive.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_evaluation_reference(root)


if __name__ == "__main__":
    unittest.main()
