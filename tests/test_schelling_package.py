import importlib.util
import unittest

import mam_bench
from mam_bench.simulations.schelling import (
    dataset,
    evaluation,
    interaction,
    profile,
    reference,
)
from mam_bench.simulations.schelling.evidence import validate_evaluation_artifact
from mam_bench.simulations.schelling.scoring import ordinary_edge_homophily
from mam_bench.simulations.schelling.simulation import SchellingBenchmarkSimulation


class SchellingPackageContractionTests(unittest.TestCase):
    def test_canonical_package_owns_the_complete_schelling_lifecycle(self) -> None:
        self.assertTrue(profile.PROFILE.profile_version.startswith("schelling-reference"))
        self.assertEqual(reference.run_reference.__module__, reference.__name__)
        self.assertEqual(dataset.validate_dataset.__module__, dataset.__name__)
        self.assertIs(
            evaluation.validate_evaluation_artifact,
            validate_evaluation_artifact,
        )
        self.assertIs(
            evaluation.ordinary_edge_homophily,
            ordinary_edge_homophily,
        )
        self.assertEqual(
            interaction.RuntimeInfluenceTeam.__module__,
            interaction.__name__,
        )
        self.assertEqual(
            SchellingBenchmarkSimulation.descriptor.simulation_id,
            "schelling-influence-pilot-v1",
        )

    def test_obsolete_root_compatibility_modules_are_absent(self) -> None:
        self.assertFalse(hasattr(mam_bench, "PROFILE"))
        self.assertFalse(hasattr(mam_bench, "run_reference"))
        for module_name in (
            "mam_bench.dataset",
            "mam_bench.influence",
            "mam_bench.profile",
            "mam_bench.pydantic_team",
            "mam_bench.reference",
        ):
            with self.subTest(module_name=module_name):
                self.assertIsNone(importlib.util.find_spec(module_name))


if __name__ == "__main__":
    unittest.main()
