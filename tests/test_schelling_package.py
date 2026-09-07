import unittest

from mam_bench.simulations.schelling import SchellingSim


class SchellingPackageTests(unittest.TestCase):
    def test_exports_the_built_in_simulation(self) -> None:
        simulation = SchellingSim()
        self.assertEqual(simulation.simulation_id, "schelling-influence-pilot-v1")
        self.assertEqual(simulation.simulation_version, "schelling-influence-v4")


if __name__ == "__main__":
    unittest.main()
