import unittest

from mam_bench.simulations.schelling.simulation import SchellingBenchmarkSimulation


class SchellingPackageTests(unittest.TestCase):
    def test_exports_the_built_in_simulation(self) -> None:
        simulation = SchellingBenchmarkSimulation()
        self.assertEqual(simulation.simulation_id, "schelling-influence-pilot-v1")
        self.assertEqual(simulation.simulation_version, "schelling-influence-v2")


if __name__ == "__main__":
    unittest.main()
