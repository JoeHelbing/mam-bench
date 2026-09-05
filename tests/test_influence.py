import unittest

import numpy as np

from mam_bench.simulations.schelling.profile import LandscapeCell, Rational
from mam_bench.simulations.schelling.reference import TerminalStatus
from mam_bench.simulations.schelling.runtime import (
    influence_actor_ids,
    ordinary_edge_homophily,
)
from mam_bench.simulations.schelling.utils.reference_data import build_evaluation_reference


class InfluenceMetricTests(unittest.TestCase):
    def test_fixed_actor_ids_are_symmetric(self) -> None:
        self.assertEqual(influence_actor_ids(300), (*range(8), *range(150, 158)))

    def test_edge_homophily_masks_would_be_actor_edges(self) -> None:
        locations = np.asarray([0, 1, 2, 3], dtype=np.uint16)
        agent_types = np.asarray([1, 1, 2, 2], dtype=np.uint8)

        score = ordinary_edge_homophily(
            locations,
            agent_types,
            excluded_agent_ids=(0,),
            grid_size=20,
        )

        self.assertEqual(score, 0.5)

    def test_counterfactual_reference_uses_the_held_out_seed(self) -> None:
        reference = build_evaluation_reference(
            LandscapeCell(20, Rational(3, 4), Rational(1, 4)), seed_id=50
        )

        self.assertEqual(reference.seed_id, 50)
        self.assertEqual(reference.terminal_status, TerminalStatus.EQUILIBRIUM)
        self.assertEqual(reference.rounds_completed, 20)
        self.assertAlmostEqual(reference.masked_final_homophily, 0.9748, places=4)
        self.assertEqual(reference.masked_agent_count, 284)


if __name__ == "__main__":
    unittest.main()
