import copy
import unittest

import numpy as np

from mam_bench.simulations.schelling.profile import LandscapeCell, Rational
from mam_bench.simulations.schelling.reference import (
    TerminalStatus,
    candidate_mask,
    choose_nearest_index,
    evaluate_satisfaction,
    neighbor_counts,
    toroidal_chebyshev_distance,
)
from mam_bench.simulations.schelling.simulation import SchellingSim


class SchellingReferenceTests(unittest.TestCase):
    def test_population_scales_to_every_board_size(self) -> None:
        for board_size in (20, 60, 100):
            with self.subTest(board_size=board_size):
                trajectory = SchellingSim(
                    LandscapeCell(board_size, Rational(0, 1), Rational(1, 2)),
                    seed_id=0,
                    max_transitions=0,
                ).run_reference()
                self.assertEqual(trajectory.cell_types.shape, (1, board_size, board_size))
                self.assertEqual(trajectory.agent_locations.shape, (1, board_size**2 // 2))

    def test_toroidal_chebyshev_distance_wraps_at_grid_edges(self) -> None:
        self.assertEqual(toroidal_chebyshev_distance(0, 399, size=20), 1)
        self.assertEqual(toroidal_chebyshev_distance(0, 210, size=20), 10)

    def test_zero_neighbor_agent_is_satisfied(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1
        satisfied = evaluate_satisfaction(grid, Rational(1, 2))
        self.assertTrue(satisfied[2, 2])

    def test_fractional_tolerance_accepts_exact_equality(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1
        grid[1, 2] = 1
        grid[2, 1] = 2
        at_half = evaluate_satisfaction(grid, Rational(1, 2))
        above_half = evaluate_satisfaction(grid, Rational(4, 7))
        self.assertTrue(at_half[2, 2])
        self.assertFalse(above_half[2, 2])

    def test_candidate_evaluation_treats_movers_origin_as_empty(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 1] = 1
        grid[1, 2] = 2
        self.assertFalse(
            candidate_mask(
                grid,
                np.asarray([12], dtype=np.uint16),
                11,
                1,
                Rational(1, 2),
                *neighbor_counts(grid),
            )[0]
        )

    def test_single_nearest_candidate_does_not_consume_tie_rng(self) -> None:
        rng = np.random.Generator(np.random.PCG64(123))
        state_before = copy.deepcopy(rng.bit_generator.state)
        selected = choose_nearest_index(np.asarray([7], dtype=np.int64), rng)
        self.assertEqual(selected, 7)
        self.assertEqual(rng.bit_generator.state, state_before)

    def test_initial_grid_is_paired_across_tolerances(self) -> None:
        low_tolerance = SchellingSim(
            LandscapeCell(20, Rational(0, 1), Rational(1, 4)), seed_id=4, max_transitions=1
        ).run_reference()
        high_tolerance = SchellingSim(
            LandscapeCell(20, Rational(4, 5), Rational(1, 4)), seed_id=4, max_transitions=1
        ).run_reference()
        np.testing.assert_array_equal(low_tolerance.cell_types[0], high_tolerance.cell_types[0])
        np.testing.assert_array_equal(
            low_tolerance.agent_locations[0], high_tolerance.agent_locations[0]
        )

    def test_initial_grid_has_exact_vacancy_and_group_counts(self) -> None:
        trajectory = SchellingSim(
            LandscapeCell(20, Rational(0, 1), Rational(1, 4)), seed_id=0
        ).run_reference()
        initial = trajectory.cell_types[0]
        self.assertEqual(np.count_nonzero(initial == 0), 100)
        self.assertEqual(np.count_nonzero(initial == 1), 150)
        self.assertEqual(np.count_nonzero(initial == 2), 150)
        self.assertEqual(trajectory.terminal_status, TerminalStatus.EQUILIBRIUM)
        self.assertEqual(trajectory.trajectory_length, 1)

    def test_blocked_run_does_not_append_a_duplicate_terminal_state(self) -> None:
        trajectory = SchellingSim(
            LandscapeCell(20, Rational(1, 1), Rational(1, 10)), seed_id=0
        ).run_reference()
        self.assertEqual(trajectory.terminal_status, TerminalStatus.BLOCKED)
        self.assertEqual(trajectory.trajectory_length, 2)
        self.assertEqual(trajectory.rounds_completed, 1)

    def test_shortened_test_horizon_classifies_a_still_movable_state(self) -> None:
        trajectory = SchellingSim(
            LandscapeCell(20, Rational(1, 2), Rational(1, 4)), seed_id=0, max_transitions=0
        ).run_reference()
        self.assertEqual(trajectory.terminal_status, TerminalStatus.HORIZON_EXHAUSTED)
        self.assertEqual(trajectory.trajectory_length, 1)

    def test_same_seed_and_cell_reproduce_bit_exact_trajectory(self) -> None:
        cell = LandscapeCell(20, Rational(1, 2), Rational(1, 4))
        first = SchellingSim(cell, seed_id=7, max_transitions=10).run_reference()
        second = SchellingSim(cell, seed_id=7, max_transitions=10).run_reference()
        self.assertEqual(first.terminal_status, second.terminal_status)
        np.testing.assert_array_equal(first.cell_types, second.cell_types)
        np.testing.assert_array_equal(first.agent_locations, second.agent_locations)

    def test_every_applied_destination_was_vacant_at_start_of_round(self) -> None:
        trajectory = SchellingSim(
            LandscapeCell(20, Rational(2, 3), Rational(1, 4)), seed_id=2, max_transitions=5
        ).run_reference()
        self.assertGreater(trajectory.trajectory_length, 1)
        for step in range(1, trajectory.trajectory_length):
            before_locations = trajectory.agent_locations[step - 1]
            after_locations = trajectory.agent_locations[step]
            before_occupied = set(int(value) for value in before_locations)
            moved = before_locations != after_locations
            moved_destinations = [int(value) for value in after_locations[moved]]
            self.assertEqual(len(moved_destinations), len(set(moved_destinations)))
            self.assertTrue(all(value not in before_occupied for value in moved_destinations))


if __name__ == "__main__":
    unittest.main()
