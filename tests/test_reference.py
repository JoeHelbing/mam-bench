import copy
import unittest

import numpy as np

from mam_bench.simulations.schelling.profile import (
    BOARD_SIZES,
    LANDSCAPE_CELL_COUNT,
    LANDSCAPE_SEED_IDS,
    MAX_TRANSITIONS,
    TOLERANCES,
    VACANCY_FRACTIONS,
    LandscapeCell,
    Rational,
    landscape_cells,
)
from mam_bench.simulations.schelling.reference import (
    TerminalStatus,
    _choose_nearest_index,  # pyright: ignore[reportPrivateUsage]
    candidate_is_satisfactory,
    evaluate_satisfaction,
    run_reference,
    toroidal_chebyshev_distance,
)


class SchellingReferenceTests(unittest.TestCase):
    def test_tolerances_are_exact_farey_sequence_of_order_eight(self) -> None:
        expected = (
            "0/1",
            "1/8",
            "1/7",
            "1/6",
            "1/5",
            "1/4",
            "2/7",
            "1/3",
            "3/8",
            "2/5",
            "3/7",
            "1/2",
            "4/7",
            "3/5",
            "5/8",
            "2/3",
            "5/7",
            "3/4",
            "4/5",
            "5/6",
            "6/7",
            "7/8",
            "1/1",
        )

        self.assertEqual(tuple(str(value) for value in TOLERANCES), expected)

    def test_v2_cartesian_contract_is_exact(self) -> None:
        self.assertEqual(BOARD_SIZES, (20, 60, 100))
        self.assertEqual(
            tuple(str(value) for value in VACANCY_FRACTIONS),
            ("1/100", "1/40", "1/20", "1/10", "1/5", "1/4", "3/10", "2/5", "1/2"),
        )
        self.assertEqual(LANDSCAPE_SEED_IDS, tuple(range(50)))
        self.assertEqual(MAX_TRANSITIONS, 30)
        self.assertEqual(len(landscape_cells()), LANDSCAPE_CELL_COUNT)
        self.assertEqual(LANDSCAPE_CELL_COUNT, 621)

    def test_population_scales_to_every_board_size(self) -> None:
        for board_size in BOARD_SIZES:
            with self.subTest(board_size=board_size):
                trajectory = run_reference(
                    LandscapeCell(board_size, Rational(0, 1), Rational(1, 2)),
                    seed_id=0,
                    max_transitions=0,
                )
                self.assertEqual(trajectory.cell_types.shape, (1, board_size, board_size))
                self.assertEqual(trajectory.agent_locations.shape, (1, board_size**2 // 2))

    def test_toroidal_chebyshev_distance_wraps_at_grid_edges(self) -> None:
        self.assertEqual(toroidal_chebyshev_distance(0, 399, size=20), 1)
        self.assertEqual(toroidal_chebyshev_distance(0, 210, size=20), 10)

    def test_zero_neighbor_agent_is_satisfied(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1

        satisfied = evaluate_satisfaction(grid, TOLERANCES[11])

        self.assertTrue(satisfied[2, 2])

    def test_fractional_tolerance_accepts_exact_equality(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 2] = 1
        grid[1, 2] = 1
        grid[2, 1] = 2

        at_half = evaluate_satisfaction(grid, TOLERANCES[11])
        above_half = evaluate_satisfaction(grid, TOLERANCES[12])

        self.assertTrue(at_half[2, 2])
        self.assertFalse(above_half[2, 2])

    def test_candidate_evaluation_treats_movers_origin_as_empty(self) -> None:
        grid = np.zeros((5, 5), dtype=np.uint8)
        grid[2, 1] = 1
        grid[1, 2] = 2

        self.assertFalse(
            candidate_is_satisfactory(
                grid,
                origin=11,
                destination=12,
                agent_type=1,
                tolerance=TOLERANCES[11],
            )
        )

    def test_single_nearest_candidate_does_not_consume_tie_rng(self) -> None:
        rng = np.random.Generator(np.random.PCG64(123))
        state_before = copy.deepcopy(rng.bit_generator.state)

        selected = _choose_nearest_index(np.asarray([7], dtype=np.int64), rng)

        self.assertEqual(selected, 7)
        self.assertEqual(rng.bit_generator.state, state_before)

    def test_initial_grid_is_paired_across_tolerances(self) -> None:
        low_tolerance = run_reference(
            LandscapeCell(20, Rational(0, 1), Rational(1, 4)),
            seed_id=4,
            max_transitions=1,
        )
        high_tolerance = run_reference(
            LandscapeCell(20, Rational(4, 5), Rational(1, 4)),
            seed_id=4,
            max_transitions=1,
        )

        np.testing.assert_array_equal(low_tolerance.cell_types[0], high_tolerance.cell_types[0])
        np.testing.assert_array_equal(
            low_tolerance.agent_locations[0], high_tolerance.agent_locations[0]
        )

    def test_initial_grid_has_exact_vacancy_and_group_counts(self) -> None:
        trajectory = run_reference(
            LandscapeCell(20, Rational(0, 1), Rational(1, 4)),
            seed_id=0,
        )
        initial = trajectory.cell_types[0]

        self.assertEqual(np.count_nonzero(initial == 0), 100)
        self.assertEqual(np.count_nonzero(initial == 1), 150)
        self.assertEqual(np.count_nonzero(initial == 2), 150)
        self.assertEqual(trajectory.terminal_status, TerminalStatus.EQUILIBRIUM)
        self.assertEqual(trajectory.trajectory_length, 1)

    def test_blocked_run_does_not_append_a_duplicate_terminal_state(self) -> None:
        trajectory = run_reference(
            LandscapeCell(20, Rational(1, 1), Rational(1, 10)),
            seed_id=0,
        )

        self.assertEqual(trajectory.terminal_status, TerminalStatus.BLOCKED)
        self.assertEqual(trajectory.trajectory_length, 2)
        self.assertEqual(trajectory.rounds_completed, 1)

    def test_shortened_test_horizon_classifies_a_still_movable_state(self) -> None:
        trajectory = run_reference(
            LandscapeCell(20, Rational(1, 2), Rational(1, 4)),
            seed_id=0,
            max_transitions=0,
        )

        self.assertEqual(trajectory.terminal_status, TerminalStatus.HORIZON_EXHAUSTED)
        self.assertEqual(trajectory.trajectory_length, 1)

    def test_same_seed_and_cell_reproduce_bit_exact_trajectory(self) -> None:
        cell = LandscapeCell(20, Rational(1, 2), Rational(1, 4))

        first = run_reference(cell, seed_id=7, max_transitions=10)
        second = run_reference(cell, seed_id=7, max_transitions=10)

        self.assertEqual(first.terminal_status, second.terminal_status)
        np.testing.assert_array_equal(first.cell_types, second.cell_types)
        np.testing.assert_array_equal(first.agent_locations, second.agent_locations)

    def test_every_applied_destination_was_vacant_at_start_of_round(self) -> None:
        trajectory = run_reference(
            LandscapeCell(20, Rational(2, 3), Rational(1, 4)),
            seed_id=2,
            max_transitions=5,
        )
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
