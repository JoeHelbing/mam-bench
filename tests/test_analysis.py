import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mam_bench.simulations.schelling.evaluation import SteeringObjective
from mam_bench.simulations.schelling.profile import TOLERANCES
from mam_bench.simulations.schelling.reference import TerminalStatus
from mam_bench_analysis.analysis import (
    ManifoldAnalysis,
    ManifoldPoint,
    SpotRole,
    final_satisfaction,
    select_test_spots,
    write_reference_landscape_analysis,
)
from mam_bench_analysis.evaluation_analysis import (
    MetricTrajectory,
    ModelEvaluationAnalysis,
    write_model_evaluation_analysis,
)


class FinalSatisfactionTests(unittest.TestCase):
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


class SpotSelectionTests(unittest.TestCase):
    @staticmethod
    def _point(
        tolerance_index: int,
        vacancy_index: int,
        satisfaction: float,
        *,
        standard_deviation: float = 0.0,
        blocked_rate: float = 0.0,
    ) -> ManifoldPoint:
        tolerance = TOLERANCES[tolerance_index]
        return ManifoldPoint(
            tolerance_index=tolerance_index,
            tolerance_fraction=str(tolerance),
            fractional_preference=tolerance.numerator / tolerance.denominator,
            vacancy_index=vacancy_index,
            empty_percentage=(vacancy_index + 2) * 5.0,
            mean_final_satisfaction=satisfaction,
            final_satisfaction_standard_deviation=standard_deviation,
            equilibrium_rate=1.0 - blocked_rate,
            blocked_rate=blocked_rate,
            horizon_exhausted_rate=0.0,
            median_trajectory_length=5.0,
        )

    def _points(self) -> tuple[ManifoldPoint, ...]:
        return (
            self._point(11, 3, 1.0),
            self._point(17, 3, 0.717, standard_deviation=0.291, blocked_rate=0.5),
            self._point(18, 4, 0.62, standard_deviation=0.2),
            self._point(19, 5, 0.53, standard_deviation=0.3, blocked_rate=0.75),
            self._point(20, 3, 0.10, standard_deviation=0.04, blocked_rate=1.0),
            self._point(21, 4, 0.07, standard_deviation=0.03, blocked_rate=1.0),
        )

    def test_selects_three_preferences_at_fixed_vacancy(self) -> None:
        points = self._points()

        spots = select_test_spots(points)

        self.assertEqual(tuple(spot.role for spot in spots), tuple(SpotRole))
        self.assertEqual(
            tuple((spot.point.tolerance_index, spot.point.vacancy_index) for spot in spots),
            ((11, 3), (17, 3), (20, 3)),
        )

    def test_writes_machine_readable_reference_and_model_analysis(self) -> None:
        points = self._points()
        analysis = ManifoldAnalysis(points=points, selected_spots=select_test_spots(points))
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            json_path, csv_path = write_reference_landscape_analysis(analysis, output)
            trajectory = MetricTrajectory(
                rounds=(0, 1),
                ordinary_homophily=(0.5, 0.75),
                ordinary_satisfaction=(0.15, 0.6),
            )
            evaluation = ModelEvaluationAnalysis(
                case_path="results/case",
                model_name="qwen/qwen3.8-27b",
                provider="openrouter",
                provider_backend="phala",
                objective=SteeringObjective.INTEGRATION,
                tolerance_fraction="3/4",
                vacancy_percentage=25.0,
                seed_id=22,
                scored_ordinary_agent_count=284,
                reference_terminal_status=TerminalStatus.EQUILIBRIUM,
                reference_rounds_completed=1,
                model_rounds_completed=1,
                initial_homophily=0.5,
                initial_satisfaction=0.15,
                reference_final_homophily=0.99,
                reference_final_satisfaction=1.0,
                model_final_homophily=0.75,
                model_final_satisfaction=0.6,
                final_directional_lift=0.24,
                directional_trajectory_area=0.3,
                relative_final_homophily_reduction=0.24,
                best_directional_lift=0.49,
                reference_trajectory=trajectory,
                model_trajectory=trajectory,
                accepted_actor_moves=173,
                actor_stays=122,
                actor_collisions=22,
                occupied_destination_failures=3,
                policy_error_outcomes=0,
                ordinary_moves=119,
                accepted_actor_moves_by_round=(15,),
                ordinary_moves_by_round=(25,),
                coordination_post_count=320,
                empty_coordination_posts=4,
                median_coordination_characters=1332.0,
                coordination_posts_over_2000_characters=90,
                coordination_posts_over_5000_characters=24,
                longest_coordination_post_characters=10058,
                longest_coordination_post_round=3,
                longest_coordination_post_actor=2,
                collision_language_posts=275,
                hold_language_posts=270,
                boundary_language_posts=190,
                anchor_language_posts=106,
                opposite_type_language_posts=40,
                total_requests=1281,
                total_input_tokens=63636863,
                total_output_tokens=560126,
            )
            evaluation_path = write_model_evaluation_analysis(
                evaluation,
                output / "model-evaluation-analysis.json",
            )

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(len(payload["selected_spots"]), 3)
            self.assertIn("fractional_preference", csv_path.read_text(encoding="utf-8"))
            reloaded_evaluation = ModelEvaluationAnalysis.model_validate_json(
                evaluation_path.read_text(encoding="utf-8")
            )
            self.assertEqual(reloaded_evaluation, evaluation)


if __name__ == "__main__":
    unittest.main()
