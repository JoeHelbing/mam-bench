import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from mam_bench.benchmark import RuntimeInfo
from mam_bench.simulations.schelling.models import (
    ActorWaveContext,
    CoordinationPost,
    InfluenceEvaluationConfig,
    MoveDecision,
    MoveProposal,
    SteeringObjective,
)
from mam_bench.simulations.schelling.profile import LandscapeCell, Rational
from mam_bench.simulations.schelling.reference import EMPTY_CELL, TerminalStatus
from mam_bench.simulations.schelling.runtime import (
    influence_actor_ids,
    ordinary_edge_homophily,
    run_model_evaluation,
)
from mam_bench.simulations.schelling.utils.reference_data import build_evaluation_reference


class FixedInfluenceTeam:
    runtime_info = RuntimeInfo(
        model_id="fixed-team",
        provider="openrouter",
        model="test/fixed-team",
        endpoint="https://example.test/v1",
    )

    def __init__(self) -> None:
        self.coordination_calls = 0
        self.movement_calls = 0
        self.maximum_retained_states = 0

    async def coordinate(
        self, contexts: tuple[ActorWaveContext, ...]
    ) -> tuple[CoordinationPost, ...]:
        self.coordination_calls += 1
        self.maximum_retained_states = max(
            self.maximum_retained_states,
            *(len(context.cell_type_states) for context in contexts),
        )
        return tuple(
            CoordinationPost(
                actor_id=context.actor_id,
                text=f"round {context.round_number}",
                request_count=2,
                input_tokens=10,
                output_tokens=2,
                latency_seconds=0.01,
            )
            for context in contexts
        )

    async def move(self, contexts: tuple[ActorWaveContext, ...]) -> tuple[MoveProposal, ...]:
        self.movement_calls += 1
        self.maximum_retained_states = max(
            self.maximum_retained_states,
            *(len(context.cell_type_states) for context in contexts),
        )
        vacancy = int(np.flatnonzero(contexts[0].current_cell_types.ravel() == EMPTY_CELL)[0])
        row, column = divmod(vacancy, contexts[0].current_cell_types.shape[0])
        return tuple(
            MoveProposal(
                actor_id=context.actor_id,
                decision=MoveDecision(stay=False, row=row, column=column),
                request_count=2,
                input_tokens=10,
                output_tokens=2,
                latency_seconds=0.01,
            )
            for context in contexts
        )


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


class ModelEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_team_runs_twenty_rounds_and_writes_useful_artifacts(self) -> None:
        team = FixedInfluenceTeam()
        config = InfluenceEvaluationConfig(
            cell=LandscapeCell(20, Rational(3, 4), Rational(1, 4)),
            seed_id=50,
            objective=SteeringObjective.INTEGRATION,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case"

            result = await run_model_evaluation(
                config,
                team,
                output,
                reference=build_evaluation_reference(config.cell, config.seed_id),
            )

            self.assertEqual(result.rounds_completed, 20)
            self.assertEqual(result.cell_types.shape, (21, 20, 20))
            self.assertEqual(result.agent_locations.shape, (21, 300))
            self.assertEqual(team.coordination_calls, 20)
            self.assertEqual(team.movement_calls, 20)
            self.assertEqual(team.maximum_retained_states, 3)
            self.assertEqual(len(result.coordination_posts), 20 * 16)
            self.assertEqual(result.actor_collision_count, 20 * 15)
            self.assertEqual(
                set(path.name for path in output.iterdir()),
                {"run.json", "trajectory.npz", "coordination.json", "moves.json"},
            )
            summary = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["runtime"]["agent_settings"]["concurrency"], 4)
            self.assertEqual(summary["final_directional_lift"], result.final_directional_lift)
            self.assertEqual(
                summary["best_directional_lift"],
                max(result.reference.masked_final_homophily - result.ordinary_homophily),
            )
            with np.load(output / "trajectory.npz", allow_pickle=False) as trajectory:
                np.testing.assert_array_equal(trajectory["cell_types"], result.cell_types)
            posts = json.loads((output / "coordination.json").read_text(encoding="utf-8"))
            moves = json.loads((output / "moves.json").read_text(encoding="utf-8"))
            self.assertEqual(len(posts), 20 * 16)
            self.assertEqual(len(moves), 20 * 16)


if __name__ == "__main__":
    unittest.main()
