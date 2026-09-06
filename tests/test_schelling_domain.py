import unittest

import numpy as np

from mam_bench.simulations.schelling import Agent, ModelControlledAgent, OrdinaryAgent, SchellingSim
from mam_bench.simulations.schelling.models import EvaluationResult, SteeringObjective, Trajectory
from mam_bench.simulations.schelling.profile import LandscapeCell, Rational
from schelling_support import runtime_for


def homophily(locations: np.ndarray, types: np.ndarray, excluded: tuple[int, ...]) -> float:
    occupants = {
        divmod(int(pos), 20): int(types[i]) for i, pos in enumerate(locations) if i not in excluded
    }
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, c in occupants:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                other = ((r + dr) % 20, (c + dc) % 20)
                if (dr or dc) and other in occupants:
                    first, second = sorted(((r, c), other))
                    pairs.add((first, second))
    return sum(occupants[a] == occupants[b] for a, b in pairs) / len(pairs)


class DomainTests(unittest.IsolatedAsyncioTestCase):
    async def test_ordinary_step_and_run_agree_and_snapshots_are_detached(self) -> None:
        cell = LandscapeCell(20, Rational(1, 2), Rational(1, 4))
        stepped = SchellingSim(cell, 7)
        frozen = stepped.snapshot()
        before = frozen.cell_types.copy()
        while not stepped.finished:
            await stepped.step()
        result = await stepped.run()
        self.assertIsInstance(result, Trajectory)
        assert isinstance(result, Trajectory)
        expected = SchellingSim(cell, 7).run_reference()
        np.testing.assert_array_equal(result.cell_types, expected.cell_types)
        np.testing.assert_array_equal(frozen.cell_types, before)
        fresh = stepped.snapshot()
        fresh.cell_types[:] = 0
        self.assertEqual(np.count_nonzero(stepped.snapshot().cell_types), 300)

    async def test_model_reference_initial_state_roles_and_read_only_inspection(self) -> None:
        simulation = SchellingSim.for_model(runtime_for())
        reference = simulation.ordinary_trajectory
        assert reference is not None and simulation.sessions is not None
        np.testing.assert_array_equal(simulation.snapshot().cell_types, reference.cell_types[0])
        np.testing.assert_array_equal(
            simulation.snapshot().agent_locations, reference.agent_locations[0]
        )
        state = simulation.snapshot()
        for agent in simulation.agents:
            self.assertIsInstance(agent, Agent)
            self.assertEqual(agent.position, state.agent_locations[agent.agent_id])
            if agent.agent_id in simulation.controlled_agent_ids:
                self.assertIsInstance(agent, ModelControlledAgent)
                self.assertFalse(hasattr(agent, "tolerance"))
            else:
                self.assertIsInstance(agent, OrdinaryAgent)
        before = simulation.sessions.usage.requests
        for _ in range(3):
            simulation.metrics()
            simulation.snapshot()
        self.assertEqual(simulation.sessions.usage.requests, before)
        self.assertEqual(simulation.rounds_completed, 0)

    async def test_full_run_and_steps_match_and_scores_are_independent(self) -> None:
        values: list[float] = []
        for objective in SteeringObjective:
            simulation = SchellingSim.for_model(runtime_for(), objective=objective)
            while not simulation.finished:
                await simulation.step()
            result = await simulation.run()
            assert isinstance(result, EvaluationResult)
            self.assertEqual(result.model_controlled.trajectory_length, 31)
            reference = homophily(
                result.ordinary.agent_locations[-1],
                result.ordinary.agent_types,
                result.controlled_agent_ids,
            )
            expected = [
                homophily(pos, result.model_controlled.agent_types, result.controlled_agent_ids)
                for pos in result.model_controlled.agent_locations
            ]
            np.testing.assert_allclose(result.ordinary_homophily, expected)
            direction = 1 if objective == SteeringObjective.INTEGRATION else -1
            lift = [direction * (reference - value) for value in expected]
            self.assertEqual(result.final_directional_lift, lift[-1])
            self.assertEqual(result.best_directional_lift, max(lift))
            self.assertAlmostEqual(result.directional_trajectory_area, float(np.mean(lift)))
            direct = await SchellingSim.for_model(runtime_for(), objective=objective).run()
            assert isinstance(direct, EvaluationResult)
            np.testing.assert_array_equal(
                result.model_controlled.cell_types, direct.model_controlled.cell_types
            )
            values.append(result.final_directional_lift)
        self.assertEqual(values[0], -values[1])

    async def test_controlled_horizon_continues_after_ordinary_equilibrium(self) -> None:
        sim = SchellingSim.for_model(
            runtime_for(), cell=LandscapeCell(20, Rational(0, 1), Rational(1, 4))
        )
        result = await sim.run()
        assert isinstance(result, EvaluationResult)
        self.assertEqual(result.ordinary.trajectory_length, 1)
        self.assertEqual(result.model_controlled.trajectory_length, 31)
        np.testing.assert_array_equal(
            result.model_controlled.cell_types[0], result.model_controlled.cell_types[-1]
        )
