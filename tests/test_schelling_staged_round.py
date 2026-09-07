import asyncio
import unittest

import numpy as np
from pydantic_ai.messages import ModelMessage, ModelResponse, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo

from mam_bench.simulations.schelling import ModelControlledAgent, SchellingSim
from schelling_support import identity, runtime_for


class RoundTests(unittest.IsolatedAsyncioTestCase):
    async def test_output_token_limit_stays_then_actor_can_move_next_round(self) -> None:
        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            _, agent_id = identity(messages)
            if agent_id == 0:
                if sim.rounds_completed == 0:
                    return ModelResponse(parts=[ThinkingPart("unfinished")], finish_reason="length")
                row, column = divmod(destination, 20)
                return ModelResponse(parts=[ToolCallPart("move", {"row": row, "column": column})])
            return ModelResponse(parts=[ToolCallPart("stay", {})])

        sim = SchellingSim.for_model(runtime_for(script))
        origin = sim.agent_position(0)
        first = await sim.step()
        assert first is not None
        self.assertEqual(first.controlled_moves, ())
        self.assertEqual(sim.agent_position(0), origin)
        destination = int(np.flatnonzero(sim.snapshot().cell_types.ravel() == 0)[0])
        second = await sim.step()
        assert second is not None
        self.assertEqual(second.controlled_moves, ((0, destination),))
        self.assertEqual(sim.rounds_completed, 2)

    async def test_rolling_reservations_collisions_and_frozen_settlement(self) -> None:
        release = asyncio.Event()
        admitted = asyncio.Event()
        calls: dict[int, int] = {}
        order: list[int] = []
        active = 0
        maximum = 0

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal active, maximum
            _, agent_id = identity(messages)
            if agent_id not in calls:
                order.append(agent_id)
            calls[agent_id] = calls.get(agent_id, 0) + 1
            slot = order.index(agent_id)
            active += 1
            maximum = max(maximum, active)
            try:
                if slot == 0:
                    await release.wait()
                if len(order) == 16:
                    admitted.set()
                if slot == 1:
                    destination = vacancies[0]
                elif slot == 2:
                    destination = vacancies[0] if calls[agent_id] == 1 else vacancies[1]
                elif slot == 3 and calls[agent_id] == 1:
                    destination = origin
                else:
                    return ModelResponse(parts=[ToolCallPart("stay", {})])
                row, column = divmod(int(destination), 20)
                return ModelResponse(parts=[ToolCallPart("move", {"row": row, "column": column})])
            finally:
                active -= 1

        sim = SchellingSim.for_model(runtime_for(script, concurrency=2))
        initial = sim.snapshot()
        vacancies = np.flatnonzero(initial.cell_types.ravel() == 0)
        origin = int(initial.agent_locations[sim.controlled_agent_ids[0]])
        task = asyncio.create_task(sim.step())
        try:
            await asyncio.wait_for(admitted.wait(), 10)
            np.testing.assert_array_equal(sim.snapshot().cell_types, initial.cell_types)
            np.testing.assert_array_equal(sim.snapshot().agent_locations, initial.agent_locations)
            self.assertEqual(sim.remaining_vacancies, 98)
        finally:
            release.set()
        result = await asyncio.wait_for(task, 10)
        assert result is not None
        self.assertEqual(maximum, 2)
        self.assertEqual(result.admission_order, tuple(order))
        self.assertEqual(
            result.controlled_moves, ((order[1], vacancies[0]), (order[2], vacancies[1]))
        )
        self.assertEqual(calls[order[2]], 2)
        self.assertEqual(calls[order[3]], 2)
        destinations = [d for _, d in (*result.controlled_moves, *result.ordinary_moves)]
        self.assertEqual(len(destinations), len(set(destinations)))
        self.assertTrue(all(initial.cell_types.flat[d] == 0 for d in destinations))
        for agent in sim.agents:
            self.assertEqual(agent.position, result.agent_locations[agent.agent_id])

    async def test_retry_exhaustion_stays_and_sessions_continue(self) -> None:
        async def invalid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart("move", {"row": -1, "column": 0})])

        sim = SchellingSim.for_model(runtime_for(invalid))
        initial = sim.snapshot()
        first = await sim.step()
        assert first is not None and sim.sessions is not None
        self.assertEqual(first.controlled_moves, ())
        history = len(sim.sessions.history("0"))
        second = await sim.step()
        assert second is not None
        self.assertEqual(second.controlled_moves, ())
        self.assertGreater(len(sim.sessions.history("0")), history)
        for agent_id in sim.controlled_agent_ids:
            self.assertEqual(sim.agent_position(agent_id), initial.agent_locations[agent_id])

    async def test_terminal_stay_skips_batched_post_and_observation_is_local(self) -> None:
        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart("post_message", {"text": "must not be posted"}),
                    ToolCallPart("stay", {}),
                ]
            )

        sim = SchellingSim.for_model(runtime_for(script))
        agent = sim.agents[0]
        assert isinstance(agent, ModelControlledAgent)
        snapshot = sim.snapshot()
        observation = agent.observe()
        self.assertEqual(len(observation.neighborhood), 48)
        row, col = divmod(agent.position, 20)
        expected = {
            ((row + dr) % 20, (col + dc) % 20)
            for dr in range(-3, 4)
            for dc in range(-3, 4)
            if dr or dc
        }
        self.assertEqual(
            {(n.location.row, n.location.column) for n in observation.neighborhood}, expected
        )
        self.assertEqual(observation.current_homophily, sim.metrics().ordinary_homophily)
        self.assertFalse(hasattr(observation, "cell_types"))
        await sim.step()
        assert sim.sessions is not None
        self.assertEqual((await sim.sessions.board.read("inspector")).content, "")
        for agent_id in sim.controlled_agent_ids:
            self.assertEqual(sim.agent_position(agent_id), snapshot.agent_locations[agent_id])
