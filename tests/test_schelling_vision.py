"""Coordinate/Fraction oracle independent of production neighborhood calculations."""

import unittest
from fractions import Fraction

import numpy as np
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo

from mam_bench.simulations.schelling.models import Trajectory
from mam_bench.simulations.schelling.occupants import ModelControlledAgent, OrdinaryAgent
from mam_bench.simulations.schelling.reference import (
    CellGrid,
    neighbor_counts,
    reference_rng,
)
from mam_bench.simulations.schelling.settings import SchellingSettings
from mam_bench.simulations.schelling.simulation import SchellingSim
from schelling_support import identity as prompt_identity
from schelling_support import runtime_for


def neighborhood(place: int, size: int, radius: int) -> set[int]:
    row, col = divmod(place, size)
    return {
        ((row + dr) % size) * size + (col + dc) % size
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if dr or dc
    }


def quality(board: CellGrid, places: set[int], kind: int) -> Fraction:
    occupied = [int(board.flat[p]) for p in places if board.flat[p]]
    return Fraction(occupied.count(kind), len(occupied)) if occupied else Fraction(1)


def candidates(board: CellGrid, origin: int, available: set[int], radius: int) -> list[int]:
    size = len(board)
    kind = int(board.flat[origin])
    visible = neighborhood(origin, size, radius)
    current = quality(board, neighborhood(origin, size, 1), kind)
    for distance in range(1, radius + 1):
        choices = sorted(
            place
            for place in available & neighborhood(origin, size, distance)
            if quality(board, neighborhood(place, size, 1) & visible, kind) > current
        )
        if choices:
            return choices
    return []


def verify_trajectory(result: Trajectory, radius: int) -> None:
    """Reconstruct each frozen-board reservation using exact fraction comparisons."""
    order_rng = reference_rng(result.cell, result.seed_id, 1)
    tie_rng = reference_rng(result.cell, result.seed_id, 2)
    tolerance = Fraction(result.cell.tolerance.numerator, result.cell.tolerance.denominator)
    for step, board in enumerate(result.cell_types[:-1]):
        before = result.agent_locations[step]
        expected = before.copy()
        available = set(np.flatnonzero(board.ravel() == 0).tolist())
        unhappy = [
            identity
            for identity, origin in enumerate(before)
            if quality(board, neighborhood(int(origin), len(board), 1), int(board.flat[origin]))
            < tolerance
        ]
        for identity in order_rng.permutation(unhappy):
            choices = candidates(board, int(before[identity]), available, radius)
            if choices:
                destination = choices[0] if len(choices) == 1 else int(tie_rng.choice(choices))
                expected[identity] = destination
                available.remove(destination)
        np.testing.assert_array_equal(expected, result.agent_locations[step + 1])
        settled = np.zeros_like(board)
        settled.ravel()[expected] = result.agent_types
        np.testing.assert_array_equal(settled, result.cell_types[step + 1])


class StrictVisionTests(unittest.IsolatedAsyncioTestCase):
    def test_default_thirty_round_trajectory_matches_fraction_oracle(self) -> None:
        verify_trajectory(SchellingSim().run_reference(), 3)

    def test_configured_radii_match_fraction_oracle(self) -> None:
        for radius in (1, 2, 3):
            with self.subTest(radius=radius):
                settings = SchellingSettings(board_size=8, vision_radius=radius, max_transitions=5)
                result = SchellingSim(settings=settings).run_reference()
                verify_trajectory(result, radius)

    def test_hidden_cells_cannot_change_an_ordinary_choice(self) -> None:
        for radius in (1, 2, 3):
            sim = SchellingSim(settings=SchellingSettings(vision_radius=radius))
            board = sim.snapshot().cell_types
            for agent in sim.agents[:12]:
                assert isinstance(agent, OrdinaryAgent)
                visible = neighborhood(agent.position, len(board), radius) | {agent.position}
                altered = board.copy()
                for place in set(range(board.size)) - visible:
                    altered.flat[place] = 2 if board.flat[place] == 1 else 1
                choices: list[int | None] = []
                for frozen in (board, altered):
                    vacancies = np.flatnonzero(frozen.ravel() == 0).astype(np.uint16)
                    index = agent.choose_destination_index(
                        frozen,
                        vacancies,
                        np.ones(len(vacancies), dtype=np.bool_),
                        *neighbor_counts(frozen),
                        np.random.default_rng(73),
                    )
                    choices.append(None if index is None else int(vacancies[index]))
                self.assertEqual(choices[0], choices[1])

    async def test_nondefault_world_cohort_horizon_and_observation(self) -> None:
        settings = SchellingSettings(
            board_size=8, controlled_agent_count=4, vision_radius=2, max_transitions=2
        )
        sim = SchellingSim.for_model(runtime_for(), settings=settings)
        self.assertEqual(len(sim.controlled_agent_ids), 4)
        agent = sim.agents[sim.controlled_agent_ids[0]]
        assert isinstance(agent, ModelControlledAgent)
        observation = agent.observe()
        self.assertEqual(len(observation.neighborhood), 24)
        self.assertEqual(observation.horizon, 2)
        self.assertEqual(
            {n.location.row * 8 + n.location.column for n in observation.neighborhood},
            neighborhood(agent.position, 8, 2),
        )
        await sim.run()
        self.assertEqual(sim.rounds_completed, 2)
        assert sim.ordinary_trajectory is not None
        self.assertLessEqual(sim.ordinary_trajectory.rounds_completed, 2)
        self.assertEqual(sim.snapshot().cell_types.shape, (8, 8))

    async def test_model_can_move_to_vacancy_outside_visible_radius(self) -> None:
        async def move_far(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            _, agent_id = prompt_identity(messages)
            if agent_id == 0:
                row, column = divmod(destination, 20)
                return ModelResponse(parts=[ToolCallPart("move", {"row": row, "column": column})])
            return ModelResponse(parts=[ToolCallPart("stay", {})])

        sim = SchellingSim.for_model(runtime_for(move_far))
        visible = neighborhood(sim.agent_position(0), 20, 3)
        destination = next(
            int(place)
            for place in np.flatnonzero(sim.snapshot().cell_types.ravel() == 0)
            if int(place) not in visible
        )
        result = await sim.step()
        assert result is not None
        self.assertEqual(result.controlled_moves, ((0, destination),))

    async def test_model_move_bounds_follow_configured_board(self) -> None:
        async def check(size: int) -> None:
            calls = 0

            async def move(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                nonlocal calls
                _, agent_id = prompt_identity(messages)
                if agent_id != 0:
                    return ModelResponse(parts=[ToolCallPart("stay", {})])
                calls += 1
                row, column = (size, 0) if calls == 1 else divmod(destination, size)
                return ModelResponse(parts=[ToolCallPart("move", {"row": row, "column": column})])

            settings = SchellingSettings(
                board_size=size, controlled_agent_count=2, max_transitions=1
            )
            sim = SchellingSim.for_model(runtime_for(move), settings=settings)
            vacancies = np.flatnonzero(sim.snapshot().cell_types.ravel() == 0)
            destination = int(vacancies[vacancies >= 20 * size][0] if size > 20 else vacancies[0])
            result = await sim.step()
            assert result is not None
            with self.subTest(board_size=size):
                self.assertEqual(calls, 2)
                self.assertEqual(result.controlled_moves, ((0, destination),))

        for size in (8, 24):
            await check(size)
