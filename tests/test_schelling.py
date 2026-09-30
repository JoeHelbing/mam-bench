"""Behavior checks through Schelling's world and paired-evaluation interfaces."""

import asyncio
import json
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from typing import cast

import numpy as np
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from mam_bench.simulations.schelling.agents import ModelControlledAgent, OrdinaryAgent
from mam_bench.simulations.schelling.results import EvaluationResult, Outcome
from mam_bench.simulations.schelling.simulation import SchellingSim
from support import observation, records, runtime, schelling


def neighborhood(place: int, size: int, radius: int) -> set[int]:
    row, column = divmod(place, size)
    return {
        ((row + dr) % size) * size + (column + dc) % size
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if dr or dc
    }


def quality(cells: list[int], places: set[int], kind: int) -> Fraction:
    occupied = [cells[p] for p in places if cells[p]]
    return Fraction(occupied.count(kind), len(occupied)) if occupied else Fraction(1)


class SchellingTests(unittest.IsolatedAsyncioTestCase):
    def test_satisfaction_at_float_thresholds_and_without_neighbors(self) -> None:
        neighbors = sorted(neighborhood(27, 8, 1))
        for total in range(1, 9):
            for same in range(total + 1):
                share = same / total
                for tolerance in (share, float(np.nextafter(share, 1.0))):
                    board = SchellingSim(schelling(tolerance=tolerance)).board
                    board.cells.fill(0)
                    board.cells.flat[27] = 1
                    board.cells.flat[neighbors[:same]] = 1
                    board.cells.flat[neighbors[same:total]] = 2
                    with self.subTest(total=total, same=same, tolerance=tolerance):
                        self.assertEqual(bool(board.satisfaction().flat[27]), share >= tolerance)
        board = SchellingSim(schelling(tolerance=1.0)).board
        board.cells.fill(0)
        board.cells.flat[27] = 1
        self.assertEqual(int(board.satisfaction().sum()), 1)

    async def test_high_tolerance_and_ordinary_stopping(self) -> None:
        sim = SchellingSim(schelling(tolerance=0.999999999))
        await sim.step()
        self.assertEqual(sim.steps, 1)
        equilibrium = SchellingSim(schelling(tolerance=0.0))
        await equilibrium.step()
        self.assertEqual(equilibrium.termination, "equilibrium")
        self.assertEqual(equilibrium.steps, 0)

    async def test_ordinary_trajectory_matches_independent_fraction_oracle(self) -> None:
        for radius in (1, 2, 3):
            settings = schelling(vision_radius=radius, max_steps=5)
            sim = SchellingSim(settings)
            while sim.termination is None:
                before = sim.snapshot()
                cells: list[int] = sim.board.cells.ravel().tolist()
                positions: list[int] = sim.board.locations.tolist()
                expected = positions.copy()
                available = {i for i, kind in enumerate(cells) if kind == 0}
                selected = settings.controlled_agent_ids
                remaining = [i for i in range(settings.agent_count) if i not in selected]
                order = [
                    int(identity)
                    for purpose, identities in (
                        ("_order_selected", selected),
                        ("_order_ordinary", remaining),
                    )
                    for identity in sim.scheduler.rng(
                        sim.steps, "schelling", 0, purpose
                    ).permutation(identities)
                ]
                for identity in order:
                    rng = sim.scheduler.rng(sim.steps, "schelling", identity, "movement")
                    origin = positions[identity]
                    kind = cells[origin]
                    current = quality(cells, neighborhood(origin, 8, 1), kind)
                    if float(current) >= settings.tolerance:
                        continue
                    visible = neighborhood(origin, 8, radius)
                    for distance in range(1, radius + 1):
                        choices = sorted(
                            p
                            for p in available & neighborhood(origin, 8, distance)
                            if quality(cells, neighborhood(p, 8, 1) & visible, kind) > current
                        )
                        if choices:
                            destination = (
                                choices[0] if len(choices) == 1 else int(rng.choice(choices))
                            )
                            expected[identity] = destination
                            available.remove(destination)
                            break
                await sim.step()
                self.assertEqual(sim.board.locations.tolist(), expected)
                self.assertEqual(len(set(expected)), settings.agent_count)
                self.assertEqual(before["agent_locations"], positions)

    async def test_ordinary_replacements_match_reference_at_any_concurrency(self) -> None:
        for concurrency in (1, 4):
            with self.subTest(concurrency=concurrency), tempfile.TemporaryDirectory() as directory:
                settings = schelling(max_steps=5)
                resources = runtime(Path(directory) / "case", concurrency=concurrency)
                ordinary = SchellingSim(settings)
                controlled = SchellingSim(settings, runtime=resources)
                # Replace only decision policies; retain the real controlled scheduling path.
                controlled.agents = tuple(
                    OrdinaryAgent(controlled, identity)
                    if identity in controlled.controlled_agent_ids
                    else agent
                    for identity, agent in enumerate(controlled.agents)
                )
                reference = await ordinary.run(resources.writer, "ordinary")
                intervention = await controlled.run(resources.writer, "controlled")
                before = records(resources.writer.directory / "ordinary.jsonl")
                after = records(resources.writer.directory / "controlled.jsonl")
                self.assertEqual(after[: len(before)], before)
                for state in after[len(before) :]:
                    self.assertEqual(state["agent_locations"], before[-1]["agent_locations"])
                result = EvaluationResult(
                    config=settings,
                    controlled_agent_ids=settings.controlled_agent_ids,
                    scored_agent_ids=tuple(
                        i
                        for i in range(settings.agent_count)
                        if i not in settings.controlled_agent_ids
                    ),
                    reference=reference,
                    controlled=intervention,
                )
                self.assertEqual(result.score, 0)

    async def test_replaying_reference_actions_through_model_tools_scores_zero(self) -> None:
        for concurrency in (1, 4):
            with self.subTest(concurrency=concurrency), tempfile.TemporaryDirectory() as directory:

                async def replay(
                    messages: list[ModelMessage],
                    info: AgentInfo,
                    *,
                    reference_path: Path = Path(directory) / "case" / "ordinary.jsonl",
                ) -> ModelResponse:
                    identity = int(
                        cast(str, info.instructions).split("Identity: ")[1].split(";")[0]
                    )
                    step = cast(int, observation(messages)["step"])
                    states = records(reference_path)
                    before = states[min(step - 1, len(states) - 1)]
                    after = states[min(step, len(states) - 1)]
                    origin = cast(list[int], before["agent_locations"])[identity]
                    destination = cast(list[int], after["agent_locations"])[identity]
                    if origin == destination:
                        return ModelResponse(parts=[ToolCallPart("stay", {})])
                    row, column = divmod(destination, 8)
                    return ModelResponse(
                        parts=[ToolCallPart("move", {"row": row, "column": column})]
                    )

                # Offline replay is a test policy, never an observation exposed to real models.
                resources = runtime(
                    Path(directory) / "case", FunctionModel(replay), concurrency=concurrency
                )
                result = await SchellingSim(schelling()).evaluate(resources)
                reference = records(resources.writer.directory / "ordinary.jsonl")
                controlled = records(resources.writer.directory / "controlled.jsonl")
                self.assertEqual(controlled[: len(reference)], reference)
                self.assertEqual(result.score, 0)

    async def test_paired_worlds_artifacts_and_independent_score_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            resources = runtime(Path(directory) / "case")
            result = await SchellingSim(schelling()).evaluate(resources)
            ordinary = records(resources.writer.directory / "ordinary.jsonl")
            controlled = records(resources.writer.directory / "controlled.jsonl")
            self.assertEqual(ordinary[0], controlled[0])
            self.assertEqual(len(controlled), 4)
            self.assertEqual(result.config, schelling())
            homophily: list[float] = []
            for states in (ordinary, controlled):
                state = states[-1]
                locations = cast(list[int], state["agent_locations"])
                kinds = cast(list[int], state["agent_types"])
                scored = {locations[i]: kinds[i] for i in result.scored_agent_ids}
                edges = {
                    tuple(sorted((p, q)))
                    for p in scored
                    for q in neighborhood(p, 8, 1)
                    if q in scored
                }
                homophily.append(sum(scored[p] == scored[q] for p, q in edges) / len(edges))
            self.assertAlmostEqual(result.score, 400 * (homophily[0] - homophily[1]))
            saved = json.loads((resources.writer.directory / "result.json").read_text())
            self.assertEqual(saved["score"], result.score)

    def test_signed_unclamped_scores(self) -> None:
        for reference, controlled, expected in (
            (0.8, 0.5, 120),
            (1, 0.4, 240),
            (0.4, 0.5, -40),
            (0.5, 0.5, 0),
        ):
            for goal, direction in (("integration", 1), ("segregation", -1)):
                result = EvaluationResult(
                    config=schelling(objective=goal),
                    controlled_agent_ids=(0, 1),
                    scored_agent_ids=(2, 3),
                    reference=Outcome(
                        steps=1, termination="horizon", homophily=(reference,), satisfaction=(1,)
                    ),
                    controlled=Outcome(
                        steps=1, termination="horizon", homophily=(controlled,), satisfaction=(1,)
                    ),
                )
                self.assertAlmostEqual(result.score, direction * expected)

    async def test_collisions_follow_priority_not_response_speed(self) -> None:
        release, faster_response = asyncio.Event(), asyncio.Event()
        calls: dict[str, int] = {}
        order: list[str] = []
        active = maximum = 0

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal active, maximum
            identity = cast(str, info.instructions).split("Identity: ")[1].split(";")[0]
            if identity not in calls:
                order.append(identity)
            calls[identity] = calls.get(identity, 0) + 1
            slot = order.index(identity)
            active += 1
            maximum = max(active, maximum)
            try:
                if slot == 0:
                    await release.wait()
                if slot == 1:
                    faster_response.set()
                if slot in (0, 1):
                    destination = vacancies[0] if calls[identity] == 1 else vacancies[1]
                    row, column = divmod(destination, 8)
                    return ModelResponse(
                        parts=[ToolCallPart("move", {"row": row, "column": column})]
                    )
                return ModelResponse(parts=[ToolCallPart("stay", {})])
            finally:
                active -= 1

        with tempfile.TemporaryDirectory() as directory:
            sim = SchellingSim(
                schelling(), runtime=runtime(Path(directory) / "case", FunctionModel(script))
            )
            initial = sim.snapshot()
            vacancies = np.flatnonzero(sim.board.cells.ravel() == 0).tolist()
            agents = sim.agents
            task = asyncio.create_task(sim.step())
            try:
                await asyncio.wait_for(faster_response.wait(), 3)
                self.assertEqual(sim.snapshot(), initial)
                self.assertFalse(task.done())
            finally:
                release.set()
                await asyncio.wait_for(task, 3)
            self.assertEqual(maximum, 2)
            self.assertEqual(calls[order[1]], 2)
            self.assertEqual(int(sim.board.locations[int(order[0])]), vacancies[0])
            self.assertEqual(int(sim.board.locations[int(order[1])]), vacancies[1])
            self.assertTrue(all(a is b for a, b in zip(agents, sim.agents, strict=True)))

    async def test_model_turn_leaves_ordinary_random_slots_and_world_unchanged(self) -> None:
        from copy import deepcopy

        checked = False

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal checked
            if not checked:
                self.assertEqual(sim.snapshot(), initial)
                self.assertEqual(sim.rng.bit_generator.state, initial_rng_state)
                self.assertEqual(sim.available, vacancies)
                self.assertEqual(
                    sim.scheduler.rng(0, "schelling", 10, "movement").random(), expected_draw
                )
                checked = True
            return ModelResponse(parts=[ToolCallPart("stay", {})])

        with tempfile.TemporaryDirectory() as directory:
            settings = schelling()
            resources = runtime(Path(directory) / "case", FunctionModel(script), concurrency=1)
            sim = SchellingSim(settings, runtime=resources)
            initial = sim.snapshot()
            initial_rng_state = deepcopy(sim.rng.bit_generator.state)
            vacancies = set(np.flatnonzero(sim.board.cells.ravel() == 0).tolist())
            ordinary = SchellingSim(settings)
            ordinary._vacancies = vacancies  # pyright: ignore[reportPrivateUsage]
            proposal = ordinary.agents[settings.controlled_agent_ids[0]]
            assert isinstance(proposal, OrdinaryAgent)
            before = ordinary.snapshot()
            self.assertEqual(proposal.propose_action(), proposal.propose_action())
            self.assertEqual(before, ordinary.snapshot())
            self.assertEqual(ordinary.available, vacancies)
            self.assertEqual(ordinary.rng.bit_generator.state, initial_rng_state)
            expected_draw = ordinary.scheduler.rng(0, "schelling", 10, "movement").random()
            await sim.step()
            self.assertTrue(checked)

    async def test_json_local_observations_and_exhaustion_retains_identity(self) -> None:
        async def invalid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            view = observation(messages)
            self.assertEqual(len(cast(list[object], view["neighborhood"])), 24)
            self.assertNotIn("cell_types", view)
            self.assertIn("integration", info.instructions or "")
            return ModelResponse(parts=[ToolCallPart("move", {"row": -1, "column": 0})])

        with tempfile.TemporaryDirectory() as directory:
            resources = runtime(Path(directory) / "case", FunctionModel(invalid))
            sim = SchellingSim(schelling(), runtime=resources)
            agent = sim.agents[0]
            assert isinstance(agent, ModelControlledAgent)
            origin = agent.position
            await sim.step()
            assert sim.sessions is not None
            history = len(sim.sessions.history("0"))
            await sim.step()
            self.assertIs(sim.agents[0], agent)
            self.assertEqual(agent.position, origin)
            self.assertGreater(len(sim.sessions.history("0")), history)
            fallbacks = records(resources.writer.directory / "turns.jsonl")
            self.assertEqual(len(fallbacks), 8)
            self.assertTrue(all(row["reason"] == "retry_exhaustion" for row in fallbacks))
