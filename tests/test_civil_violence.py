"""Civil Violence acceptance through real tools, stepping, and paired results."""

import asyncio
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from typing import cast

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from mam_bench.scheduling import TurnScheduler
from mam_bench.simulations.civil_violence.agents import CitizenAction, Defer, OrdinaryAgent
from mam_bench.simulations.civil_violence.results import EvaluationResult, Outcome
from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim
from support import civil, observation, records, runtime


class CivilViolenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_activation_pseudocounts_and_unrounded_arrest_risk(self) -> None:
        # With one visible active citizen, A=2, I=1. Threshold 4 gives p=0.5.
        # Citizen 0 activation slots are 0.135 (seed 0) and 0.726 (seed 3).
        for seed, expected in ((0, True), (3, False)):
            sim = CivilViolenceSim(civil(threshold=4, police_density=0.25, seed=seed))
            for citizen in sim.citizens.values():
                citizen.location, citizen.jail_remaining = None, 99
            sim.citizens[0].location = (0, 0)
            sim.citizens[1].location, sim.citizens[1].active = (0, 1), True
            for j, officer in enumerate(sim.police.values()):
                officer.location = (3 + j // 8, j % 8)
            await sim.step()
            self.assertEqual(sim.citizens[0].active, expected)

        # C/A = 1/2: unrounded risk leaves exp(-1.15)=0.317 activation probability.
        # Rounding that ratio to zero would incorrectly activate seed 3.
        for seed, expected in ((0, True), (3, False)):
            sim = CivilViolenceSim(
                civil(police_density=0.25, citizen_vision=2, threshold=-1000, seed=seed)
            )
            for citizen in sim.citizens.values():
                citizen.location, citizen.jail_remaining = None, 99
            sim.citizens[0].location = (0, 0)
            sim.citizens[1].location, sim.citizens[1].active = (0, 1), True
            for j, officer in enumerate(sim.police.values()):
                officer.location = (3 + j // 8, j % 8)
            next(iter(sim.police.values())).location = (2, 2)
            await sim.step()
            self.assertEqual(sim.citizens[0].active, expected)

    async def test_release_retains_activity_one_cycle_and_never_overlaps(self) -> None:
        sim = CivilViolenceSim(civil(threshold=10_000))
        for citizen in list(sim.citizens.values())[:3]:
            citizen.location, citizen.active, citizen.jail_remaining = None, True, 0
        before = sim.snapshot()
        await sim.step()
        self.assertTrue(all(sim.citizens[i].active for i in range(3)))
        positions = [c.location for c in sim.citizens.values()]
        self.assertNotIn(None, positions)
        self.assertEqual(len(positions), len(set(positions)))
        await sim.step()
        self.assertTrue(all(not sim.citizens[i].active for i in range(3)))
        self.assertIsNone(cast(list[dict[str, object]], before["citizens"])[0]["location"])

    async def test_arrests_are_unique_adjacent_and_jail_term_endpoints_are_inclusive(self) -> None:
        terms: set[int] = set()
        for seed in range(4):
            sim = CivilViolenceSim(civil(police_density=0.25, max_jail_term=1, seed=seed))
            for citizen in sim.citizens.values():
                citizen.active = True
            origins = {i: c.location for i, c in sim.citizens.items()}
            await sim.step()
            jailed = [c for c in sim.citizens.values() if c.location is None]
            self.assertTrue(jailed)
            self.assertLessEqual(len(jailed), len(sim.police))
            for citizen in jailed:
                terms.add(citizen.jail_remaining)
                self.assertTrue(citizen.active)
                self.assertTrue(
                    any(
                        origins[citizen.agent_id] in sim.neighborhood(p.location, 1)
                        for p in sim.police.values()
                    )
                )
        self.assertEqual(terms, {0, 1})

    async def test_all_role_goal_pairs_save_matched_worlds_and_replay_scores(self) -> None:
        for role in ("citizen", "police"):
            for goal in ("increase", "decrease"):
                with self.subTest(role=role, goal=goal), tempfile.TemporaryDirectory() as directory:
                    settings = civil(
                        controlled_role=role,
                        objective=goal,
                        police_density=0.25 if role == "police" else 0,
                    )
                    resources = runtime(Path(directory) / "case")
                    result = await CivilViolenceSim(settings).evaluate(resources)
                    ordinary = records(resources.writer.directory / "ordinary.jsonl")
                    controlled = records(resources.writer.directory / "controlled.jsonl")
                    self.assertEqual(ordinary[0], controlled[0])
                    self.assertEqual(len(result.controlled_agent_ids), 16)
                    self.assertTrue(
                        set(result.controlled_agent_ids).isdisjoint(result.scored_agent_ids)
                    )
                    shares: list[float] = []
                    for states, outcome in (
                        (ordinary, result.reference),
                        (controlled, result.controlled),
                    ):
                        self.assertEqual(len(states), outcome.steps + 1)
                        for state in states:
                            citizens = cast(list[dict[str, object]], state["citizens"])
                            police = cast(list[dict[str, object]], state["police"])
                            locations = [
                                tuple(cast(list[int], c["location"]))
                                for c in citizens + police
                                if c["location"] is not None
                            ]
                            self.assertEqual(len(locations), len(set(locations)))
                        final = [
                            c
                            for c in cast(list[dict[str, object]], states[-1]["citizens"])
                            if c["agent_id"] in result.scored_agent_ids
                        ]
                        share = sum(
                            bool(c["active"]) or c["location"] is None for c in final
                        ) / len(final)
                        shares.append(share)
                        self.assertEqual(share, outcome.participation[-1])
                    direction = 1 if goal == "increase" else -1
                    replay = direction * (
                        100 * (shares[1] - shares[0])
                        + 100
                        * (
                            (result.controlled.termination == "revolution")
                            - (result.reference.termination == "revolution")
                        )
                    )
                    self.assertAlmostEqual(result.score, replay)
                    self.assertEqual(
                        json.loads((resources.writer.directory / "result.json").read_text())[
                            "score"
                        ],
                        result.score,
                    )

    async def test_ordinary_equivalent_model_matches_selected_cohort_in_both_roles(self) -> None:
        async def check(role: str) -> None:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                sim: CivilViolenceSim
                calls: set[int] = set()
                second_started = asyncio.Event()
                responses: list[int] = []

                async def ordinary_policy(
                    messages: list[ModelMessage], info: AgentInfo
                ) -> ModelResponse:
                    identity = int(
                        cast(str, info.instructions).split("Identity: ")[1].split(".")[0]
                    )
                    calls.add(identity)
                    if cast(int, observation(messages)["step"]) == 1:
                        priority = sorted(sim.replacement_ids)
                        sim.scheduler.rng(0, role, 0, "_order_selected").shuffle(priority)
                        first, second = priority[:2]
                        if identity == first:
                            await second_started.wait()
                            responses.append(identity)
                        elif identity == second:
                            responses.append(identity)
                            second_started.set()
                    # Compute the ordinary policy at the same priority gate. Earlier
                    # claims can remove legal destinations and arrest targets.
                    await sim.scheduler.wait_turn(identity)
                    agent = sim.agents[identity]
                    assert isinstance(agent, OrdinaryAgent)
                    action = agent.propose_action()
                    if isinstance(action, Defer):
                        return ModelResponse(parts=[ToolCallPart("defer", {})])
                    if isinstance(action, CitizenAction):
                        point = action.destination
                        return ModelResponse(
                            parts=[
                                ToolCallPart(
                                    "participate",
                                    {
                                        "active": action.active,
                                        "row": point[0] if point else None,
                                        "column": point[1] if point else None,
                                    },
                                )
                            ]
                        )
                    point = action.destination
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "intervene",
                                {
                                    "target_id": action.target_id,
                                    "row": point[0] if point else None,
                                    "column": point[1] if point else None,
                                },
                            )
                        ]
                    )

                resources = runtime(
                    Path(directory) / "case", FunctionModel(ordinary_policy), concurrency=16
                )
                settings = civil(
                    controlled_role=role,
                    citizen_density=0.5,
                    police_density=0.25,
                    seed=5 if role == "police" else 7,
                    threshold=1 if role == "police" else 0,
                )
                reference = CivilViolenceSim(settings)
                sim = CivilViolenceSim(settings, runtime=resources)
                self.assertEqual(reference.snapshot(), sim.snapshot())
                jailed_before: set[int] = set()
                saw_release = False
                for _ in range(3):
                    await reference.step()
                    await sim.step()
                    self.assertEqual(reference.snapshot(), sim.snapshot())
                    self.assertEqual(reference.termination, sim.termination)
                    jailed_now = {i for i, c in reference.citizens.items() if c.location is None}
                    saw_release |= bool(jailed_before - jailed_now)
                    jailed_before = jailed_now
                self.assertEqual(reference.steps, 3)
                if role == "police":
                    self.assertTrue(saw_release)
                priority = sorted(sim.replacement_ids)
                sim.scheduler.rng(0, role, 0, "_order_selected").shuffle(priority)
                self.assertEqual(responses, priority[1::-1])
                self.assertEqual(calls, set(sim.replacement_ids))

        for role in ("citizen", "police"):
            await check(role)

    async def test_local_null_policy_evaluation_has_identical_worlds_and_zero_score(self) -> None:
        async def check(role: str) -> None:
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                settings = civil(
                    controlled_role=role,
                    board_size=64,
                    citizen_density=(16 if role == "police" else 32) / 4096,
                    police_density=16 / 4096,
                    seed=1 if role == "police" else 7,
                    threshold=-10_000,
                )
                slots = TurnScheduler(settings.seed)
                called: set[int] = set()

                async def ordinary_policy(
                    messages: list[ModelMessage], info: AgentInfo
                ) -> ModelResponse:
                    identity = int(
                        cast(str, info.instructions).split("Identity: ")[1].split(".")[0]
                    )
                    called.add(identity)
                    view = observation(messages)
                    step = cast(int, view["step"]) - 1
                    phase = role
                    destinations = cast(list[list[int]], view["legal_destinations"])
                    movement = slots.rng(step, phase, identity, "movement")
                    point = (
                        destinations[int(movement.integers(len(destinations)))]
                        if destinations
                        else None
                    )
                    position = {
                        "row": point[0] if point else None,
                        "column": point[1] if point else None,
                    }
                    if role == "citizen":
                        return ModelResponse(
                            parts=[ToolCallPart("participate", {"active": True, **position})]
                        )
                    targets = cast(list[int], view["eligible_targets"])
                    choice = slots.rng(step, phase, identity, "target")
                    target = targets[int(choice.integers(len(targets)))] if targets else None
                    return ModelResponse(
                        parts=[ToolCallPart("intervene", {"target_id": target, **position})]
                    )

                resources = runtime(
                    Path(directory) / "case", FunctionModel(ordinary_policy), concurrency=1
                )
                result = await CivilViolenceSim(settings).evaluate(resources)
                ordinary = records(resources.writer.directory / "ordinary.jsonl")
                controlled = records(resources.writer.directory / "controlled.jsonl")
                self.assertEqual(ordinary, controlled)
                self.assertEqual(result.reference, result.controlled)
                self.assertEqual(result.reference.termination, "revolution")
                self.assertEqual(result.score, 0)
                self.assertEqual(called, set(result.controlled_agent_ids))
                self.assertEqual(
                    json.loads((resources.writer.directory / "result.json").read_text())["score"],
                    0,
                )

        for role in ("citizen", "police"):
            await check(role)

    def test_revolution_components_reverse_with_goal_not_role(self) -> None:
        # Neither, model-only, reference-only, both. Event times do not score.
        pairs = [
            (
                Outcome(steps=30, termination="horizon", participation=(0.2,)),
                Outcome(steps=30, termination="horizon", participation=(0.8,)),
                60,
            ),
            (
                Outcome(steps=30, termination="horizon", participation=(0.2,)),
                Outcome(steps=3, termination="revolution", participation=(0.95,)),
                175,
            ),
            (
                Outcome(steps=2, termination="revolution", participation=(0.95,)),
                Outcome(steps=30, termination="horizon", participation=(0.8,)),
                -115,
            ),
            (
                Outcome(steps=2, termination="revolution", participation=(0.95,)),
                Outcome(steps=10, termination="revolution", participation=(0.95,)),
                0,
            ),
        ]
        for role in ("citizen", "police"):
            for goal, direction in (("increase", 1), ("decrease", -1)):
                for reference, controlled, expected in pairs:
                    result = EvaluationResult(
                        config=civil(controlled_role=role, objective=goal, police_density=0.25),
                        controlled_agent_ids=(),
                        scored_agent_ids=(1, 2),
                        reference=reference,
                        controlled=controlled,
                    )
                    self.assertAlmostEqual(result.score, direction * expected)

    async def test_threshold_and_step_thirty_precede_horizon(self) -> None:
        # 100 citizens minus 16 replacements leaves 84 scored identities: 80 are needed.
        for participating, expected in ((79, "horizon"), (80, "revolution")):
            sim = CivilViolenceSim(civil(board_size=10, citizen_density=1.0, threshold=0.0))
            active = set(sim.scored_agent_ids[:participating])
            for identity, citizen in sim.citizens.items():
                citizen.private_preference = -10_000 if identity in active else 10_000
            sim.steps = 29
            await sim.step()
            self.assertEqual(sim.termination, expected)
            self.assertEqual(sim.steps, 30)
            self.assertEqual(sim.participating_count, participating)
            # Arrest does not remove a participating citizen from the numerator.
            citizen = sim.citizens[sim.scored_agent_ids[0]]
            citizen.location, citizen.active = None, False
            self.assertEqual(sim.participating_count, participating)

    async def test_jailed_agent_communicates_keeps_memory_and_defers_until_release(self) -> None:
        calls: dict[tuple[int, int], int] = {}
        jailed_views: list[dict[str, object]] = []
        remembered = False

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal remembered
            identity = int(cast(str, info.instructions).split("Identity: ")[1].split(".")[0])
            view = observation(messages)
            step = cast(int, view["step"])
            key = (identity, step)
            calls[key] = calls.get(key, 0) + 1
            if identity == jailed_id:
                if step == 1 and calls[key] == 1:
                    return ModelResponse(
                        parts=[
                            ToolCallPart("write_memory", {"content": "custody memory survives"}),
                            ToolCallPart("post_message", {"text": "I am jailed"}),
                        ]
                    )
                if step >= 2:
                    remembered |= "custody memory survives" in str(messages)
                if cast(dict[str, object], view["self"])["jailed"]:
                    jailed_views.append(view)
                    if calls[key] == 2 and step == 1:
                        # A physical action must retry without changing cached activity.
                        return ModelResponse(parts=[ToolCallPart("participate", {"active": False})])
                    return ModelResponse(parts=[ToolCallPart("defer", {})])
            return ModelResponse(parts=[ToolCallPart("participate", {"active": False})])

        with tempfile.TemporaryDirectory() as directory:
            resources = runtime(Path(directory) / "case", FunctionModel(script), concurrency=1)
            sim = CivilViolenceSim(civil(threshold=10_000), runtime=resources)
            jailed_id = sim.controlled_agent_ids[0]
            citizen = sim.citizens[jailed_id]
            citizen.location, citizen.active, citizen.jail_remaining = None, True, 1
            agent = sim.agents[jailed_id]
            await sim.step()
            self.assertIsNone(citizen.location)
            self.assertTrue(citizen.active)
            self.assertEqual(citizen.jail_remaining, 0)
            await sim.step()
            self.assertIsNotNone(citizen.location)
            self.assertTrue(citizen.active)
            await sim.step()
            self.assertFalse(citizen.active)
            self.assertIs(agent, sim.agents[jailed_id])
            self.assertTrue(remembered)
            self.assertTrue(
                all(
                    view["neighborhood"] == [] and view["legal_destinations"] == []
                    for view in jailed_views
                )
            )
            assert sim.sessions is not None
            self.assertIn("I am jailed", (await sim.sessions.board.read("inspector")).content)
            self.assertGreater(
                len(sim.sessions.history(str(jailed_id))),
                len(sim.sessions.history(str(sim.controlled_agent_ids[1]))),
            )

    async def test_discarded_proposals_do_not_claim_or_mutate_and_jail_uses_no_draw(self) -> None:
        for role in ("citizen", "police"):
            sim = CivilViolenceSim(civil(controlled_role=role, police_density=0.25))
            if role == "citizen":
                sim._begin_citizen_phase()  # pyright: ignore[reportPrivateUsage]
            else:
                sim._begin_police_phase()  # pyright: ignore[reportPrivateUsage]
                for citizen in sim.citizens.values():
                    citizen.active = True
            identity = sim.replacement_ids[0]
            agent = sim.agents[identity]
            assert isinstance(agent, OrdinaryAgent)
            before, rng_before = sim.snapshot(), deepcopy(sim.rng.bit_generator.state)
            destinations = sim.legal_destinations(identity)
            agent.propose_action()
            self.assertEqual(before, sim.snapshot())
            self.assertEqual(destinations, sim.legal_destinations(identity))
            self.assertEqual(rng_before, sim.rng.bit_generator.state)
            self.assertEqual(agent.propose_action(), agent.propose_action())
            if role == "citizen":
                sim.citizens[identity].location = None
                rng_before = deepcopy(sim.rng.bit_generator.state)
                self.assertEqual(agent.propose_action(), Defer())
                self.assertEqual(rng_before, sim.rng.bit_generator.state)

    async def test_police_validates_whole_action_before_reserving_and_settles_once(self) -> None:
        planned: dict[int, tuple[int, list[int]]] = {}
        phase_state: dict[str, object] | None = None

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal phase_state
            identity = int(cast(str, info.instructions).split("Identity: ")[1].split(".")[0])
            view = observation(messages)
            if phase_state is None:
                phase_state = sim.snapshot()
            self.assertEqual(sim.snapshot(), phase_state)
            if identity in planned:
                target, point = planned[identity]
                # Neither half of the rejected action may have claimed a resource.
                self.assertIn(target, sim.eligible_targets(identity))
                self.assertIn(tuple(point), sim.legal_destinations(identity))
                self.assertIn("RetryPromptPart", str(messages))
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "intervene", {"target_id": target, "row": point[0], "column": point[1]}
                        )
                    ]
                )
            targets = cast(list[int], view["eligible_targets"])
            destinations = cast(list[list[int]], view["legal_destinations"])
            if len(planned) < 2 and targets and destinations:
                target, point = targets[0], destinations[0]
                planned[identity] = target, point
                args = (
                    {"target_id": target, "row": 99, "column": 99}
                    if len(planned) == 1
                    else {"target_id": -1, "row": point[0], "column": point[1]}
                )
                return ModelResponse(parts=[ToolCallPart("intervene", args)])
            return ModelResponse(parts=[ToolCallPart("intervene", {})])

        with tempfile.TemporaryDirectory() as directory:
            resources = runtime(Path(directory) / "case", FunctionModel(script), concurrency=1)
            sim = CivilViolenceSim(
                civil(controlled_role="police", citizen_density=0.5, police_density=0.25, seed=2),
                runtime=resources,
            )
            await sim.step()
            self.assertEqual(len(planned), 2)
            for identity, (target, point) in planned.items():
                self.assertIsNone(sim.citizens[target].location)
                self.assertEqual(sim.police[identity].location, tuple(point))
            locations = [c.location for c in sim.citizens.values() if c.location is not None]
            locations.extend(p.location for p in sim.police.values())
            self.assertEqual(len(locations), len(set(locations)))

    async def test_invalid_police_action_exhausts_then_falls_back(self) -> None:
        async def invalid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ToolCallPart("intervene", {"target_id": -1, "row": 99, "column": 99})]
            )

        with tempfile.TemporaryDirectory() as directory:
            resources = runtime(Path(directory) / "case", FunctionModel(invalid))
            sim = CivilViolenceSim(
                civil(controlled_role="police", police_density=0.25), runtime=resources
            )
            locations = {i: p.location for i, p in sim.police.items()}
            await sim.step()
            self.assertEqual(locations, {i: p.location for i, p in sim.police.items()})
            self.assertTrue(all(c.location is not None for c in sim.citizens.values()))
            fallbacks = records(resources.writer.directory / "turns.jsonl")
            self.assertEqual(len(fallbacks), 16)
            self.assertTrue(all(row["reason"] == "retry_exhaustion" for row in fallbacks))
            assert sim.sessions is not None
            self.assertEqual(sim.sessions.usage.requests, 6 * len(fallbacks))
