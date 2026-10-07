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
from mam_bench.simulations.civil_violence.agents import (
    CitizenAction,
    CitizenAgent,
    Defer,
    ModelCitizenAgent,
    ModelPoliceAgent,
    OrdinaryCitizenAgent,
    OrdinaryPoliceAgent,
    PoliceAgent,
)
from mam_bench.simulations.civil_violence.results import EvaluationResult, Outcome
from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim
from support import civil, observation, records, runtime, stay, turn_calls


class CivilViolenceTests(unittest.IsolatedAsyncioTestCase):
    async def test_all_roles_share_vision_radius(self) -> None:
        for role in ("citizen", "police"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                seen: dict[int, int] = {}

                async def script(
                    messages: list[ModelMessage],
                    info: AgentInfo,
                    *,
                    seen: dict[int, int] = seen,
                ) -> ModelResponse:
                    if not turn_calls(messages):
                        identity = int(
                            cast(str, info.instructions).split("Identity: ")[1].split(".")[0]
                        )
                        view = observation(messages)
                        seen[identity] = len(cast(list[object], view["neighborhood"]))
                    return await stay(messages, info)

                sim = CivilViolenceSim(
                    civil(controlled_role=role, police_density=0.25, vision_radius=2),
                    runtime=runtime(Path(directory) / "case", FunctionModel(script)),
                )
                await sim.step()
                self.assertEqual(set(seen), set(sim.controlled_agent_ids))
                self.assertEqual(set(seen.values()), {24})

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
                civil(police_density=0.25, vision_radius=2, threshold=-1000, seed=seed)
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

    async def test_released_ordinary_citizens_choose_activity_and_never_overlap(self) -> None:
        sim = CivilViolenceSim(civil(threshold=10_000))
        for citizen in list(sim.citizens.values())[:3]:
            citizen.location, citizen.active, citizen.jail_remaining = None, True, 0
        await sim.step()
        self.assertTrue(all(sim.citizens[i].location is not None for i in range(3)))
        self.assertTrue(all(not sim.citizens[i].active for i in range(3)))
        positions = [c.location for c in sim.citizens.values()]
        self.assertEqual(len(positions), len(set(positions)))

    async def test_released_model_citizen_starts_inactive_and_decides_this_round(self) -> None:
        views: list[dict[str, object]] = []

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            identity = int(cast(str, info.instructions).split("Identity: ")[1].split(".")[0])
            view = observation(messages)
            if identity == released_id:
                if not turn_calls(messages):
                    views.append(view)
                if cast(dict[str, object], view["self"])["jailed"]:
                    return ModelResponse(parts=[ToolCallPart("defer", {})])
            return ModelResponse(
                parts=[
                    ToolCallPart("act", {"active": identity == released_id, "destination": None})
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            resources = runtime(Path(directory) / "case", FunctionModel(script), concurrency=1)
            sim = CivilViolenceSim(civil(threshold=10_000), runtime=resources)
            released_id = sim.controlled_agent_ids[0]
            citizen = sim.citizens[released_id]
            citizen.location, citizen.active, citizen.jail_remaining = None, True, 0
            occupied = {c.location for c in sim.citizens.values() if c.location is not None}
            occupied.update(p.location for p in sim.police.values())
            vacancies = sorted(
                (r, c)
                for r in range(sim.settings.board_size)
                for c in range(sim.settings.board_size)
                if (r, c) not in occupied
            )
            rng = sim.scheduler.rng(0, "citizen", released_id, "release")
            expected_location = vacancies[int(rng.integers(len(vacancies)))]
            await sim.step()
            self.assertEqual(len(views), 1)
            self.assertEqual(cast(dict[str, object], views[0]["self"])["active"], False)
            self.assertEqual(cast(dict[str, object], views[0]["self"])["jailed"], False)
            self.assertEqual(citizen.location, expected_location)
            self.assertTrue(citizen.active)

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
                    assert isinstance(agent, (CitizenAgent, PoliceAgent))
                    action = agent.propose_action()
                    if isinstance(action, Defer):
                        return ModelResponse(parts=[ToolCallPart("defer", {})])
                    if isinstance(action, CitizenAction):
                        arguments: dict[str, object] = {
                            "active": action.active,
                            "destination": None,
                        }
                    else:
                        target = action.target_location
                        arguments = {
                            "target_location": list(target) if target else None,
                            "destination": None,
                        }
                    if action.destination is not None:
                        arguments["destination"] = list(action.destination)
                    return ModelResponse(parts=[ToolCallPart("act", arguments)])

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
                geometry = CivilViolenceSim(settings)
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
                    if role == "citizen":
                        return ModelResponse(
                            parts=[ToolCallPart("act", {"active": True, "destination": point})]
                        )
                    location = point or cast(
                        list[int], cast(dict[str, object], view["self"])["location"]
                    )
                    center = (location[0], location[1])
                    adjacent = set(geometry.neighborhood(center, 1))
                    targets = [
                        cast(list[int], person["location"])
                        for person in cast(list[dict[str, object]], view["neighborhood"])
                        if person["role"] == "citizen"
                        and person["active"]
                        and tuple(cast(list[int], person["location"])) in adjacent
                    ]
                    choice = slots.rng(step, phase, identity, "target")
                    target = targets[int(choice.integers(len(targets)))] if targets else None
                    return ModelResponse(
                        parts=[
                            ToolCallPart("act", {"target_location": target, "destination": point})
                        ]
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

    async def test_revolution_precedes_configured_horizon(self) -> None:
        # 100 citizens minus 16 replacements leaves 84 scored identities: 80 are needed.
        for max_steps in (1, 30, 31):
            for participating, expected in ((79, "horizon"), (80, "revolution")):
                with self.subTest(max_steps=max_steps, participating=participating):
                    sim = CivilViolenceSim(
                        civil(
                            board_size=10,
                            citizen_density=1.0,
                            threshold=0.0,
                            max_steps=max_steps,
                        )
                    )
                    active = set(sim.scored_agent_ids[:participating])
                    for identity, citizen in sim.citizens.items():
                        citizen.private_preference = -10_000 if identity in active else 10_000
                    sim.steps = max_steps - 1
                    await sim.step()
                    self.assertEqual(sim.termination, expected)
                    self.assertEqual(sim.steps, max_steps)
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
                        return ModelResponse(
                            parts=[ToolCallPart("act", {"active": False, "destination": None})]
                        )
                    return ModelResponse(parts=[ToolCallPart("defer", {})])
            return ModelResponse(
                parts=[ToolCallPart("act", {"active": False, "destination": None})]
            )

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
            self.assertFalse(citizen.active)
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
            assert isinstance(agent, (CitizenAgent, PoliceAgent))
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

    async def test_controlled_citizens_and_police_can_move_beyond_their_view(self) -> None:
        async def check(role: str) -> None:
            with tempfile.TemporaryDirectory() as directory:
                moved: dict[int, tuple[int, int]] = {}

                async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                    identity = int(
                        cast(str, info.instructions).split("Identity: ")[1].split(".")[0]
                    )
                    destination: tuple[int, int] | None = None
                    if not moved:
                        view = observation(messages)
                        location = cast(
                            list[int], cast(dict[str, object], view["self"])["location"]
                        )
                        adjacent = set(sim.neighborhood((location[0], location[1]), 1))
                        state = sim.snapshot()
                        occupied = {
                            (point[0], point[1])
                            for group in ("citizens", "police")
                            for person in cast(list[dict[str, object]], state[group])
                            if (point := cast(list[int] | None, person["location"])) is not None
                        }
                        destination = next(
                            (row, column)
                            for row in range(sim.settings.board_size)
                            for column in range(sim.settings.board_size)
                            if (row, column) not in occupied | adjacent
                        )
                        self.assertNotIn(
                            list(destination), cast(list[list[int]], view["legal_destinations"])
                        )
                        moved[identity] = destination
                    arguments: dict[str, object] = (
                        {"active": False, "destination": None}
                        if role == "citizen"
                        else {"target_location": None, "destination": None}
                    )
                    if destination is not None:
                        arguments["destination"] = list(destination)
                    return ModelResponse(parts=[ToolCallPart("act", arguments)])

                resources = runtime(Path(directory) / "case", FunctionModel(script), concurrency=1)
                settings = civil(
                    controlled_role=role,
                    citizen_density=0.5,
                    police_density=0.25 if role == "police" else 0.0,
                )
                sim = CivilViolenceSim(settings, runtime=resources)
                await sim.step()
                self.assertEqual(len(moved), 1)
                identity, destination = next(iter(moved.items()))
                person = sim.citizens[identity] if role == "citizen" else sim.police[identity]
                self.assertEqual(person.location, destination)

        for role in ("citizen", "police"):
            with self.subTest(role=role):
                await check(role)

    async def test_police_can_move_then_arrest_by_target_square(self) -> None:
        planned: dict[int, tuple[list[int], list[int]]] = {}

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            self.assertIn("act", {tool.name for tool in info.output_tools})
            identity = int(cast(str, info.instructions).split("Identity: ")[1].split(".")[0])
            view = observation(messages)
            if not planned:
                origin = cast(list[int], cast(dict[str, object], view["self"])["location"])
                candidates = [
                    (destination, cast(list[int], person["location"]))
                    for destination in cast(list[list[int]], view["legal_destinations"])
                    for person in cast(list[dict[str, object]], sim.snapshot()["citizens"])
                    if person["active"]
                    and person["location"] is not None
                    and tuple(cast(list[int], person["location"]))
                    in sim.neighborhood((destination[0], destination[1]), 1)
                    and tuple(cast(list[int], person["location"]))
                    not in sim.neighborhood((origin[0], origin[1]), 1)
                ]
                if candidates:
                    planned[identity] = candidates[0]
            if identity in planned:
                destination, target = planned[identity]
                return ModelResponse(
                    parts=[
                        ToolCallPart("act", {"destination": destination, "target_location": target})
                    ]
                )
            return ModelResponse(
                parts=[ToolCallPart("act", {"destination": None, "target_location": None})]
            )

        with tempfile.TemporaryDirectory() as directory:
            sim = CivilViolenceSim(
                civil(
                    controlled_role="police",
                    board_size=12,
                    citizen_density=0.25,
                    police_density=16 / 144,
                ),
                runtime=runtime(Path(directory) / "case", FunctionModel(script), concurrency=1),
            )
            await sim.step()
            self.assertTrue(planned)
            identity, (destination, target) = next(iter(planned.items()))
            self.assertEqual(sim.police[identity].location, tuple(destination))
            self.assertNotIn(tuple(target), {c.location for c in sim.citizens.values()})

    async def test_ordinary_police_move_before_choosing_arrest_target(self) -> None:
        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            view = observation(messages)
            if cast(dict[str, object], view["self"])["jailed"]:
                return ModelResponse(parts=[ToolCallPart("defer", {})])
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "act",
                        {
                            "active": True,
                            "destination": None,
                        },
                    )
                ]
            )

        with tempfile.TemporaryDirectory() as directory:
            sim = CivilViolenceSim(
                civil(board_size=12, citizen_density=0.25, police_density=16 / 144),
                runtime=runtime(Path(directory) / "case", FunctionModel(script)),
            )
            target_id = sim.controlled_agent_ids[0]
            for citizen in sim.citizens.values():
                citizen.location, citizen.jail_remaining = None, 99
            sim.citizens[target_id].location = (0, 2)
            priority = sorted(sim.police)
            sim.scheduler.rng(0, "police", 0, "_order_ordinary").shuffle(priority)
            first = priority[0]
            origin = (0, 0)
            destination = (0, 1)
            neighbors = [point for point in sim.neighborhood(origin, 1) if point != destination]
            sim.police[first].location = origin
            for officer_id, point in zip(priority[1:8], neighbors, strict=True):
                sim.police[officer_id].location = point
            occupied = {origin, (0, 2), *neighbors, destination}
            far = [
                (row, column)
                for row in range(12)
                for column in range(12)
                if (row, column) not in occupied
                and (row, column) not in sim.neighborhood((0, 2), 2)
            ]
            for officer_id, point in zip(priority[8:], far[:8], strict=True):
                sim.police[officer_id].location = point
            self.assertNotIn((0, 2), sim.neighborhood(origin, 1))
            self.assertIn((0, 2), sim.neighborhood(destination, 1))
            await sim.step()
            self.assertEqual(sim.police[first].location, destination)
            self.assertIsNone(sim.citizens[target_id].location)

    async def test_invalid_whole_police_action_retries_without_reserving_either_resource(
        self,
    ) -> None:
        planned: dict[int, tuple[list[int], list[int]]] = {}
        retried: set[int] = set()

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            identity = int(cast(str, info.instructions).split("Identity: ")[1].split(".")[0])
            view = observation(messages)
            if identity in planned:
                target, destination = planned[identity]
                self.assertIn("RetryPromptPart", str(messages))
                self.assertIn(tuple(destination), sim.legal_destinations(identity))
                self.assertEqual(
                    [c.location for c in sim.citizens.values() if c.location == tuple(target)],
                    [tuple(target)],
                )
                retried.add(identity)
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "act",
                            {
                                "target_location": target,
                                "destination": destination,
                            },
                        )
                    ]
                )
            targets = cast(list[list[int]], view["eligible_target_locations_if_staying"])
            destinations = cast(list[list[int]], view["legal_destinations"])
            pairs = [
                (target, destination)
                for destination in destinations
                for target in targets
                if tuple(target) in sim.neighborhood((destination[0], destination[1]), 1)
            ]
            if not planned and pairs:
                target, destination = pairs[0]
                planned[identity] = target, destination
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "act",
                            {
                                "target_location": target,
                                "destination": [99, 99],
                            },
                        )
                    ]
                )
            return ModelResponse(
                parts=[ToolCallPart("act", {"destination": None, "target_location": None})]
            )

        with tempfile.TemporaryDirectory() as directory:
            resources = runtime(Path(directory) / "case", FunctionModel(script), concurrency=1)
            sim = CivilViolenceSim(
                civil(controlled_role="police", citizen_density=0.5, police_density=0.25, seed=2),
                runtime=resources,
            )
            await sim.step()
            self.assertEqual(len(planned), 1)
            self.assertEqual(retried, set(planned))
            identity, (target, destination) = next(iter(planned.items()))
            self.assertEqual(sim.police[identity].location, tuple(destination))
            self.assertNotIn(tuple(target), {c.location for c in sim.citizens.values()})
            self.assertFalse((resources.writer.directory / "turns.jsonl").exists())

    async def test_invalid_target_square_does_not_claim_valid_destination(self) -> None:
        planned: dict[int, list[int]] = {}
        retried: set[int] = set()

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            identity = int(cast(str, info.instructions).split("Identity: ")[1].split(".")[0])
            if identity in planned:
                destination = planned[identity]
                self.assertIn("RetryPromptPart", str(messages))
                self.assertIn((destination[0], destination[1]), sim.legal_destinations(identity))
                retried.add(identity)
                return ModelResponse(
                    parts=[
                        ToolCallPart("act", {"destination": destination, "target_location": None})
                    ]
                )
            destinations = cast(list[list[int]], observation(messages)["legal_destinations"])
            if not planned and destinations:
                destination = destinations[0]
                planned[identity] = destination
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            "act", {"destination": destination, "target_location": [99, 99]}
                        )
                    ]
                )
            return ModelResponse(
                parts=[ToolCallPart("act", {"destination": None, "target_location": None})]
            )

        with tempfile.TemporaryDirectory() as directory:
            resources = runtime(Path(directory) / "case", FunctionModel(script), concurrency=1)
            sim = CivilViolenceSim(
                civil(controlled_role="police", citizen_density=0.5, police_density=0.25, seed=2),
                runtime=resources,
            )
            await sim.step()
            self.assertEqual(len(planned), 1)
            self.assertEqual(retried, set(planned))
            identity, destination = next(iter(planned.items()))
            self.assertEqual(sim.police[identity].location, tuple(destination))
            self.assertFalse((resources.writer.directory / "turns.jsonl").exists())

    async def test_each_role_has_one_terminal_act_tool_and_role_observation(self) -> None:
        self.assertFalse(hasattr(OrdinaryCitizenAgent, "INSTRUCTIONS"))
        self.assertFalse(hasattr(OrdinaryPoliceAgent, "INSTRUCTIONS"))
        for role in ("citizen", "police"):
            with self.subTest(role=role), tempfile.TemporaryDirectory() as directory:
                targets: dict[int, list[int]] = {}
                observed: set[int] = set()

                async def script(
                    messages: list[ModelMessage],
                    info: AgentInfo,
                    *,
                    role: str = role,
                    targets: dict[int, list[int]] = targets,
                    observed: set[int] = observed,
                ) -> ModelResponse:
                    identity = int(
                        cast(str, info.instructions).split("Identity: ")[1].split(".")[0]
                    )
                    observed.add(identity)
                    self.assertEqual(
                        {tool.name for tool in info.output_tools},
                        {"act", "defer"} if role == "citizen" else {"act"},
                    )
                    self.assertFalse(
                        {"submit", "choose_activity", "choose_arrest", "move", "stay"}
                        & {tool.name for tool in info.function_tools}
                    )
                    instructions = cast(str, info.instructions)
                    self.assertIn("Ordinary citizens use local counts", instructions)
                    self.assertIn("They return inactive, then choose activity", instructions)
                    self.assertIn(
                        "Citizen decisions" if role == "citizen" else "Police decisions",
                        instructions,
                    )
                    view = observation(messages)
                    self.assertNotIn("terminal_action", view)
                    self.assertNotIn("eligible_targets", view)
                    if role == "citizen":
                        self.assertNotIn("eligible_target_locations_if_staying", view)
                        return ModelResponse(
                            parts=[ToolCallPart("act", {"active": True, "destination": None})]
                        )
                    if "RetryPromptPart" in str(messages):
                        targets.pop(identity, None)
                        return ModelResponse(
                            parts=[
                                ToolCallPart("act", {"destination": None, "target_location": None})
                            ]
                        )
                    eligible = cast(list[list[int]], view["eligible_target_locations_if_staying"])
                    if eligible:
                        targets[identity] = eligible[0]
                    return ModelResponse(
                        parts=[
                            ToolCallPart(
                                "act",
                                {
                                    "target_location": eligible[0] if eligible else None,
                                    "destination": None,
                                },
                            )
                        ]
                    )

                resources = runtime(Path(directory) / "case", FunctionModel(script), concurrency=1)
                sim = CivilViolenceSim(
                    civil(
                        controlled_role=role,
                        police_density=0.25 if role == "police" else 0,
                        threshold=-10_000,
                    ),
                    runtime=resources,
                )
                model_type = ModelCitizenAgent if role == "citizen" else ModelPoliceAgent
                self.assertTrue(
                    all(isinstance(sim.agents[i], model_type) for i in sim.controlled_agent_ids)
                )
                origins = {
                    identity: sim.agents[identity].location for identity in sim.controlled_agent_ids
                }
                await sim.step()
                self.assertEqual(observed, set(sim.controlled_agent_ids))
                self.assertEqual(
                    origins, {identity: sim.agents[identity].location for identity in origins}
                )
                if role == "citizen":
                    self.assertTrue(all(sim.citizens[i].active for i in sim.controlled_agent_ids))
                else:
                    self.assertTrue(targets)
                    self.assertTrue(
                        all(
                            tuple(target) not in {c.location for c in sim.citizens.values()}
                            for target in targets.values()
                        )
                    )
                self.assertFalse((resources.writer.directory / "turns.jsonl").exists())

    async def test_act_missing_required_activity_retries_then_falls_back(self) -> None:
        async def missing(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart("act", {"destination": None})])

        with tempfile.TemporaryDirectory() as directory:
            resources = runtime(Path(directory) / "case", FunctionModel(missing))
            sim = CivilViolenceSim(civil(threshold=10_000), runtime=resources)
            origins = {i: sim.citizens[i].location for i in sim.controlled_agent_ids}
            await sim.step()
            self.assertEqual(origins, {i: sim.citizens[i].location for i in origins})
            self.assertTrue(all(not sim.citizens[i].active for i in origins))
            fallbacks = records(resources.writer.directory / "turns.jsonl")
            self.assertEqual(len(fallbacks), 16)
            self.assertTrue(all(row["reason"] == "retry_exhaustion" for row in fallbacks))

    async def test_invalid_police_action_exhausts_then_falls_back(self) -> None:
        async def invalid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "act",
                        {
                            "target_location": [-1, -1],
                            "destination": [99, 99],
                        },
                    )
                ]
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
