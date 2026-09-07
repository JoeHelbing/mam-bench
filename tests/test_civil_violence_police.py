"""Police tactical behavior through real sessions and complete simulation cycles."""

import json
import tempfile
import unittest
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo

from mam_bench.simulations.civil_violence import (
    Citizen,
    CivilViolenceSettings,
    CivilViolenceSim,
    Police,
)
from schelling_support import runtime_for


class PoliceObservation(TypedDict):
    agent_id: int
    eligible_targets: list[int]
    legal_destinations: list[list[int]]
    neighborhood: list[dict[str, object]]


def observation(messages: list[ModelMessage]) -> PoliceObservation:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in reversed(message.parts):
                if isinstance(part, UserPromptPart):
                    return cast(PoliceObservation, json.loads(str(part.content)))
    raise AssertionError("missing local observation")


class PoliceParticipationTests(unittest.IsolatedAsyncioTestCase):
    async def test_combined_conflicts_retry_without_leaking_target_or_move_claims(self) -> None:
        accepted_target: int | None = None
        second_target: int | None = None
        first_destination: list[int] = []
        second_destination: list[int] = []
        calls: dict[int, int] = {}
        retry_messages: list[str] = []

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal accepted_target, second_target, first_destination, second_destination
            view = observation(messages)
            agent_id = view["agent_id"]
            calls[agent_id] = calls.get(agent_id, 0) + 1
            for message in messages:
                if isinstance(message, ModelRequest):
                    retry_messages.extend(
                        str(p.content) for p in message.parts if isinstance(p, RetryPromptPart)
                    )
            if accepted_target is None:
                self.assertGreaterEqual(len(view["eligible_targets"]), 2)
                self.assertGreaterEqual(len(view["legal_destinations"]), 2)
                accepted_target, second_target = view["eligible_targets"][:2]
                first_destination, second_destination = view["legal_destinations"][:2]
                target, destination = accepted_target, first_destination
            elif calls[agent_id] == 1:
                # A new target plus already-claimed movement must claim neither.
                target, destination = second_target, first_destination
            elif calls[agent_id] == 2:
                # A new destination plus already-claimed arrest must claim neither.
                target, destination = accepted_target, second_destination
            else:
                target, destination = second_target, second_destination
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "intervene",
                        {
                            "target_id": target,
                            "row": destination[0],
                            "column": destination[1],
                        },
                    )
                ]
            )

        settings = CivilViolenceSettings(
            board_size=3, controlled_agent_count=2, max_transitions=1, seed_id=2
        )
        citizens = tuple(Citizen(i, divmod(i, 3), -1000, True) for i in range(5))
        police = (Police(5, (2, 1)), Police(6, (2, 2)))
        sim = CivilViolenceSim(
            settings=settings,
            role="police",
            citizens=citizens,
            police=police,
        )
        with tempfile.TemporaryDirectory() as directory:
            await sim.run(runtime_for(script, concurrency=1), Path(directory) / "police")
        state = sim.snapshot()
        self.assertEqual(sorted(calls.values()), [1, 3])
        self.assertEqual(
            {c.agent_id for c in state.citizens if c.location is None},
            {accepted_target, second_target},
        )
        self.assertEqual(
            {p.location for p in state.police},
            {tuple(first_destination), tuple(second_destination)},
        )
        self.assertTrue(any("destination" in retry for retry in retry_messages))
        self.assertTrue(any("target" in retry for retry in retry_messages))

    async def test_police_observe_settled_citizens_locally_without_private_traits(self) -> None:
        views: list[PoliceObservation] = []

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            views.append(observation(messages))
            return ModelResponse(parts=[ToolCallPart("intervene", {})])

        citizens = tuple(Citizen(i, divmod(i, 5), -1000) for i in range(24))
        settings = CivilViolenceSettings(
            board_size=5,
            citizen_vision=1,
            police_vision=1,
            controlled_agent_count=1,
            max_transitions=1,
            seed_id=0,
        )
        sim = CivilViolenceSim(
            settings=settings,
            role="police",
            citizens=citizens,
            police=(Police(24, (4, 4)),),
        )
        initial = sim.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            await sim.run(runtime_for(script), Path(directory) / "police")
        settled = sim.snapshot()
        self.assertFalse(any(c.active for c in initial.citizens))
        self.assertEqual(len(views), 1)
        nearby = views[0]["neighborhood"]
        self.assertEqual(len(nearby), 8)
        self.assertNotIn("private_preference", json.dumps(views[0]))
        self.assertNotIn("mean_activity", json.dumps(views[0]))
        by_id = {c.agent_id: c for c in settled.citizens}
        self.assertTrue(any(cell["active"] for cell in nearby))
        for cell in nearby:
            citizen = by_id[cast(int, cell["agent_id"])]
            self.assertEqual(cell["active"], citizen.active)
            self.assertEqual((cell["row"], cell["column"]), citizen.location)
        self.assertNotIn(12, [cell["agent_id"] for cell in nearby])

    async def test_exhausted_invalid_police_turn_falls_back_to_no_arrest_and_stay(self) -> None:
        calls = 0

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(parts=[ToolCallPart("intervene", {"target_id": 9999})])

        citizens = tuple(Citizen(i, divmod(i, 3), -1000, True) for i in range(8))
        settings = CivilViolenceSettings(board_size=3, controlled_agent_count=1, max_transitions=1)
        sim = CivilViolenceSim(
            settings=settings,
            role="police",
            citizens=citizens,
            police=(Police(8, (2, 2)),),
        )
        with tempfile.TemporaryDirectory() as directory:
            await sim.run(runtime_for(script, concurrency=1), Path(directory) / "police")
        state = sim.snapshot()
        self.assertEqual(calls, 3)
        self.assertTrue(all(c.location is not None for c in state.citizens))
        self.assertEqual(state.police[0].location, (2, 2))
        self.assertTrue(sim.finished)

    async def test_police_score_is_reference_minus_controlled_activity_replayed_from_npz(
        self,
    ) -> None:
        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart("intervene", {})])

        settings = CivilViolenceSettings(
            board_size=5,
            citizen_density=0.8,
            police_density=0.12,
            private_preference_mean=-1000,
            controlled_agent_count=3,
            max_transitions=3,
        )
        sim = CivilViolenceSim(settings, role="police")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "police"
            score = await sim.run(runtime_for(script), output)
            means: list[float] = []
            for filename in ("ordinary.npz", "model-controlled.npz"):
                with np.load(output / filename, allow_pickle=False) as saved:
                    citizens = saved["roles"] == 0
                    measured = saved["active"][1:, citizens] & ~saved["jailed"][1:, citizens]
                    means.append(float(measured.mean()))
                    self.assertEqual(measured.shape, (3, 20))
            summary = json.loads((output / "result.json").read_text())
            self.assertAlmostEqual(score.value, means[0] - means[1])
            self.assertEqual(summary["role"], "police")
            self.assertEqual(len(summary["scored_agent_ids"]), 20)
            self.assertAlmostEqual(summary["ordinary"]["mean_activity"], means[0])
            self.assertAlmostEqual(summary["model_controlled"]["mean_activity"], means[1])
