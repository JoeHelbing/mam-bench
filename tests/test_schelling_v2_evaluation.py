import json
import tempfile
import unittest
from collections import defaultdict
from decimal import Decimal
from importlib.resources import as_file, files
from pathlib import Path

import numpy as np
from pydantic_ai.usage import RunUsage

from mam_bench.benchmark import PairFailure, RuntimeInfo
from mam_bench.communication import SharedCommunication
from mam_bench.evidence import EvidenceEvent, EvidenceRecorder
from mam_bench.simulations.schelling.agent import SchellingTurnCoordinator
from mam_bench.simulations.schelling.fixture import load_evaluation_reference
from mam_bench.simulations.schelling.models import (
    ActorTurnContext,
    ActorTurnCoordinator,
    ActorTurnResult,
    InfluenceEvaluationConfig,
    PublicDocumentRecord,
    Stay,
    SteeringObjective,
)
from mam_bench.simulations.schelling.profile import LandscapeCell, Rational
from mam_bench.simulations.schelling.runtime import (
    EvidencePublicationError,
    influence_actor_ids,
    ordinary_edge_homophily,
    ordinary_satisfaction_fraction,
    publish_v2_failure,
    publish_v2_success,
    run_v2_model_evaluation,
)
from mam_bench.usage import AgentSessionUsage


class NullNotebook:
    async def write(
        self,
        content: str,
        *,
        file: str = "MEMORY.md",
        old_text: str | None = None,
    ) -> object:
        del content, file, old_text
        return object()

    async def delete(self, file: str) -> object:
        del file
        return object()


class ScriptedEvaluationTeam:
    runtime_info = RuntimeInfo(
        model_id="scripted-v2",
        provider="openrouter",
        model="test/scripted-v2",
        endpoint="https://example.test/v1",
    )

    def __init__(self) -> None:
        self.turns_by_round: dict[int, list[int]] = defaultdict(list)
        self.context_homophily: dict[int, list[float]] = defaultdict(list)
        self._model_requests = 0
        self._usage = RunUsage()

    @property
    def usage(self) -> AgentSessionUsage:
        return AgentSessionUsage(
            usage=self._usage,
            model_requests=self._model_requests,
        )

    async def run_turn(
        self,
        context: ActorTurnContext,
        coordinator: ActorTurnCoordinator,
    ) -> ActorTurnResult:
        self.turns_by_round[context.round_number].append(context.actor_id)
        self.context_homophily[context.round_number].append(context.current_homophily)
        accepted_calls = 1
        rejections: tuple[str, ...] = ()
        request_count = 1
        if context.actor_id == 0:
            rejections = ("scripted policy retry",)
            request_count += 1
        elif context.actor_id == 1:
            accepted_calls += 1
            request_count += 1
        summary_requests = int(context.actor_id == 2)
        self._model_requests += request_count
        total_requests = request_count + summary_requests
        self._usage.incr(
            RunUsage(
                requests=total_requests,
                tool_calls=accepted_calls,
                input_tokens=10 * total_requests,
                cache_read_tokens=4 * total_requests,
                output_tokens=2 * total_requests,
                cost=Decimal("0.01") * total_requests,
                details={"reasoning_tokens": 2 * total_requests},
            )
        )
        action = await coordinator.commit_terminal(
            context.actor_id,
            Stay(),
            NullNotebook(),
        )
        return ActorTurnResult(
            actor_id=context.actor_id,
            action=action,
            accepted_call_count=accepted_calls,
            policy_rejections=rejections,
            memory_operation_count=int(context.actor_id == 1),
        )


class V2ModelEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_publishes_exact_atomic_redacted_failure_evidence(self) -> None:
        recorder = EvidenceRecorder()
        await recorder.record(
            "model_request_started",
            model_id="scripted-v2",
            api_key="must-not-appear",
        )
        failure = PairFailure(
            simulation_id="schelling-influence-pilot-v1",
            simulation_version="schelling-influence-v2",
            model_id="scripted-v2",
            provider="openrouter",
            model="test/scripted-v2",
            kind="provider",
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pair"
            publish_v2_failure(output, failure, recorder.events)

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"failure.json", "failed-events.jsonl"},
            )
            failure_payload = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            self.assertEqual(failure_payload, failure.model_dump(mode="json"))
            events_text = (output / "failed-events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("must-not-appear", events_text)
            self.assertIn("[REDACTED]", events_text)

            with self.assertRaises(FileExistsError):
                publish_v2_failure(output, failure, recorder.events)

            failed_output = root / "failed-pair"
            invalid = EvidenceEvent(1, "2026-01-01T00:00:00+00:00", "invalid", {})
            with self.assertRaises(ValueError):
                publish_v2_failure(failed_output, failure, (invalid,))
            self.assertFalse(failed_output.exists())

    async def test_publishes_exact_atomic_success_evidence(self) -> None:
        resource = files("mam_bench.data").joinpath("schelling-reference-v2")
        with as_file(resource) as dataset_root:
            reference = load_evaluation_reference(dataset_root)
        config = InfluenceEvaluationConfig(
            cell=LandscapeCell(20, Rational(3, 4), Rational(1, 4)),
            seed_id=50,
            objective=SteeringObjective.INTEGRATION,
        )
        team = ScriptedEvaluationTeam()
        evidence = EvidenceRecorder()
        communication: SharedCommunication[PublicDocumentRecord] = SharedCommunication()
        coordinator = SchellingTurnCoordinator(communication, evidence)
        result = await run_v2_model_evaluation(
            config,
            team,
            coordinator,
            reference=reference,
            evidence=evidence,
        )
        for actor_id in influence_actor_ids(config.cell.agent_count):
            await evidence.record("notebook_snapshot", session_id=str(actor_id), files={})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "pair"
            publish_v2_success(output, result, evidence.events)

            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"run.json", "trajectory.npz", "events.jsonl"},
            )
            summary = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["simulation_version"], "schelling-influence-v2")
            expected_usage = {
                "model_requests": 540,
                "summary_requests": 30,
                "tool_calls": 510,
                "input_tokens": 5700,
                "cache_write_tokens": 0,
                "cache_read_tokens": 2280,
                "output_tokens": 1140,
                "input_audio_tokens": 0,
                "cache_audio_read_tokens": 0,
                "output_audio_tokens": 0,
                "cost_usd": "5.70",
                "details": [["reasoning_tokens", 1140]],
                "accepted_calls": 510,
                "policy_rejections": 30,
                "memory_operations": 30,
            }
            self.assertEqual(summary["usage"], expected_usage)
            events = [
                json.loads(line)
                for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([event["sequence"] for event in events], list(range(len(events))))
            self.assertEqual(
                sum(event["kind"] == "notebook_snapshot" for event in events),
                16,
            )
            self.assertTrue(
                {
                    "round_started",
                    "actor_admitted",
                    "actor_turn_started",
                    "stay",
                    "actor_turn_completed",
                    "round_settled",
                    "round_usage",
                    "evaluation_usage",
                }.issubset({event["kind"] for event in events})
            )
            evaluation_usage = next(
                event["usage"] for event in events if event["kind"] == "evaluation_usage"
            )
            self.assertEqual(evaluation_usage, expected_usage)
            round_usage = [event["usage"] for event in events if event["kind"] == "round_usage"]
            self.assertEqual(len(round_usage), 30)
            self.assertTrue(all(item["cost_usd"] == "0.19" for item in round_usage))
            self.assertTrue(
                all(item["details"] == [["reasoning_tokens", 38]] for item in round_usage)
            )

            with self.assertRaises(FileExistsError):
                publish_v2_success(output, result, evidence.events)

            failed_output = root / "failed-pair"
            invalid = EvidenceEvent(0, "2026-01-01T00:00:00+00:00", "invalid", {"bad": object()})
            with self.assertRaises(EvidencePublicationError):
                publish_v2_success(failed_output, result, (invalid,))
            self.assertFalse(failed_output.exists())

    async def test_runs_and_scores_thirty_rounds_with_scripted_turns(self) -> None:
        resource = files("mam_bench.data").joinpath("schelling-reference-v2")
        with as_file(resource) as dataset_root:
            reference = load_evaluation_reference(dataset_root)
        initial_cell_types = reference.initial_cell_types.copy()
        initial_locations = reference.initial_agent_locations.copy()
        terminal_cell_types = reference.terminal_cell_types.copy()
        terminal_locations = reference.terminal_agent_locations.copy()
        config = InfluenceEvaluationConfig(
            cell=LandscapeCell(20, Rational(3, 4), Rational(1, 4)),
            seed_id=50,
            objective=SteeringObjective.INTEGRATION,
        )
        team = ScriptedEvaluationTeam()
        communication: SharedCommunication[PublicDocumentRecord] = SharedCommunication()
        coordinator = SchellingTurnCoordinator(communication)

        result = await run_v2_model_evaluation(
            config,
            team,
            coordinator,
            reference=reference,
        )

        self.assertEqual(result.simulation_id, "schelling-influence-pilot-v1")
        self.assertEqual(result.simulation_version, "schelling-influence-v2")
        self.assertEqual(result.rounds_completed, 30)
        self.assertEqual(result.cell_types.shape, (31, 20, 20))
        self.assertEqual(result.agent_locations.shape, (31, 300))
        self.assertEqual(reference.rounds_completed, 20)
        np.testing.assert_array_equal(reference.initial_cell_types, initial_cell_types)
        np.testing.assert_array_equal(reference.initial_agent_locations, initial_locations)
        np.testing.assert_array_equal(reference.terminal_cell_types, terminal_cell_types)
        np.testing.assert_array_equal(reference.terminal_agent_locations, terminal_locations)

        actor_ids = influence_actor_ids(config.cell.agent_count)
        for round_number in range(1, 31):
            self.assertEqual(set(team.turns_by_round[round_number]), set(actor_ids))
            self.assertEqual(len(team.turns_by_round[round_number]), 16)
            self.assertTrue(
                all(
                    value == result.ordinary_homophily[round_number - 1]
                    for value in team.context_homophily[round_number]
                )
            )

        expected_homophily = np.asarray(
            [
                ordinary_edge_homophily(
                    locations,
                    reference.agent_types,
                    excluded_agent_ids=actor_ids,
                    grid_size=config.cell.board_size,
                )
                for locations in result.agent_locations
            ]
        )
        expected_satisfaction = np.asarray(
            [
                ordinary_satisfaction_fraction(cell_types, locations, cell=config.cell)
                for cell_types, locations in zip(
                    result.cell_types,
                    result.agent_locations,
                    strict=True,
                )
            ]
        )
        np.testing.assert_allclose(result.ordinary_homophily, expected_homophily)
        np.testing.assert_allclose(result.ordinary_satisfaction, expected_satisfaction)
        directional = reference.masked_final_homophily - expected_homophily
        self.assertEqual(result.final_directional_lift, directional[-1])
        self.assertEqual(result.best_directional_lift, max(directional))
        self.assertEqual(result.directional_trajectory_area, np.mean(directional))

        self.assertEqual(len(result.round_usage), 30)
        self.assertTrue(all(item.accepted_calls == 17 for item in result.round_usage))
        self.assertTrue(all(item.policy_rejections == 1 for item in result.round_usage))
        self.assertTrue(all(item.session.model_requests == 18 for item in result.round_usage))
        self.assertTrue(all(item.session.summary_requests == 1 for item in result.round_usage))
        self.assertTrue(all(item.session.usage.tool_calls == 17 for item in result.round_usage))
        self.assertTrue(all(item.memory_operations == 1 for item in result.round_usage))
        self.assertEqual(result.usage.session.model_requests, 30 * 18)
        self.assertEqual(result.usage.session.summary_requests, 30)
        self.assertEqual(result.usage.session.usage.tool_calls, 30 * 17)
        self.assertEqual(result.usage.accepted_calls, 30 * 17)
        self.assertEqual(result.usage.policy_rejections, 30)
        self.assertEqual(result.usage.memory_operations, 30)
        self.assertTrue(np.all(result.round_model_latency_seconds > 0))
        self.assertEqual(sum(item.accepted_calls for item in result.round_usage), 30 * 17)
        self.assertEqual(sum(item.policy_rejections for item in result.round_usage), 30)


if __name__ == "__main__":
    unittest.main()
