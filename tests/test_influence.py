import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

import numpy as np

from mam_bench.runtime import (
    RuntimeCapability,
    RuntimeDescriptor,
    RuntimeMessage,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from mam_bench.simulations.schelling.evaluation import (
    ActorWaveContext,
    CoordinationPost,
    InfluenceEvaluationConfig,
    InfluenceInfrastructureError,
    MoveDecision,
    MoveProposal,
    SteeringObjective,
    build_counterfactual_reference,
    influence_actor_ids,
    ordinary_edge_homophily,
    run_model_evaluation,
    validate_evaluation_artifact,
)
from mam_bench.simulations.schelling.profile import landscape_cell
from mam_bench.simulations.schelling.reference import EMPTY_CELL, TerminalStatus
from mam_bench_analysis.evaluation_analysis import analyze_model_evaluation


class FixedInfluenceTeam:
    runtime_descriptor = RuntimeDescriptor(
        model_id="fixed-influence-team",
        runtime="deterministic-test",
        provider="test",
        model="test/fixed-influence-team",
        adapter_version="fixed-v1",
        capabilities=frozenset(RuntimeCapability),
        sampling_controls=frozenset(),
        max_concurrent_requests=16,
    )

    def __init__(self) -> None:
        self.coordination_calls = 0
        self.movement_calls = 0
        self.histories: dict[int, list[RuntimeMessage]] = {
            actor_id: [] for actor_id in influence_actor_ids(300)
        }

    async def coordinate(
        self, contexts: tuple[ActorWaveContext, ...]
    ) -> tuple[CoordinationPost, ...]:
        self.coordination_calls += 1
        posts: list[CoordinationPost] = []
        for context in contexts:
            messages = _fixed_messages(context, phase="coordination")
            self.histories[context.actor_id].extend(messages)
            posts.append(
                CoordinationPost(
                    actor_id=context.actor_id,
                    text=f"round {context.round_number}",
                    request_count=2,
                    input_tokens=10,
                    output_tokens=2,
                    latency_seconds=0.01,
                    new_messages_json=_messages_json(messages),
                )
            )
        return tuple(posts)

    async def move(self, contexts: tuple[ActorWaveContext, ...]) -> tuple[MoveProposal, ...]:
        self.movement_calls += 1
        vacancy = int(np.flatnonzero(contexts[0].current_cell_types.ravel() == EMPTY_CELL)[0])
        row, column = divmod(vacancy, contexts[0].current_cell_types.shape[0])
        proposals: list[MoveProposal] = []
        for context in contexts:
            messages = _fixed_messages(context, phase="movement")
            self.histories[context.actor_id].extend(messages)
            proposals.append(
                MoveProposal(
                    actor_id=context.actor_id,
                    decision=MoveDecision(stay=False, row=row, column=column),
                    request_count=2,
                    input_tokens=10,
                    output_tokens=2,
                    latency_seconds=0.01,
                    new_messages_json=_messages_json(messages),
                )
            )
        return tuple(proposals)

    def history_payloads(self) -> dict[int, bytes]:
        return {
            actor_id: _messages_json(messages).encode("utf-8")
            for actor_id, messages in self.histories.items()
        }


def _fixed_messages(
    context: ActorWaveContext,
    *,
    phase: str,
) -> list[RuntimeMessage]:
    inspect_call_id = f"round-{context.round_number}-{phase}-actor-{context.actor_id}-inspect"
    messages = [
        RuntimeMessage(role="user", parts=(TextPart(text="inspect state"),)),
        RuntimeMessage(
            role="assistant",
            parts=(
                ToolCallPart(
                    call_id=inspect_call_id,
                    name="inspect_state",
                    arguments={
                        "board_round_offset": 0,
                        "coordination_round_offset": 0,
                        "include_reference": False,
                    },
                ),
            ),
        ),
        RuntimeMessage(
            role="tool",
            parts=(
                ToolResultPart(
                    call_id=inspect_call_id,
                    name="inspect_state",
                    result={"selected": True},
                ),
            ),
        ),
        RuntimeMessage(role="user", parts=(TextPart(text=f"{phase} action"),)),
    ]
    if phase == "coordination":
        messages.append(
            RuntimeMessage(
                role="assistant",
                parts=(TextPart(text=f"round {context.round_number}"),),
            )
        )
    else:
        vacancy = int(np.flatnonzero(context.current_cell_types.ravel() == EMPTY_CELL)[0])
        row, column = divmod(vacancy, context.current_cell_types.shape[0])
        messages.append(
            RuntimeMessage(
                role="assistant",
                parts=(
                    ToolCallPart(
                        call_id=(
                            f"round-{context.round_number}-movement-actor-{context.actor_id}-move"
                        ),
                        name="submit_move",
                        arguments={"stay": False, "row": row, "column": column},
                    ),
                ),
            )
        )
    return messages


def _messages_json(messages: list[RuntimeMessage]) -> str:
    return json.dumps([message.model_dump(mode="json") for message in messages])


class InfrastructureFailingTeam(FixedInfluenceTeam):
    async def coordinate(
        self, contexts: tuple[ActorWaveContext, ...]
    ) -> tuple[CoordinationPost, ...]:
        raise InfluenceInfrastructureError(
            "Authorization: Bearer do-not-store endpoint unavailable"
        )


class MovementInfrastructureFailingTeam(FixedInfluenceTeam):
    async def move(self, contexts: tuple[ActorWaveContext, ...]) -> tuple[MoveProposal, ...]:
        raise InfluenceInfrastructureError("endpoint unavailable")


class CredentialLeakingTeam(FixedInfluenceTeam):
    def history_payloads(self) -> dict[int, bytes]:
        payloads = super().history_payloads()
        payloads[0] = _messages_json(
            [RuntimeMessage(role="user", parts=(TextPart(text="Bearer do-not-store"),))]
        ).encode("utf-8")
        return payloads


class InfluenceMetricTests(unittest.TestCase):
    def test_fixed_actor_ids_are_symmetric(self) -> None:
        self.assertEqual(
            influence_actor_ids(300),
            (*range(8), *range(150, 158)),
        )

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
        reference = build_counterfactual_reference(landscape_cell(17, 3), seed_id=22)

        self.assertEqual(reference.seed_id, 22)
        self.assertEqual(reference.terminal_status, TerminalStatus.EQUILIBRIUM)
        self.assertEqual(reference.rounds_completed, 14)
        self.assertAlmostEqual(reference.masked_final_homophily, 0.9894, places=4)
        self.assertEqual(reference.masked_agent_count, 284)


class ModelEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fake_team_runs_the_complete_twenty_round_case(self) -> None:
        team = FixedInfluenceTeam()
        config = InfluenceEvaluationConfig(
            cell=landscape_cell(17, 3),
            seed_id=22,
            objective=SteeringObjective.INTEGRATION,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case"

            result = await run_model_evaluation(config, team, output)
            validated = validate_evaluation_artifact(output)
            comparison = analyze_model_evaluation(output)

            self.assertEqual(result.rounds_completed, 20)
            self.assertEqual(result.cell_types.shape, (21, 20, 20))
            self.assertEqual(result.agent_locations.shape, (21, 300))
            self.assertEqual(team.coordination_calls, 20)
            self.assertEqual(team.movement_calls, 20)
            self.assertEqual(len(result.coordination_posts), 20 * 16)
            self.assertEqual(result.actor_collision_count, 20 * 15)
            self.assertEqual(validated.final_directional_lift, result.final_directional_lift)
            self.assertEqual(
                comparison.reference_final_homophily,
                result.reference.masked_final_homophily,
            )
            self.assertEqual(comparison.reference_final_satisfaction, 1.0)
            self.assertEqual(
                comparison.model_final_homophily,
                float(result.ordinary_homophily[-1]),
            )
            self.assertEqual(len(comparison.reference_trajectory.rounds), 15)
            self.assertEqual(len(comparison.model_trajectory.rounds), 21)
            self.assertEqual(comparison.coordination_post_count, 320)
            self.assertTrue((output / "run.json").is_file())
            self.assertTrue((output / "trajectory.npz").is_file())
            self.assertTrue((output / "events.jsonl").is_file())
            board_path = output / "coordination-board.jsonl"
            self.assertTrue(board_path.is_file())
            board_rounds = [
                json.loads(line) for line in board_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(board_rounds), 20)
            self.assertEqual(board_rounds[0]["round_number"], 1)
            self.assertEqual(
                [post["actor_id"] for post in board_rounds[0]["posts"]],
                list(influence_actor_ids(300)),
            )
            self.assertEqual(
                [post["text"] for post in board_rounds[0]["posts"]],
                ["round 1"] * 16,
            )
            self.assertEqual(
                set(board_rounds[0]["posts"][0]),
                {"actor_id", "text", "policy_error"},
            )
            self.assertEqual(len(tuple((output / "actor-histories").glob("*.json"))), 16)
            event_types = [
                json.loads(line)["event_type"]
                for line in (output / "events.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertIn("phase_boundary", event_types)
            self.assertIn("tool_call", event_types)
            self.assertIn("model_usage", event_types)
            self.assertIn("coordination_post", event_types)
            self.assertIn("move_proposal", event_types)
            self.assertIn("move_outcome", event_types)
            self.assertEqual(event_types[0], "phase_boundary")
            checkpoint_directory = output / "checkpoints"
            self.assertEqual(len(tuple(checkpoint_directory.glob("*.json"))), 42)
            self.assertEqual(len(tuple(checkpoint_directory.glob("*.npz"))), 42)
            summary = json.loads((output / "run.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["accepted_actor_move_count"], 20)
            self.assertEqual(summary["request_count"], 20 * 2 * 16 * 2)
            self.assertEqual(summary["input_tokens"], 20 * 2 * 16 * 10)
            self.assertEqual(summary["output_tokens"], 20 * 2 * 16 * 2)
            self.assertAlmostEqual(summary["model_latency_seconds"], 20 * 2 * 16 * 0.01)
            self.assertIn("best_directional_lift", summary)

    async def test_validator_rejects_deleted_duplicated_changed_or_reordered_events(
        self,
    ) -> None:
        mutations: dict[str, Callable[[list[str]], list[str]]] = {
            "deleted": lambda lines: lines[:100] + lines[101:],
            "duplicated": lambda lines: lines[:100] + [lines[100]] + lines[100:],
            "changed": _change_event,
            "reordered": lambda lines: [lines[1], lines[0], *lines[2:]],
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "case"
                await run_model_evaluation(
                    InfluenceEvaluationConfig(
                        cell=landscape_cell(17, 3),
                        seed_id=22,
                        objective=SteeringObjective.INTEGRATION,
                    ),
                    FixedInfluenceTeam(),
                    output,
                )
                events_path = output / "events.jsonl"
                lines = events_path.read_text(encoding="utf-8").splitlines()
                events_path.write_text(
                    "\n".join(mutate(lines)) + "\n",
                    encoding="utf-8",
                )

                with self.assertRaises(ValueError):
                    validate_evaluation_artifact(output)

    async def test_validator_recomputes_counterfactual_reference(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case"
            await run_model_evaluation(
                InfluenceEvaluationConfig(
                    cell=landscape_cell(17, 3),
                    seed_id=22,
                    objective=SteeringObjective.INTEGRATION,
                ),
                FixedInfluenceTeam(),
                output,
            )
            run_path = output / "run.json"
            summary = json.loads(run_path.read_text(encoding="utf-8"))
            summary["reference_masked_final_homophily"] += 0.1
            summary["final_directional_lift"] += 0.1
            run_path.write_text(json.dumps(summary), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "Counterfactual Reference"):
                validate_evaluation_artifact(output)

    async def test_validator_rejects_board_that_diverges_from_events(self) -> None:
        team = FixedInfluenceTeam()
        config = InfluenceEvaluationConfig(
            cell=landscape_cell(17, 3),
            seed_id=22,
            objective=SteeringObjective.INTEGRATION,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case"
            await run_model_evaluation(config, team, output)
            board_path = output / "coordination-board.jsonl"
            board_lines = board_path.read_text(encoding="utf-8").splitlines()
            first_round = json.loads(board_lines[0])
            first_round["posts"][0]["text"] = "tampered"
            board_lines[0] = json.dumps(first_round)
            board_path.write_text("\n".join(board_lines) + "\n", encoding="utf-8")

            with self.assertRaisesRegex(
                ValueError, "Coordination Board does not match coordination events"
            ):
                validate_evaluation_artifact(output)

    async def test_board_retains_published_round_before_movement_failure(self) -> None:
        team = MovementInfrastructureFailingTeam()
        config = InfluenceEvaluationConfig(
            cell=landscape_cell(17, 3),
            seed_id=22,
            objective=SteeringObjective.INTEGRATION,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case"

            with self.assertRaises(InfluenceInfrastructureError):
                await run_model_evaluation(config, team, output)

            board_lines = (
                (output / "coordination-board.jsonl").read_text(encoding="utf-8").splitlines()
            )
            self.assertEqual(len(board_lines), 1)
            board_round = json.loads(board_lines[0])
            self.assertEqual(board_round["round_number"], 1)
            self.assertEqual(len(board_round["posts"]), 16)
            checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(checkpoint["phase"], "infrastructure_failure")
            checkpoint_directory = output / "checkpoints"
            self.assertEqual(len(tuple(checkpoint_directory.glob("*.json"))), 3)
            self.assertEqual(len(tuple(checkpoint_directory.glob("*.npz"))), 3)

    async def test_credentials_are_rejected_before_history_persistence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case"

            with self.assertRaisesRegex(ValueError, "cannot contain credentials"):
                await run_model_evaluation(
                    InfluenceEvaluationConfig(
                        cell=landscape_cell(17, 3),
                        seed_id=22,
                        objective=SteeringObjective.INTEGRATION,
                    ),
                    CredentialLeakingTeam(),
                    output,
                )

            stored_text = "\n".join(
                path.read_text(encoding="utf-8")
                for path in output.rglob("*")
                if path.is_file() and path.suffix in {".json", ".jsonl"}
            )
            self.assertNotIn("do-not-store", stored_text)

    async def test_infrastructure_failure_checkpoints_an_unscored_case(self) -> None:
        team = InfrastructureFailingTeam()
        config = InfluenceEvaluationConfig(
            cell=landscape_cell(17, 3),
            seed_id=22,
            objective=SteeringObjective.INTEGRATION,
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "case"

            with self.assertRaises(InfluenceInfrastructureError):
                await run_model_evaluation(config, team, output)

            self.assertFalse((output / "run.json").exists())
            failure = json.loads((output / "failure.json").read_text(encoding="utf-8"))
            checkpoint = json.loads((output / "checkpoint.json").read_text(encoding="utf-8"))
            self.assertEqual(failure["status"], "infrastructure_failure")
            self.assertNotIn("do-not-store", json.dumps(failure))
            self.assertEqual(checkpoint["phase"], "infrastructure_failure")
            self.assertEqual(checkpoint["state_count"], 1)
            checkpoint_directory = output / "checkpoints"
            self.assertEqual(len(tuple(checkpoint_directory.glob("*.json"))), 2)
            self.assertEqual(len(tuple(checkpoint_directory.glob("*.npz"))), 2)


def _change_event(lines: list[str]) -> list[str]:
    changed = list(lines)
    event = json.loads(changed[100])
    event["round_number"] = 20
    changed[100] = json.dumps(event)
    return changed


if __name__ == "__main__":
    unittest.main()
