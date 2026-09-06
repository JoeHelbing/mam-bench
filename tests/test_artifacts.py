import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic_ai import Agent, ModelMessagesTypeAdapter
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai_harness.step_persistence import FileStepStore

from mam_bench.agent import AgentSessionRuntime
from mam_bench.benchmark import AgentInfrastructureFailure
from mam_bench.communication import MessageBoard
from mam_bench.config import AgentSettings


class ArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_archive_keeps_precompaction_messages_and_reasoning(self) -> None:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ThinkingPart("recorded reasoning"), TextPart("short summary")]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = AgentSessionRuntime(
                Agent(FunctionModel(respond)),
                artifact_directory=root,
                settings=AgentSettings(
                    context_window_tokens=200,
                    compaction_tail_tokens=20,
                    summary_completion_tokens=16,
                    memory_injection_tokens=100,
                ),
            )
            await runtime.run("agent-a", "A" * 600, deps=None)
            await runtime.run("agent-a", "B" * 600, deps=None)
            store = FileStepStore(root / "agent-messages", media_store=None)
            runs = await store.list_runs(conversation_id="agent-a")
            self.assertEqual(len(runs), 2)
            snapshots = [
                snapshot
                for run in runs
                for snapshot in await store.list_snapshots(
                    run_id=run.run_id, include_interrupted=True
                )
            ]
            serialized = [
                ModelMessagesTypeAdapter.dump_json(snapshot.messages) for snapshot in snapshots
            ]
            self.assertTrue(any(b"A" * 600 in value for value in serialized))
            self.assertTrue(any(b"recorded reasoning" in value for value in serialized))
            self.assertNotIn(
                b"A" * 600, ModelMessagesTypeAdapter.dump_json(list(runtime.history("agent-a")))
            )
            for value in serialized:
                self.assertTrue(ModelMessagesTypeAdapter.validate_json(value))

    async def test_failed_run_keeps_tool_messages_and_board_without_exception_body(self) -> None:
        calls = 0

        async def fail_after_post(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart("post_message", {"text": "hello board"})])
            raise ModelAPIError("offline", "SECRET-PROVIDER-BODY")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = AgentSessionRuntime(
                Agent(FunctionModel(fail_after_post)), artifact_directory=root
            )
            with self.assertRaises(AgentInfrastructureFailure):
                await runtime.run("agent-a", "hello agent", deps=None)
            rows = [
                json.loads(line) for line in (root / "message-board.jsonl").read_text().splitlines()
            ]
            self.assertEqual(
                rows,
                [
                    {
                        "sequence": 0,
                        "author_id": "agent-a",
                        "text": "hello board",
                        "announcement": False,
                    }
                ],
            )
            store = FileStepStore(root / "agent-messages", media_store=None)
            runs = await store.list_runs(conversation_id="agent-a")
            events = await store.list_events(run_id=runs[0].run_id)
            self.assertTrue(any(event.kind == "run_failed" for event in events))
            snapshots = await store.list_snapshots(run_id=runs[0].run_id, include_interrupted=True)
            self.assertTrue(snapshots)
            history = ModelMessagesTypeAdapter.dump_json(snapshots[-1].messages)
            self.assertIn(b"tool-call", history)
            self.assertIn(b"tool-return", history)
            for path in root.rglob("*"):
                if path.is_file():
                    self.assertNotIn("SECRET-PROVIDER-BODY", path.read_text())

    async def test_board_archive_preserves_concurrent_order_and_announcements(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "message-board.jsonl"
            board = MessageBoard(archive_path=path)
            await board.post("simulation", "round started", announcement=True)
            await asyncio.gather(*(board.post(str(i), f"message {i}") for i in range(20)))
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([row["sequence"] for row in rows], list(range(21)))
            self.assertTrue(rows[0]["announcement"])
            self.assertEqual({row["author_id"] for row in rows[1:]}, {str(i) for i in range(20)})

    async def test_archive_write_failure_aborts_as_artifact_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runtime = AgentSessionRuntime(
                Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("ok")]))),
                artifact_directory=Path(directory),
            )
            with (
                patch.object(FileStepStore, "_sync_register_run", side_effect=OSError("disk full")),
                self.assertRaises(AgentInfrastructureFailure) as raised,
            ):
                await runtime.run("agent-a", "hello", deps=None)
            self.assertEqual(raised.exception.kind, "artifact_write")
