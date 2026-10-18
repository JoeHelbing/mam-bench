import asyncio
import json
import tempfile
import unittest
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import patch

import httpx2 as httpx
from openai import AsyncOpenAI
from pydantic_ai import Agent, ModelMessagesTypeAdapter, ToolOutput
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.providers.vllm import VLLMProvider
from pydantic_ai_harness.step_persistence import FileStepStore

from mam_bench.communication import MessageBoard
from mam_bench.config import AgentSettings
from mam_bench.diagnostics import ExecutionFailure
from mam_bench.kludge import VLLMChatModel
from mam_bench.sessions import AgentSessions, Stopped
from support import streaming_model


class ArtifactTests(unittest.IsolatedAsyncioTestCase):
    async def test_timeout_keeps_raw_vllm_tokens_even_without_parsed_content(self) -> None:
        class Body(httpx.AsyncByteStream):
            async def __aiter__(self) -> AsyncIterator[bytes]:
                chunk: dict[str, object] = {
                    "id": "raw-request-1",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "scripted",
                    "choices": [
                        {"index": 0, "delta": {}, "token_ids": [11, 12], "finish_reason": None}
                    ],
                    "prompt_token_ids": [1, 2],
                }
                yield ("data: " + json.dumps(chunk) + "\n\n").encode()
                # The empty delta produces no native text/tool event. It must
                # already be on disk while the provider is still generating.
                files = list((root / "provider-streams").glob("*.jsonl"))
                self_test.assertEqual(len(files), 1)
                row = json.loads(files[0].read_text().splitlines()[0])
                self_test.assertEqual(row["chunk"]["choices"][0]["token_ids"], [11, 12])
                await asyncio.Event().wait()

        def respond(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.content)
            self.assertIs(body["stream"], True)
            self.assertIs(body.get("return_token_ids"), True)
            self.assertIs(body.get("include_reasoning"), True)
            self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
            return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=Body())

        self_test = self
        async with AsyncOpenAI(
            api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        ) as client:
            model = VLLMChatModel(
                "scripted",
                provider=VLLMProvider(openai_client=client),
                settings={"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}},
            )
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runtime = AgentSessions(
                    Agent(model),
                    context_window_tokens=260_000,
                    artifact_directory=root,
                    settings=AgentSettings(timeout_seconds=2),
                )
                with self.assertRaises(ExecutionFailure) as raised:
                    await runtime.run("agent-a", "generate", deps=None)
                self.assertEqual(raised.exception.kind, "timeout")
                (path,) = (root / "provider-streams").glob("*.jsonl")
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                self.assertEqual(rows[0]["conversation_id"], "agent-a")
                self.assertEqual(rows[0]["run_step"], 1)
                self.assertEqual(rows[0]["chunk"]["prompt_token_ids"], [1, 2])
                self.assertEqual(rows[0]["chunk"]["id"], "raw-request-1")
                self.assertEqual(rows[0]["chunk"]["choices"][0]["token_ids"], [11, 12])
                store = FileStepStore(root / "agent-messages", media_store=None)
                (run,) = await store.list_runs(conversation_id="agent-a")
                self.assertTrue(
                    any(e.kind == "run_failed" for e in await store.list_events(run_id=run.run_id))
                )

    async def test_timeout_keeps_streamed_output_before_request_finishes(self) -> None:
        async def stream(messages: list[ModelMessage], info: AgentInfo) -> AsyncIterator[str]:
            yield "unfinished output"
            self.assertIn(
                "unfinished output", next((root / "agent-streams").glob("*.jsonl")).read_text()
            )
            await asyncio.Event().wait()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = AgentSessions(
                Agent(FunctionModel(stream_function=stream)),
                context_window_tokens=260_000,
                artifact_directory=root,
                settings=AgentSettings(timeout_seconds=2),
            )
            with self.assertRaises(ExecutionFailure) as raised:
                await runtime.run("agent-a", "start a long answer", deps=None)
            self.assertEqual(raised.exception.kind, "timeout")
            files = list((root / "agent-streams").glob("*.jsonl"))
            self.assertEqual(len(files), 1)
            rows = [json.loads(line) for line in files[0].read_text().splitlines()]
            self.assertEqual(rows[0]["conversation_id"], "agent-a")
            self.assertEqual(rows[0]["event"]["part"]["content"], "unfinished output")
            store = FileStepStore(root / "agent-messages", media_store=None)
            (run,) = await store.list_runs(conversation_id="agent-a")
            events = await store.list_events(run_id=run.run_id)
            self.assertTrue(any(event.kind == "run_failed" for event in events))

    async def test_streamed_repetition_retains_partial_tool_and_stop_cause(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.assertIs(json.loads(request.content)["stream"], True)
            chunks: list[dict[str, object]] = [
                {
                    "id": "partial",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "scripted",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {"name": "finish", "arguments": '{"total":'},
                                    }
                                ],
                            },
                            "finish_reason": None,
                            "token_ids": [101, 102],
                        }
                    ],
                },
                {
                    "id": "partial",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "scripted",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "repetition"}],
                },
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                + "data: [DONE]\n\n",
            )

        async def finish(total: int) -> int:
            return total

        async with AsyncOpenAI(
            api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        ) as client:
            model = VLLMChatModel("scripted", provider=VLLMProvider(openai_client=client))
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runtime = AgentSessions(
                    Agent(model, output_type=ToolOutput(finish)),
                    context_window_tokens=260_000,
                    artifact_directory=root,
                )
                self.assertEqual(
                    await runtime.run("agent-a", "finish", deps=None), Stopped("output_token_limit")
                )
                store = FileStepStore(root / "agent-messages", media_store=None)
                (run,) = await store.list_runs(conversation_id="agent-a")
                snapshots = await store.list_snapshots(run_id=run.run_id, include_interrupted=True)
                responses = [m for m in snapshots[-1].messages if isinstance(m, ModelResponse)]
                self.assertEqual(
                    (responses[-1].provider_details or {})["finish_reason"], "repetition"
                )
                self.assertIn("call-1", next((root / "agent-streams").glob("*.jsonl")).read_text())
                chunks = [
                    json.loads(line)["chunk"]
                    for line in next((root / "provider-streams").glob("*.jsonl"))
                    .read_text()
                    .splitlines()
                ]
                self.assertEqual(chunks[0]["choices"][0]["token_ids"], [101, 102])
                self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "repetition")

    async def test_native_archive_keeps_precompaction_messages_and_reasoning(self) -> None:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[ThinkingPart("recorded reasoning"), TextPart("short summary")]
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = AgentSessions(
                Agent(streaming_model(FunctionModel(respond))),
                context_window_tokens=200,
                artifact_directory=root,
                settings=AgentSettings(
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

    async def test_failed_run_keeps_tool_messages_board_and_exception_body(self) -> None:
        calls = 0

        async def fail_after_post(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(parts=[ToolCallPart("post_message", {"text": "hello board"})])
            raise ModelAPIError("offline", "SECRET-PROVIDER-BODY")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            runtime = AgentSessions(
                Agent(streaming_model(FunctionModel(fail_after_post))),
                context_window_tokens=260_000,
                artifact_directory=root,
            )
            with self.assertRaises(ExecutionFailure):
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
            failed = next(event for event in events if event.kind == "run_failed")
            self.assertIn("SECRET-PROVIDER-BODY", failed.error or "")

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
            runtime = AgentSessions(
                Agent(FunctionModel(lambda messages, info: ModelResponse(parts=[TextPart("ok")]))),
                context_window_tokens=260_000,
                artifact_directory=Path(directory),
            )
            with (
                patch.object(FileStepStore, "_sync_register_run", side_effect=OSError("disk full")),
                self.assertRaises(ExecutionFailure) as raised,
            ):
                await runtime.run("agent-a", "hello", deps=None)
            self.assertEqual(raised.exception.kind, "artifact_write")
