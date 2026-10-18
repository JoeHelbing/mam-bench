import asyncio
import json
import tempfile
import unittest
from collections.abc import AsyncIterable
from decimal import Decimal
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx2 as httpx
from openai import APIConnectionError, APIError, AsyncOpenAI
from pydantic_ai import Agent, ModelMessage, RunContext, ToolOutput, UsageLimits
from pydantic_ai.agent import EventStreamHandler
from pydantic_ai.exceptions import ModelAPIError, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.memory import InMemoryStore, MemoryFile

from mam_bench.communication import MessageBoard
from mam_bench.config import AgentSettings
from mam_bench.diagnostics import ExecutionFailure
from mam_bench.sessions import AgentSessions, Completed, Stopped


def _response(
    *parts: TextPart | ToolCallPart, cost: Decimal | None = Decimal("0.01")
) -> ModelResponse:
    return ModelResponse(
        parts=list(parts),
        usage=RequestUsage(input_tokens=5, output_tokens=2, cost=cost),
        model_name="scripted",
        provider_name="test",
    )


def _latest_request(messages: list[ModelMessage]) -> ModelRequest:
    request = messages[-1]
    assert isinstance(request, ModelRequest)
    return request


def _assert_complete_tool_history(messages: list[ModelMessage]) -> None:
    pending: set[str] = set()
    for message in messages:
        if isinstance(message, ModelResponse):
            assert not pending, f"Unanswered calls before the next response: {pending}"
            pending.update(
                part.tool_call_id for part in message.parts if isinstance(part, ToolCallPart)
            )
        else:
            for part in message.parts:
                if isinstance(part, (ToolReturnPart, RetryPromptPart)):
                    pending.discard(part.tool_call_id)
    assert not pending, f"Unanswered calls at the end of the turn: {pending}"


def _request_text(request: ModelRequest) -> str:
    rendered: list[str] = []
    for part in request.parts:
        if not isinstance(part, UserPromptPart):
            continue
        if isinstance(part.content, str):
            rendered.append(part.content)
        else:
            rendered.extend(item.content for item in part.content if isinstance(item, TextContent))
    return "\n".join(rendered)


class MemoryModel:
    def __init__(self) -> None:
        self.tool_names: set[str] = set()
        self.memory_blocks: list[str] = []

    async def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        self.tool_names = {tool.name for tool in info.function_tools}
        request = _latest_request(messages)
        tool_results = [part for part in request.parts if isinstance(part, ToolReturnPart)]
        retry_parts = [part for part in request.parts if isinstance(part, RetryPromptPart)]
        text = _request_text(request)
        self.memory_blocks.extend(
            item.content
            for part in request.parts
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
            for item in part.content
            if isinstance(item, TextContent) and item.content.startswith("<memory>")
        )

        if "recall" in text:
            return _response(TextPart("remembered" if "alpha is durable" in text else "empty"))
        if tool_results:
            return _response(TextPart("stored"))
        if retry_parts:
            return _response(TextPart("protected"))
        if "remember alpha" in text:
            return _response(
                ToolCallPart(
                    tool_name="write_memory",
                    args={"content": "alpha is durable"},
                    tool_call_id="write-alpha",
                )
            )
        if "remember large" in text:
            lines = [f"old-{index:03d}-" + "x" * 100 for index in range(300)]
            lines.append("tail fact")
            return _response(
                ToolCallPart(
                    tool_name="write_memory",
                    args={"content": "\n".join(lines)},
                    tool_call_id="write-large",
                )
            )
        if "try deleting main" in text:
            return _response(
                ToolCallPart(
                    tool_name="delete_memory",
                    args={"file": "MEMORY.md"},
                    tool_call_id="delete-main",
                )
            )
        if "recall" in text:
            remembered = "alpha is durable" in text
            return _response(TextPart("remembered" if remembered else "empty"))
        return _response(TextPart("ok"))


class FailingReadStore(InMemoryStore):
    async def read(self, path: str, *, max_chars: int) -> MemoryFile | None:
        del path, max_chars
        raise OSError("memory unavailable")


class CompactionModel:
    def __init__(
        self,
        *,
        summary_error: Exception | None = None,
        normal_cost: Decimal | None = Decimal("0.01"),
        summary_cost: Decimal | None = Decimal("0.01"),
    ) -> None:
        self.normal_requests = 0
        self.summary_requests = 0
        self.summary_max_tokens: list[int | None] = []
        self.summary_error = summary_error
        self.normal_cost = normal_cost
        self.summary_cost = summary_cost

    async def __call__(self, messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if info.instructions is not None and info.instructions.startswith(
            "You are a context summarization assistant."
        ):
            self.summary_requests += 1
            settings = cast(dict[str, object], info.model_settings or {})
            value = settings.get("max_tokens")
            self.summary_max_tokens.append(value if isinstance(value, int) else None)
            if self.summary_error is not None:
                raise self.summary_error
            return _response(
                TextPart("## Current state\nThe compacted history is retained."),
                cost=self.summary_cost,
            )

        self.normal_requests += 1
        return _response(TextPart(f"normal-{self.normal_requests}"), cost=self.normal_cost)


async def _drain_events(ctx: RunContext[None], events: AsyncIterable[AgentStreamEvent]) -> None:
    async for _ in events:
        pass


class StreamingAgent(Agent[None, str]):
    @property
    def event_stream_handler(self) -> EventStreamHandler[None]:
        return _drain_events


class AgentSessionsTests(unittest.IsolatedAsyncioTestCase):
    async def test_transient_errors_use_the_request_budget_and_reset_next_turn(self) -> None:
        errors = [
            ModelHTTPError(408, "scripted"),
            ModelHTTPError(409, "scripted"),
            ModelHTTPError(429, "scripted"),
            ModelHTTPError(503, "scripted"),
            APIConnectionError(request=httpx.Request("POST", "https://provider.example.test/v1")),
        ]
        for error in errors:
            for limit in (1, 25):
                with self.subTest(error=error, limit=limit):
                    calls = 0

                    async def model(
                        messages: list[ModelMessage],
                        info: AgentInfo,
                        failure: Exception = error,
                        max_failures: int = min(limit, 3),
                    ) -> ModelResponse:
                        nonlocal calls
                        calls += 1
                        if calls <= max_failures:
                            raise failure
                        return _response(TextPart("recovered"))

                    runtime = AgentSessions(
                        Agent(FunctionModel(model)), context_window_tokens=260_000
                    )
                    self.assertEqual(
                        await runtime.run(
                            "a", "go", deps=None, usage_limits=UsageLimits(request_limit=limit)
                        ),
                        Stopped("usage_limit" if limit == 1 else "call_error_limit"),
                    )
                    self.assertEqual(calls, min(limit, 3))
                    self.assertEqual(runtime.usage.requests, calls)
                    self.assertEqual(
                        await runtime.run("a", "next", deps=None), Completed("recovered")
                    )

    async def test_stream_errors_are_bounded_and_partial_calls_are_not_executed(self) -> None:
        for failures in (1, 3):
            for partial in (False, True):
                with self.subTest(failures=failures, partial=partial):
                    calls = 0
                    directory = tempfile.TemporaryDirectory()
                    self.addCleanup(directory.cleanup)

                    def respond(
                        request: httpx.Request, failures: int = failures, partial: bool = partial
                    ) -> httpx.Response:
                        nonlocal calls
                        calls += 1
                        choice: dict[str, object] = {
                            "index": 0,
                            "delta": {"content": "recovered"},
                            "finish_reason": "stop",
                        }
                        if calls <= failures:
                            choice = {
                                "index": 0,
                                "delta": {
                                    "tool_calls": [
                                        {
                                            "index": 0,
                                            "id": f"partial-{calls}",
                                            "type": "function",
                                            "function": {
                                                "name": "post_message",
                                                "arguments": '{"text":"unexecuted"}',
                                            },
                                        }
                                    ],
                                },
                                "finish_reason": None,
                            }
                        chunk = {
                            "id": f"response-{calls}",
                            "object": "chat.completion.chunk",
                            "created": 1,
                            "model": "scripted",
                            "choices": [choice],
                        }
                        content = "data: " + json.dumps(chunk) + "\n\n"
                        if calls <= failures:
                            content = (
                                content if partial else ""
                            ) + 'data: {"error":{"message":"upstream unavailable","code":503}}\n\n'
                        else:
                            content += "data: [DONE]\n\n"
                        return httpx.Response(
                            200, headers={"content-type": "text/event-stream"}, content=content
                        )

                    async with AsyncOpenAI(
                        api_key="offline-test",
                        base_url="https://provider.example.test/v1",
                        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
                        max_retries=0,
                    ) as client:
                        runtime = AgentSessions(
                            StreamingAgent(
                                OpenAIChatModel(
                                    "scripted", provider=OpenAIProvider(openai_client=client)
                                ),
                                deps_type=type(None),
                            ),
                            context_window_tokens=260_000,
                            artifact_directory=Path(directory.name),
                        )
                        result = await runtime.run("a", "go", deps=None)
                        self.assertEqual(
                            result,
                            Stopped("call_error_limit")
                            if failures == 3
                            else Completed("recovered"),
                        )
                        self.assertEqual(calls, 3 if failures == 3 else 2)
                        self.assertEqual(runtime.usage.requests, calls)
                        self.assertEqual((await runtime.board.read("b")).content, "")
                        _assert_complete_tool_history(list(runtime.history("a")))
                        self.assertEqual(
                            await runtime.run("a", "next", deps=None), Completed("recovered")
                        )
                        self.assertEqual(
                            await runtime.run("b", "other", deps=None), Completed("recovered")
                        )

    async def test_api_and_tool_failures_share_a_budget_across_successes(self) -> None:
        calls = 0

        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            _assert_complete_tool_history(messages)
            match calls:
                case 1 | 5:
                    return _response(ToolCallPart("post_message", {"text": ""}, f"bad-{calls}"))
                case 2:
                    return _response(ToolCallPart("post_message", {"text": "saved"}, "post"))
                case 3:
                    raise ModelAPIError("scripted", "unavailable")
                case 4:
                    return _response(ToolCallPart("read_messages", {}, "read"))
                case _:
                    return _response(TextPart("recovered"))

        runtime = AgentSessions(
            Agent(FunctionModel(model), retries=5), context_window_tokens=260_000
        )
        self.assertEqual(await runtime.run("a", "go", deps=None), Stopped("call_error_limit"))
        self.assertEqual(calls, 5)
        self.assertEqual(runtime.usage.requests, 5)
        _assert_complete_tool_history(list(runtime.history("a")))
        self.assertEqual((await runtime.board.read("b")).content.count("saved"), 1)
        self.assertEqual(await runtime.run("a", "next", deps=None), Completed("recovered"))

    async def test_permanent_http_errors_still_abort_without_retry(self) -> None:
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status):
                calls = 0

                async def model(
                    messages: list[ModelMessage], info: AgentInfo, status: int = status
                ) -> ModelResponse:
                    nonlocal calls
                    calls += 1
                    raise ModelHTTPError(status, "scripted", {"message": "invalid setup"})

                runtime = AgentSessions(Agent(FunctionModel(model)), context_window_tokens=260_000)
                for identity in ("a", "b"):
                    with self.assertRaises(ExecutionFailure):
                        await runtime.run(identity, "go", deps=None)
                self.assertEqual(calls, 1)

    async def test_failed_api_call_retries_without_repeating_completed_tools(self) -> None:
        calls = 0

        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            del info
            _assert_complete_tool_history(messages)
            calls += 1
            if calls == 1:
                return _response(ToolCallPart("post_message", {"text": "saved"}, "post"))
            if calls == 2:
                raise APIError(
                    "stream unavailable",
                    request=httpx.Request("POST", "https://provider.example.test/v1"),
                    body={"code": 503},
                )
            return _response(TextPart("recovered"))

        runtime = AgentSessions(Agent(FunctionModel(model)), context_window_tokens=260_000)
        self.assertEqual(await runtime.run("a", "go", deps=None), Completed("recovered"))
        self.assertEqual(calls, 3)
        self.assertEqual(runtime.usage.requests, 3)
        self.assertEqual((await runtime.board.read("b")).content.count("saved"), 1)
        self.assertEqual(await runtime.run("a", "next", deps=None), Completed("recovered"))

    async def test_truncated_tool_call_ends_turn_and_preserves_valid_history(self) -> None:
        calls = 0
        truncate = True

        def answer(value: int) -> int:
            return value

        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if truncate:
                return ModelResponse(
                    parts=[ToolCallPart("answer", '{"value": ', f"truncated-{calls}")],
                    finish_reason="length",
                )
            _assert_complete_tool_history(messages)
            return _response(ToolCallPart("answer", {"value": 7}))

        sessions = AgentSessions(
            Agent(FunctionModel(model), output_type=ToolOutput(answer, name="answer"), retries=1),
            context_window_tokens=260_000,
        )
        self.assertEqual(await sessions.run("a", "go", deps=None), Stopped("output_token_limit"))
        truncate = False
        self.assertEqual(await sessions.run("a", "again", deps=None), Completed(7))
        self.assertLessEqual(calls, 3)

    async def test_output_token_limit_preserves_effects_usage_and_next_turn(self) -> None:
        calls = 0

        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return _response(ToolCallPart("post_message", {"text": "saved"}, "post"))
            if calls == 2:
                return ModelResponse(
                    parts=[ThinkingPart("unfinished reasoning")],
                    finish_reason="length",
                    usage=RequestUsage(input_tokens=10, output_tokens=32768),
                )
            _assert_complete_tool_history(messages)
            return _response(TextPart("recovered"))

        runtime = AgentSessions(Agent(FunctionModel(model)), context_window_tokens=260_000)
        with self.assertLogs("mam_bench.sessions", level="INFO") as logs:
            self.assertEqual(await runtime.run("a", "go", deps=None), Stopped("output_token_limit"))
        self.assertTrue(any("reason=output_token_limit" in line for line in logs.output))
        self.assertEqual(calls, 2)
        self.assertEqual(runtime.usage.output_tokens, 32770)
        self.assertIn("saved", (await runtime.board.read("b")).content)
        self.assertTrue(
            any(
                isinstance(message, ModelResponse) and message.finish_reason == "length"
                for message in runtime.history("a")
            )
        )
        self.assertEqual(await runtime.run("a", "again", deps=None), Completed("recovered"))

    async def test_private_memory_and_history(self) -> None:
        model = MemoryModel()
        runtime = AgentSessions(Agent(FunctionModel(model)), context_window_tokens=260_000)
        self.assertEqual(await runtime.run("a", "remember alpha", deps=None), Completed("stored"))
        self.assertEqual(await runtime.run("a", "recall", deps=None), Completed("remembered"))
        self.assertEqual(await runtime.run("b", "recall", deps=None), Completed("empty"))
        self.assertGreater(len(runtime.history("a")), len(runtime.history("b")))
        self.assertIn("post_message", model.tool_names)
        self.assertIn("read_messages", model.tool_names)
        self.assertEqual(runtime.usage.requests, 4)
        snapshot = runtime.usage
        snapshot.requests = 0
        self.assertEqual(runtime.usage.requests, 4)

    async def test_limit_preserves_memory_and_history_and_resets_per_turn(self) -> None:
        model = MemoryModel()
        runtime = AgentSessions(Agent(FunctionModel(model)), context_window_tokens=260_000)
        result = await runtime.run(
            "a", "remember alpha", deps=None, usage_limits=UsageLimits(request_limit=1)
        )
        self.assertEqual(result, Stopped("usage_limit"))
        self.assertEqual(runtime.usage.requests, 1)
        self.assertTrue(runtime.history("a"))
        self.assertEqual(await runtime.run("a", "recall", deps=None), Completed("remembered"))

    async def test_successful_tool_limit_and_multi_call(self) -> None:
        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            return _response(
                ToolCallPart("post_message", {"text": "one"}, "one"),
                ToolCallPart("post_message", {"text": "two"}, "two"),
            )

        runtime = AgentSessions(Agent(FunctionModel(model)), context_window_tokens=260_000)
        self.assertEqual(
            await runtime.run("a", "go", deps=None, usage_limits=UsageLimits(tool_calls_limit=2)),
            Stopped("usage_limit"),
        )
        self.assertEqual(runtime.usage.tool_calls, 2)
        page = await runtime.board.read("b")
        self.assertIn("one", page.content)
        self.assertIn("two", page.content)

    async def test_retry_exhaustion_preserves_session_and_provider_behavior_aborts(self) -> None:
        invalid = True

        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            if invalid:
                return _response(ToolCallPart("post_message", {"text": ""}, "bad"))
            return _response(TextPart("recovered"))

        runtime = AgentSessions(
            Agent(FunctionModel(model), retries=2), context_window_tokens=260_000
        )
        self.assertEqual(await runtime.run("a", "go", deps=None), Stopped("retry_exhaustion"))
        self.assertEqual(runtime.usage.requests, 3)
        invalid = False
        self.assertEqual(await runtime.run("a", "again", deps=None), Completed("recovered"))

        async def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            raise UnexpectedModelBehavior("provider returned broken body")

        failed = AgentSessions(Agent(FunctionModel(broken)), context_window_tokens=260_000)
        for identity in ("a", "b"):
            with self.assertRaisesRegex(ExecutionFailure, "unusable response"):
                await failed.run(identity, "go", deps=None)

    async def test_compaction_uses_same_model_and_native_usage(self) -> None:
        model = CompactionModel()
        runtime = AgentSessions(
            Agent(FunctionModel(model)),
            context_window_tokens=200,
            settings=AgentSettings(
                compaction_trigger_fraction=0.5,
                compaction_tail_tokens=20,
                summary_completion_tokens=16,
                memory_injection_tokens=100,
            ),
        )
        await runtime.run("a", "A" * 600, deps=None)
        self.assertEqual(await runtime.run("a", "B" * 600, deps=None), Completed("normal-2"))
        self.assertEqual(model.summary_requests, 1)
        self.assertEqual(model.summary_max_tokens, [16])
        self.assertEqual(runtime.usage.requests, 3)
        self.assertTrue(
            any(
                isinstance(part, SystemPromptPart)
                and "Summary of previous conversation" in part.content
                for message in runtime.history("a")
                if isinstance(message, ModelRequest)
                for part in message.parts
            )
        )

    async def test_memory_store_failure_poisons_runtime(self) -> None:
        with patch("mam_bench.sessions.InMemoryStore", return_value=FailingReadStore()):
            runtime = AgentSessions(
                Agent(FunctionModel(MemoryModel())), context_window_tokens=260_000
            )
        for identity in ("a", "b"):
            with self.assertRaisesRegex(ExecutionFailure, "notebook failed"):
                await runtime.run(identity, "go", deps=None)

    async def test_message_board_pages_and_isolated_cursors(self) -> None:
        board = MessageBoard()
        for index in range(12):
            self.assertEqual(
                await board.post("writer", str(index) + "x" * 3990, announcement=index == 0), index
            )
        first = await board.read("a")
        second = await board.read("a")
        self.assertLessEqual(len(first.content), 40000)
        self.assertTrue(first.more_available)
        self.assertFalse(second.more_available)
        self.assertIn("announcement by writer", first.content)
        self.assertEqual(await board.read("b"), first)
        self.assertEqual((await board.read("a")).content, "")

    async def test_default_request_limit_stops_successful_read_loop_at_25(self) -> None:
        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            return _response(ToolCallPart("read_messages", {}, str(len(messages))))

        runtime = AgentSessions(Agent(FunctionModel(model)), context_window_tokens=260_000)
        self.assertEqual(await runtime.run("a", "go", deps=None), Stopped("usage_limit"))
        self.assertEqual(runtime.usage.requests, 25)
        self.assertEqual(runtime.usage.tool_calls, 25)

    async def test_default_request_limit_does_not_cap_calls_in_one_response(self) -> None:
        requests = 0

        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal requests
            del messages, info
            requests += 1
            if requests == 1:
                return _response(
                    *(
                        ToolCallPart("post_message", {"text": str(i)}, f"post-{i}")
                        for i in range(26)
                    )
                )
            return _response(TextPart("done"))

        runtime = AgentSessions(Agent(FunctionModel(model)), context_window_tokens=260_000)
        self.assertEqual(await runtime.run("a", "go", deps=None), Completed("done"))
        self.assertEqual(runtime.usage.requests, 2)
        self.assertEqual(runtime.usage.tool_calls, 26)
        self.assertIn("25", (await runtime.board.read("b")).content)

    async def test_terminal_tool_takes_precedence_over_other_calls(self) -> None:
        async def stay() -> str:
            return "stayed"

        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            return _response(
                ToolCallPart("post_message", {"text": "must skip"}, "post"),
                ToolCallPart("stay", {}, "stay"),
            )

        runtime = AgentSessions(
            Agent(
                FunctionModel(model),
                output_type=ToolOutput(stay, name="stay"),
                end_strategy="early",
            ),
            context_window_tokens=260_000,
        )
        self.assertEqual(await runtime.run("a", "go", deps=None), Completed("stayed"))
        self.assertEqual((await runtime.board.read("b")).content, "")

    async def test_tool_only_text_exhaustion_is_normal_stop(self) -> None:
        async def stay() -> str:
            return "stay"

        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            return _response(TextPart("I refuse to call a tool"))

        runtime = AgentSessions(
            Agent(FunctionModel(model), output_type=ToolOutput(stay, name="stay"), retries=2),
            context_window_tokens=260_000,
        )
        self.assertEqual(await runtime.run("a", "go", deps=None), Stopped("retry_exhaustion"))
        self.assertEqual(runtime.usage.requests, 3)

    async def test_compaction_request_budget_stops_turn_and_preserves_session(self) -> None:
        model = CompactionModel()
        runtime = AgentSessions(
            Agent(FunctionModel(model)),
            context_window_tokens=200,
            settings=AgentSettings(
                compaction_trigger_fraction=0.5,
                compaction_tail_tokens=20,
                summary_completion_tokens=16,
                memory_injection_tokens=100,
            ),
        )
        self.assertEqual(await runtime.run("a", "A" * 600, deps=None), Completed("normal-1"))
        self.assertEqual(
            await runtime.run("a", "B" * 600, deps=None, usage_limits=UsageLimits(request_limit=1)),
            Stopped("usage_limit"),
        )
        self.assertEqual(runtime.usage.requests, 1)
        self.assertEqual(model.summary_requests, 0)
        self.assertEqual(await runtime.run("a", "C" * 600, deps=None), Completed("normal-2"))
        self.assertEqual(model.summary_requests, 1)

    async def test_compaction_fallback_and_unrecoverable_failure(self) -> None:
        settings = AgentSettings(
            compaction_trigger_fraction=0.5,
            compaction_tail_tokens=20,
            summary_completion_tokens=16,
            memory_injection_tokens=100,
        )
        fallback = CompactionModel(summary_error=ModelAPIError("scripted", "unavailable"))
        runtime = AgentSessions(
            Agent(FunctionModel(fallback)), context_window_tokens=200, settings=settings
        )
        await runtime.run("a", "A" * 600, deps=None)
        self.assertEqual(await runtime.run("a", "B" * 600, deps=None), Completed("normal-2"))
        self.assertEqual(fallback.summary_requests, 1)
        broken = CompactionModel(
            summary_error=UnexpectedModelBehavior("Exceeded maximum output retries (2)")
        )
        failed = AgentSessions(
            Agent(FunctionModel(broken)), context_window_tokens=200, settings=settings
        )
        await failed.run("a", "A" * 600, deps=None)
        from mam_bench.diagnostics import ExecutionFailure

        with self.assertRaises(ExecutionFailure) as raised:
            await failed.run("a", "B" * 600, deps=None)
        self.assertEqual(raised.exception.kind, "compaction")

    async def test_same_session_serializes_while_other_sessions_can_overlap(self) -> None:
        first_started = asyncio.Event()
        release_first = asyncio.Event()
        calls = 0

        async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            del messages, info
            calls += 1
            if calls == 1:
                first_started.set()
                await release_first.wait()
            return _response(TextPart("ok"))

        runtime = AgentSessions(Agent(FunctionModel(model)), context_window_tokens=260_000)
        first = asyncio.create_task(runtime.run("a", "first", deps=None))
        await first_started.wait()
        second = asyncio.create_task(runtime.run("a", "second", deps=None))
        await asyncio.sleep(0)
        self.assertEqual(calls, 1)
        self.assertEqual(await runtime.run("b", "other", deps=None), Completed("ok"))
        release_first.set()
        self.assertEqual(await first, Completed("ok"))
        self.assertEqual(await second, Completed("ok"))
        self.assertEqual(len(runtime.history("a")), 4)
