import unittest
from decimal import Decimal
from typing import cast
from unittest.mock import patch

from pydantic_ai import Agent, ModelMessage
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.memory import InMemoryStore, MemoryFile

from mam_bench.agent import AgentSessionRuntime
from mam_bench.benchmark import AgentSettings


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


class AgentSessionRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_persists_history_and_private_memory_per_opaque_session(self) -> None:
        model = MemoryModel()
        agent = Agent(FunctionModel(model), output_type=str)
        runtime = AgentSessionRuntime(agent)

        first = await runtime.run("actor/../7", "remember alpha", deps=None)
        recalled = await runtime.run("actor/../7", "recall", deps=None)
        isolated = await runtime.run("actor-8", "recall", deps=None)
        fresh_trial = AgentSessionRuntime(agent)
        fresh = await fresh_trial.run("actor/../7", "recall", deps=None)

        self.assertEqual(first.output, "stored")
        self.assertEqual(recalled.output, "remembered")
        self.assertEqual(isolated.output, "empty")
        self.assertEqual(fresh.output, "empty")
        self.assertEqual(
            model.tool_names,
            {"write_memory", "read_memory", "search_memory", "delete_memory"},
        )
        self.assertGreater(len(runtime.history("actor/../7")), len(runtime.history("actor-8")))

    async def test_memory_injection_is_bounded_to_approximately_four_thousand_tokens(self) -> None:
        model = MemoryModel()
        runtime = AgentSessionRuntime(Agent(FunctionModel(model), output_type=str))

        await runtime.run("actor-7", "remember large", deps=None)
        await runtime.run("actor-7", "recall", deps=None)

        injected = model.memory_blocks[-1]
        self.assertLessEqual(len(injected), 4_000 * 4)
        self.assertIn("tail fact", injected)
        self.assertNotIn("old-000", injected)

    async def test_memory_injection_store_failure_propagates(self) -> None:
        model = MemoryModel()
        with patch("mam_bench.agent.InMemoryStore", return_value=FailingReadStore()):
            runtime = AgentSessionRuntime(Agent(FunctionModel(model), output_type=str))
            with self.assertRaisesRegex(OSError, "memory unavailable"):
                await runtime.run("actor-7", "recall", deps=None)
        self.assertEqual(runtime.history("actor-7"), ())
        self.assertEqual(runtime.usage.total_requests, 0)

    async def test_main_memory_file_cannot_be_deleted(self) -> None:
        model = MemoryModel()
        runtime = AgentSessionRuntime(Agent(FunctionModel(model), output_type=str))

        result = await runtime.run("actor-7", "try deleting main", deps=None)

        self.assertEqual(result.output, "protected")
        self.assertTrue(
            any(
                isinstance(part, RetryPromptPart) and "main notebook" in str(part.content)
                for message in runtime.history("actor-7")
                if isinstance(message, ModelRequest)
                for part in message.parts
            )
        )

    async def test_failed_run_does_not_commit_memory_side_effects(self) -> None:
        failed_once = False

        async def write_then_fail(
            messages: list[ModelMessage],
            info: AgentInfo,
        ) -> ModelResponse:
            nonlocal failed_once
            del info
            request = _latest_request(messages)
            if any(isinstance(part, ToolReturnPart) for part in request.parts):
                failed_once = True
                raise RuntimeError("provider failed after write")
            text = _request_text(request)
            if "recall" in text:
                return _response(TextPart("remembered" if "ghost memory" in text else "empty"))
            return _response(
                ToolCallPart(
                    tool_name="write_memory",
                    args={"content": "ghost memory"},
                    tool_call_id="ghost-write",
                )
            )

        runtime = AgentSessionRuntime(Agent(FunctionModel(write_then_fail), output_type=str))

        with self.assertRaisesRegex(RuntimeError, "provider failed after write"):
            await runtime.run("actor-7", "write then fail", deps=None)
        result = await runtime.run("actor-7", "recall", deps=None)

        self.assertTrue(failed_once)
        self.assertEqual(result.output, "empty")
        self.assertEqual(runtime.usage.total_requests, 1)

    async def test_compaction_uses_inherited_model_and_reports_summary_usage(self) -> None:
        model = CompactionModel()
        settings = AgentSettings(
            context_window_tokens=200,
            compaction_trigger_fraction=0.5,
            compaction_tail_tokens=20,
            summary_completion_tokens=16,
            memory_injection_tokens=100,
        )
        runtime = AgentSessionRuntime(
            Agent(FunctionModel(model), output_type=str), settings=settings
        )

        await runtime.run("actor-7", "A" * 600, deps=None)
        result = await runtime.run("actor-7", "B" * 600, deps=None)

        self.assertEqual(result.output, "normal-2")
        self.assertEqual(model.summary_requests, 1)
        self.assertEqual(model.summary_max_tokens, [16])
        self.assertEqual(runtime.usage.model_requests, 2)
        self.assertEqual(runtime.usage.summary_requests, 1)
        self.assertEqual(runtime.usage.total_requests, 3)
        self.assertEqual(runtime.usage.cost_usd, Decimal("0.03"))
        self.assertTrue(
            any(
                isinstance(part, SystemPromptPart)
                and part.content.startswith("Summary of previous conversation:")
                for message in runtime.history("actor-7")
                if isinstance(message, ModelRequest)
                for part in message.parts
            )
        )
        self.assertTrue(_has_compaction_receipt(runtime.history("actor-7"), "summarized"))

    async def test_approved_summary_model_failure_uses_sliding_window_fallback(self) -> None:
        model = CompactionModel(summary_error=ModelAPIError("scripted", "summary unavailable"))
        settings = AgentSettings(
            context_window_tokens=200,
            compaction_trigger_fraction=0.5,
            compaction_tail_tokens=20,
            summary_completion_tokens=16,
            memory_injection_tokens=100,
        )
        runtime = AgentSessionRuntime(
            Agent(FunctionModel(model), output_type=str), settings=settings
        )

        await runtime.run("actor-7", "A" * 600, deps=None)
        result = await runtime.run("actor-7", "B" * 600, deps=None)

        self.assertEqual(result.output, "normal-2")
        self.assertEqual(model.summary_requests, 1)
        self.assertEqual(runtime.usage.model_requests, 2)
        self.assertEqual(runtime.usage.summary_requests, 0)
        self.assertEqual(runtime.usage.total_requests, 2)
        self.assertTrue(_has_compaction_receipt(runtime.history("actor-7"), "dropped"))

    async def test_unapproved_compaction_failure_aborts_without_replacing_history(self) -> None:
        model = CompactionModel(summary_error=RuntimeError("broken compactor"))
        settings = AgentSettings(
            context_window_tokens=200,
            compaction_trigger_fraction=0.5,
            compaction_tail_tokens=20,
            summary_completion_tokens=16,
            memory_injection_tokens=100,
        )
        runtime = AgentSessionRuntime(
            Agent(FunctionModel(model), output_type=str), settings=settings
        )

        await runtime.run("actor-7", "A" * 600, deps=None)
        before = runtime.history("actor-7")
        with self.assertRaisesRegex(RuntimeError, "broken compactor"):
            await runtime.run("actor-7", "B" * 600, deps=None)

        self.assertEqual(runtime.history("actor-7"), before)
        self.assertEqual(runtime.usage.total_requests, 1)

    async def test_mixed_known_and_unknown_request_cost_remains_null(self) -> None:
        request_number = 0

        async def mixed_cost_model(
            messages: list[ModelMessage],
            info: AgentInfo,
        ) -> ModelResponse:
            nonlocal request_number
            del info
            request_number += 1
            request = _latest_request(messages)
            if any(isinstance(part, ToolReturnPart) for part in request.parts):
                return _response(TextPart("done"), cost=Decimal("0.01"))
            return _response(
                ToolCallPart(
                    tool_name="write_memory",
                    args={"content": "cost test"},
                    tool_call_id="mixed-cost-write",
                ),
                cost=None,
            )

        runtime = AgentSessionRuntime(Agent(FunctionModel(mixed_cost_model), output_type=str))

        await runtime.run("actor-7", "write", deps=None)

        self.assertEqual(request_number, 2)
        self.assertIsNone(runtime.usage.cost_usd)

    async def test_unknown_summary_cost_makes_aggregate_cost_null(self) -> None:
        model = CompactionModel(summary_cost=None)
        settings = AgentSettings(
            context_window_tokens=200,
            compaction_trigger_fraction=0.5,
            compaction_tail_tokens=20,
            summary_completion_tokens=16,
            memory_injection_tokens=100,
        )
        runtime = AgentSessionRuntime(
            Agent(FunctionModel(model), output_type=str),
            settings=settings,
        )

        await runtime.run("actor-7", "A" * 600, deps=None)
        await runtime.run("actor-7", "B" * 600, deps=None)

        self.assertEqual(runtime.usage.summary_requests, 1)
        self.assertIsNone(runtime.usage.cost_usd)

    async def test_cost_remains_null_when_the_model_reports_no_trustworthy_price(self) -> None:
        async def costless_model(
            messages: list[ModelMessage],
            info: AgentInfo,
        ) -> ModelResponse:
            del messages, info
            return _response(TextPart("ok"), cost=None)

        runtime = AgentSessionRuntime(Agent(FunctionModel(costless_model), output_type=str))

        await runtime.run("actor-7", "hello", deps=None)

        self.assertIsNone(runtime.usage.cost_usd)

    def test_protocol_defaults_are_fixed(self) -> None:
        settings = AgentSettings()

        self.assertEqual(settings.context_window_tokens, 260_000)
        self.assertEqual(settings.compaction_trigger_fraction, 0.7)
        self.assertEqual(settings.compaction_tail_tokens, 40_000)
        self.assertEqual(settings.summary_completion_tokens, 16_000)
        self.assertEqual(settings.memory_injection_tokens, 4_000)


def _has_compaction_receipt(messages: tuple[ModelMessage, ...], word: str) -> bool:
    return any(
        isinstance(item, TextContent)
        and item.content.startswith("[History")
        and word in item.content
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
        for item in part.content
    )
