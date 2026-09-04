import asyncio
import unittest
from dataclasses import dataclass
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

from mam_bench.agent import AgentSessionRuntime, SharedRecord
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


@dataclass(frozen=True)
class SharedPayload:
    text: str


def _render_payloads(records: tuple[SharedRecord[SharedPayload], ...]) -> str:
    return "|".join(record.payload.text for record in records)


class SharedCommunicationTests(unittest.IsolatedAsyncioTestCase):
    def _runtime(self) -> AgentSessionRuntime[None, str, SharedPayload]:
        return AgentSessionRuntime(Agent(FunctionModel(MemoryModel()), output_type=str))

    async def test_pages_complete_records_with_isolated_cursors_and_trial_reset(self) -> None:
        runtime = self._runtime()
        communication = runtime.communication
        first = await communication.append("session-a", SharedPayload("aaaa"))
        second = await communication.append("session-b", SharedPayload("bb"))
        third = await communication.append("session-a", SharedPayload("ccc"))

        page_one = await communication.read(
            "reader-a",
            max_chars=7,
            renderer=_render_payloads,
        )
        page_two = await communication.read(
            "reader-a",
            max_chars=7,
            renderer=_render_payloads,
        )
        other_reader = await communication.read(
            "reader-b",
            max_chars=100,
            renderer=_render_payloads,
        )
        oversized_first = await communication.read(
            "reader-c",
            max_chars=2,
            renderer=_render_payloads,
        )
        empty = await communication.read(
            "reader-a",
            max_chars=7,
            renderer=_render_payloads,
        )
        fresh = await self._runtime().communication.read(
            "reader-a",
            max_chars=7,
            renderer=_render_payloads,
        )

        self.assertEqual((first.sequence, second.sequence, third.sequence), (0, 1, 2))
        self.assertEqual(first.author_session_id, "session-a")
        self.assertEqual(page_one.records, (first, second))
        self.assertEqual(page_one.content, "aaaa|bb")
        self.assertTrue(page_one.more_available)
        self.assertEqual(page_two.records, (third,))
        self.assertFalse(page_two.more_available)
        self.assertEqual(other_reader.records, (first, second, third))
        self.assertEqual(oversized_first.records, (first,))
        self.assertEqual(oversized_first.content, "aaaa")
        self.assertTrue(oversized_first.more_available)
        self.assertEqual(empty.records, ())
        self.assertEqual(empty.content, "")
        self.assertFalse(empty.more_available)
        self.assertEqual(empty.next_cursor, page_two.next_cursor)
        self.assertEqual(fresh.records, ())

    async def test_read_is_linearized_and_a_failed_render_does_not_advance_cursor(self) -> None:
        runtime = self._runtime()
        communication = runtime.communication
        first = await communication.append("writer", SharedPayload("first"))
        renderer_entered = asyncio.Event()
        release_renderer = asyncio.Event()

        async def controlled_renderer(
            records: tuple[SharedRecord[SharedPayload], ...],
        ) -> str:
            renderer_entered.set()
            await release_renderer.wait()
            return _render_payloads(records)

        read_task = asyncio.create_task(
            communication.read("reader", max_chars=100, renderer=controlled_renderer)
        )
        await renderer_entered.wait()
        append_task = asyncio.create_task(
            communication.append("writer", SharedPayload("after-read"))
        )
        await asyncio.sleep(0)
        self.assertFalse(append_task.done())

        release_renderer.set()
        first_page = await read_task
        second = await append_task
        second_page = await communication.read(
            "reader",
            max_chars=100,
            renderer=_render_payloads,
        )

        self.assertEqual(first_page.records, (first,))
        self.assertEqual(second_page.records, (second,))

        def broken_renderer(records: tuple[SharedRecord[SharedPayload], ...]) -> str:
            del records
            raise RuntimeError("cannot render")

        third = await communication.append("writer", SharedPayload("third"))
        with self.assertRaisesRegex(RuntimeError, "cannot render"):
            await communication.read("failing-reader", max_chars=100, renderer=broken_renderer)
        recovered = await communication.read(
            "failing-reader",
            max_chars=100,
            renderer=_render_payloads,
        )
        self.assertEqual(recovered.records, (first, second, third))

    async def test_rolling_admission_is_bounded_and_preserves_completion_order(self) -> None:
        settings = AgentSettings(concurrency=2)
        runtime: AgentSessionRuntime[None, str, SharedPayload] = AgentSessionRuntime(
            Agent(FunctionModel(MemoryModel()), output_type=str),
            settings=settings,
        )
        gates = [asyncio.Event() for _ in range(5)]
        started = [asyncio.Event() for _ in range(5)]
        active = 0
        maximum_active = 0

        async def worker(item: int) -> int:
            nonlocal active, maximum_active
            active += 1
            maximum_active = max(maximum_active, active)
            started[item].set()
            try:
                await gates[item].wait()
                await runtime.communication.append("worker", SharedPayload(str(item)))
                return item * 10
            finally:
                active -= 1

        rolling_task = asyncio.create_task(runtime.run_rolling(range(5), worker))
        await started[0].wait()
        await started[1].wait()
        gates[1].set()
        await started[2].wait()
        self.assertFalse(gates[0].is_set())
        gates[2].set()
        await started[3].wait()
        gates[0].set()
        await started[4].wait()
        gates[4].set()
        await asyncio.sleep(0)
        gates[3].set()
        completed = await rolling_task
        publication = await runtime.communication.read(
            "auditor",
            max_chars=100,
            renderer=_render_payloads,
        )

        self.assertEqual(maximum_active, 2)
        self.assertEqual([item.item for item in completed], [1, 2, 0, 4, 3])
        self.assertEqual([item.result for item in completed], [10, 20, 0, 40, 30])
        self.assertEqual([item.admission_sequence for item in completed], [1, 2, 0, 4, 3])
        self.assertEqual([item.completion_sequence for item in completed], list(range(5)))
        self.assertEqual(publication.content, "1|2|0|4|3")


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
