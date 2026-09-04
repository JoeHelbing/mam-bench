"""Persistent, model-agnostic PydanticAI sessions for benchmark trials.

The runtime owns only generic agent mechanics: normalized message histories,
private in-memory notebooks, bounded context, and usage. Benchmark simulations
continue to own prompts, tools, outputs, and domain policy.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import cast

from anyio import Lock
from pydantic_ai import Agent, AgentRunResult, ModelMessage, RunUsage, UsageLimits, UserContent
from pydantic_ai.capabilities import AbstractCapability, AgentCapability
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import KnownModelName, Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai_harness import (
    FallbackCompaction,
    Memory,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.memory import InMemoryStore

from mam_bench.benchmark import AgentSettings


@dataclass(frozen=True)
class AgentSessionUsage:
    """Normalized aggregate usage for successful session runs."""

    model_requests: int = 0
    summary_requests: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    output_tokens: int = 0
    input_audio_tokens: int = 0
    cache_audio_read_tokens: int = 0
    output_audio_tokens: int = 0
    cost_usd: Decimal | None = None
    details: tuple[tuple[str, int], ...] = ()

    @property
    def total_requests(self) -> int:
        """Return evaluated-model requests plus successful summary requests."""
        return self.model_requests + self.summary_requests

    @property
    def total_tokens(self) -> int:
        """Return all non-audio input, cache, and output tokens."""
        return (
            self.input_tokens
            + self.cache_write_tokens
            + self.cache_read_tokens
            + self.output_tokens
        )


@dataclass
class _SessionState:
    history: list[ModelMessage] = field(default_factory=lambda: list[ModelMessage]())
    usage: RunUsage = field(default_factory=RunUsage)
    model_requests: int = 0
    cost_complete: bool = True


@dataclass
class _RequestTracker[AgentDepsT](AbstractCapability[AgentDepsT]):
    """Track evaluated-model responses after PydanticAI resolves their costs."""

    responses: list[ModelResponse] = field(default_factory=lambda: list[ModelResponse]())

    @property
    def requests(self) -> int:
        return len(self.responses)

    @property
    def cost_complete(self) -> bool:
        return all(response.usage.cost is not None for response in self.responses)

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        del ctx, request_context
        self.responses.append(response)
        return response


@dataclass
class _SummaryTracker:
    responses: list[ModelResponse] = field(default_factory=lambda: list[ModelResponse]())

    @property
    def requests(self) -> int:
        return len(self.responses)

    @property
    def cost_complete(self) -> bool:
        return all(response.usage.cost is not None for response in self.responses)

    def record(self, response: ModelResponse) -> None:
        self.responses.append(response)


class _TrackedSummaryModel(WrapperModel):
    """Delegate to the evaluated model while observing summary response costs."""

    def __init__(self, wrapped: Model | KnownModelName, tracker: _SummaryTracker) -> None:
        super().__init__(wrapped)
        self._tracker = tracker

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        response = await self.wrapped.request(messages, model_settings, model_request_parameters)
        self._tracker.record(response)
        return response


class AgentSessionRuntime[AgentDepsT, OutputDataT]:
    """Run one reusable PydanticAI agent through isolated persistent sessions.

    Construct one runtime per benchmark trial. Each opaque session ID gets a
    normalized PydanticAI history and a private namespace in the runtime's
    fresh in-memory Harness store.
    """

    def __init__(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
        *,
        settings: AgentSettings | None = None,
    ) -> None:
        self._agent = agent
        self._settings = settings or AgentSettings()
        self._store = InMemoryStore()
        self._states: dict[str, _SessionState] = {}
        self._locks: dict[str, Lock] = {}
        self._usage = RunUsage()
        self._model_requests = 0
        self._cost_complete = True

    @property
    def settings(self) -> AgentSettings:
        """Return the immutable protocol settings."""
        return self._settings

    @property
    def usage(self) -> AgentSessionUsage:
        """Return usage aggregated across every session in this trial."""
        return _usage_snapshot(
            self._usage,
            model_requests=self._model_requests,
            cost_complete=self._cost_complete,
        )

    def session_usage(self, session_id: str) -> AgentSessionUsage:
        """Return usage for one opaque session identity."""
        state = self._state(session_id)
        return _usage_snapshot(
            state.usage,
            model_requests=state.model_requests,
            cost_complete=state.cost_complete,
        )

    def history(self, session_id: str) -> tuple[ModelMessage, ...]:
        """Return the session's normalized PydanticAI message history."""
        return tuple(self._state(session_id).history)

    async def run(
        self,
        session_id: str,
        user_prompt: str | Sequence[UserContent] | None,
        *,
        deps: AgentDepsT,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
    ) -> AgentRunResult[OutputDataT]:
        """Run one turn and retain successful history, notebook, and usage.

        Calls for the same session serialize so concurrent callers cannot fork
        one history. Failures propagate and leave the prior successful history
        and aggregate usage unchanged.
        """
        state = self._state(session_id)
        lock = self._locks.setdefault(session_id, Lock())
        async with lock:
            staged_store = self._staged_memory(session_id)
            model_tracker: _RequestTracker[AgentDepsT] = _RequestTracker()
            summary_tracker = _SummaryTracker()
            capabilities: tuple[AgentCapability[AgentDepsT], ...] = (
                self._compaction(summary_tracker),
                self._memory(session_id, staged_store),
                model_tracker,
            )
            result = await self._agent.run(
                user_prompt,
                message_history=state.history,
                deps=deps,
                model_settings=model_settings,
                usage_limits=usage_limits,
                capabilities=capabilities,
            )
            run_usage = result.usage
            summary_requests = run_usage.requests - model_tracker.requests
            if summary_requests != summary_tracker.requests:
                raise RuntimeError("session usage could not distinguish summary requests")

            await self._commit_memory(session_id, staged_store)
            state.history = result.all_messages()
            state.usage.incr(run_usage)
            state.model_requests += model_tracker.requests
            if not model_tracker.cost_complete or not summary_tracker.cost_complete:
                state.cost_complete = False
                self._cost_complete = False
            self._usage.incr(run_usage)
            self._model_requests += model_tracker.requests
            return result

    def _state(self, session_id: str) -> _SessionState:
        if not session_id:
            raise ValueError("session_id must not be empty")
        return self._states.setdefault(session_id, _SessionState())

    def _namespace(self, session_id: str) -> str:
        return f"session-{hashlib.sha256(session_id.encode()).hexdigest()}"

    def _staged_memory(self, session_id: str) -> InMemoryStore:
        prefix = f"{self._namespace(session_id)}/agent/"
        files = {
            path: content for path, content in self._store.files.items() if path.startswith(prefix)
        }
        return InMemoryStore(files)

    async def _commit_memory(self, session_id: str, staged_store: InMemoryStore) -> None:
        prefix = f"{self._namespace(session_id)}/agent/"
        staged_files = {
            path: content for path, content in staged_store.files.items() if path.startswith(prefix)
        }
        current_paths = await self._store.list_paths(
            prefix,
            limit=max(1, len(self._store.files) + 1),
        )
        for path in current_paths:
            if path in staged_files:
                continue
            current = await self._store.read(path, max_chars=1)
            if current is not None:
                await self._store.delete(path, expected_version=current.version)
        for path, content in staged_files.items():
            current = await self._store.read(path, max_chars=max(1, len(content) + 1))
            if current is not None and not current.truncated and current.content == content:
                continue
            await self._store.write(
                path,
                content,
                expected_version=None if current is None else current.version,
            )

    def _memory(self, session_id: str, store: InMemoryStore) -> Memory[AgentDepsT]:
        return Memory(
            store=store,
            namespace=self._namespace(session_id),
            agent_name="agent",
            max_tokens=self._settings.memory_injection_tokens,
            injection_errors="raise",
        )

    def _compaction(self, tracker: _SummaryTracker) -> TieredCompaction[AgentDepsT]:
        settings = self._settings
        model = self._agent.model
        if model is None:
            raise RuntimeError("the Agent Session Runtime requires an Agent with a model")
        summarizer: SummarizingCompaction[AgentDepsT] = SummarizingCompaction(
            model=_TrackedSummaryModel(cast("Model | KnownModelName", model), tracker),
            max_tokens=1,
            keep_tokens=settings.compaction_tail_tokens,
            incremental=True,
            preserve_first_user_message=False,
            receipts=True,
            model_settings={"max_tokens": settings.summary_completion_tokens},
        )
        sliding_window: SlidingWindowCompaction[AgentDepsT] = SlidingWindowCompaction(
            max_tokens=1,
            keep_tokens=settings.compaction_tail_tokens,
            preserve_first_user_message=False,
            receipts=True,
        )
        fallback: FallbackCompaction[AgentDepsT] = FallbackCompaction([summarizer, sliding_window])
        return TieredCompaction(
            tiers=[fallback],
            target_fraction=settings.compaction_trigger_fraction,
            context_window=settings.context_window_tokens,
        )


def _usage_snapshot(
    usage: RunUsage,
    *,
    model_requests: int,
    cost_complete: bool,
) -> AgentSessionUsage:
    summary_requests = usage.requests - model_requests
    if summary_requests < 0:
        raise RuntimeError("session usage counted more model requests than total requests")
    return AgentSessionUsage(
        model_requests=model_requests,
        summary_requests=summary_requests,
        tool_calls=usage.tool_calls,
        input_tokens=usage.input_tokens,
        cache_write_tokens=usage.cache_write_tokens,
        cache_read_tokens=usage.cache_read_tokens,
        output_tokens=usage.output_tokens,
        input_audio_tokens=usage.input_audio_tokens,
        cache_audio_read_tokens=usage.cache_audio_read_tokens,
        output_audio_tokens=usage.output_audio_tokens,
        cost_usd=usage.cost if cost_complete else None,
        details=tuple(sorted(usage.details.items())),
    )
