"""Trial-local PydanticAI sessions with private notebooks and compaction.

One reusable Agent serves isolated histories and notebooks. A failed trial
cannot resume; the simulation records its failure and discards this runtime.
"""

import hashlib
import time
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import cast

from anyio import Lock
from pydantic_ai import Agent, AgentRunResult, ModelMessage, RunUsage, UsageLimits, UserContent
from pydantic_ai.capabilities import AgentCapability
from pydantic_ai.models import KnownModelName, Model
from pydantic_ai.settings import ModelSettings
from pydantic_ai_harness import (
    FallbackCompaction,
    Memory,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.memory import InMemoryStore, MemoryToolset

from mam_bench.benchmark import AgentSettings
from mam_bench.communication import CompletedWork, SharedCommunication, run_rolling
from mam_bench.evidence import EvidenceRecorder, RequestTracker, SummaryTracker, TrackedSummaryModel
from mam_bench.usage import AgentSessionUsage, session_usage_data


@dataclass
class _SessionState:
    history: list[ModelMessage] = field(default_factory=lambda: list[ModelMessage]())
    usage: RunUsage = field(default_factory=RunUsage)
    model_requests: int = 0
    cost_complete: bool = True


class AgentSessionRuntime[AgentDepsT, OutputDataT, SharedPayloadT = object]:
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
        self._failure: BaseException | None = None
        self._usage = RunUsage()
        self._model_requests = 0
        self._cost_complete = True
        self._communication = SharedCommunication[SharedPayloadT]()
        self._evidence = EvidenceRecorder()

    @property
    def settings(self) -> AgentSettings:
        """Return the immutable protocol settings."""
        return self._settings

    @property
    def communication(self) -> SharedCommunication[SharedPayloadT]:
        """Return this trial's opaque shared communication component."""
        return self._communication

    @property
    def evidence(self) -> EvidenceRecorder:
        """Return this trial's normalized event recorder."""
        return self._evidence

    @property
    def usage(self) -> AgentSessionUsage:
        """Return usage aggregated across every session in this trial."""
        return AgentSessionUsage(
            self._usage,
            model_requests=self._model_requests,
            cost_complete=self._cost_complete,
        )

    def session_usage(self, session_id: str) -> AgentSessionUsage:
        """Return usage for one opaque session identity."""
        state = self._state(session_id)
        return AgentSessionUsage(
            state.usage,
            model_requests=state.model_requests,
            cost_complete=state.cost_complete,
        )

    def history(self, session_id: str) -> tuple[ModelMessage, ...]:
        """Return the session's normalized PydanticAI message history."""
        return tuple(self._state(session_id).history)

    def notebook_tools(self, session_id: str) -> MemoryToolset[AgentDepsT]:
        """Return Harness editing tools for one private trial notebook."""
        self._check_active()
        return MemoryToolset(self._memory(session_id))

    async def record_final_notebooks(self) -> None:
        """Append one final logical notebook snapshot for every used session."""
        self._check_active()
        for session_id in sorted(self._states):
            prefix = f"{self._namespace(session_id)}/agent/"
            files = {
                path.removeprefix(prefix): content
                for path, content in self._store.files.items()
                if path.startswith(prefix)
            }
            await self._evidence.record(
                "notebook_snapshot",
                session_id=session_id,
                files=files,
            )

    async def run(
        self,
        session_id: str,
        user_prompt: str | Sequence[UserContent] | None,
        *,
        deps: AgentDepsT,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] = (),
    ) -> AgentRunResult[OutputDataT]:
        """Run one turn and retain successful history, notebook, and usage.

        Calls for the same session serialize so concurrent callers cannot fork
        one history. Any escaping failure ends this trial runtime. Its notebook
        state is discarded with the failed pair; retries need a fresh runtime.
        """
        state = self._state(session_id)
        lock = self._locks.setdefault(session_id, Lock())
        async with lock:
            self._check_active()
            turn_started = time.perf_counter()
            await self._evidence.record("session_turn_started", session_id=session_id)
            model_tracker: RequestTracker[AgentDepsT] = RequestTracker(
                self._evidence,
                session_id,
            )
            summary_tracker = SummaryTracker()
            run_capabilities: tuple[AgentCapability[AgentDepsT], ...] = (
                self._compaction(summary_tracker, session_id),
                self._memory(session_id),
                *capabilities,
                model_tracker,
            )
            try:
                result = await self._agent.run(
                    user_prompt,
                    message_history=state.history,
                    deps=deps,
                    model_settings=model_settings,
                    usage_limits=usage_limits,
                    capabilities=run_capabilities,
                )
                run_usage = result.usage
                summary_requests = run_usage.requests - model_tracker.requests
                if summary_requests != summary_tracker.requests:
                    raise RuntimeError("session usage could not distinguish summary requests")

                state.history = result.all_messages()
                state.usage.incr(run_usage)
                state.model_requests += model_tracker.requests
                if not model_tracker.cost_complete or not summary_tracker.cost_complete:
                    state.cost_complete = False
                    self._cost_complete = False
                self._usage.incr(run_usage)
                self._model_requests += model_tracker.requests
                await self._evidence.record(
                    "session_turn_completed",
                    session_id=session_id,
                    latency_seconds=time.perf_counter() - turn_started,
                    usage=session_usage_data(
                        AgentSessionUsage(
                            run_usage,
                            model_requests=model_tracker.requests,
                            cost_complete=model_tracker.cost_complete
                            and summary_tracker.cost_complete,
                        )
                    ),
                )
                return result
            except BaseException as error:
                if self._failure is None:
                    self._failure = error
                raise

    async def run_rolling[ItemT, ResultT](
        self,
        items: Iterable[ItemT],
        worker: Callable[[ItemT], Awaitable[ResultT]],
    ) -> tuple[CompletedWork[ItemT, ResultT], ...]:
        """Run queued work up to configured concurrency and return completion order."""
        return await run_rolling(
            items,
            worker,
            concurrency=self._settings.concurrency,
        )

    def _state(self, session_id: str) -> _SessionState:
        if not session_id:
            raise ValueError("session_id must not be empty")
        return self._states.setdefault(session_id, _SessionState())

    def _check_active(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _namespace(self, session_id: str) -> str:
        if not session_id:
            raise ValueError("session_id must not be empty")
        return f"session-{hashlib.sha256(session_id.encode()).hexdigest()}"

    def _memory(self, session_id: str) -> Memory[AgentDepsT]:
        return Memory(
            store=self._store,
            namespace=self._namespace(session_id),
            agent_name="agent",
            max_tokens=self._settings.memory_injection_tokens,
            injection_errors="raise",
        )

    def _compaction(
        self,
        tracker: SummaryTracker,
        session_id: str,
    ) -> TieredCompaction[AgentDepsT]:
        settings = self._settings
        model = self._agent.model
        if model is None:
            raise RuntimeError("the Agent Session Runtime requires an Agent with a model")
        summarizer: SummarizingCompaction[AgentDepsT] = SummarizingCompaction(
            model=TrackedSummaryModel(
                cast("Model | KnownModelName", model),
                tracker,
                self._evidence,
                session_id,
            ),
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
