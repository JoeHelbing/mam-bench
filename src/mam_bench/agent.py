"""Persistent, model-agnostic PydanticAI sessions for benchmark trials.

The runtime owns only generic agent mechanics: normalized message histories,
private in-memory notebooks, bounded context, and usage. Benchmark simulations
continue to own prompts, tools, outputs, and domain policy.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal, cast

from anyio import Lock
from pydantic_ai import (
    Agent,
    AgentRunResult,
    ModelMessage,
    ModelRetry,
    RunUsage,
    UsageLimits,
    UserContent,
)
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

MEMORY_FILE_CHARS = 65_536
_MEMORY_FILENAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}")


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


@dataclass(frozen=True)
class SharedRecord[PayloadT]:
    """One immutable envelope in the trial's append-only communication stream."""

    sequence: int
    author_session_id: str
    payload: PayloadT


@dataclass(frozen=True)
class SharedPage[PayloadT]:
    """One successful chronological read from a session-specific cursor."""

    records: tuple[SharedRecord[PayloadT], ...]
    content: str
    more_available: bool
    next_cursor: int


class SharedCommunication[PayloadT]:
    """Store opaque ordered records and isolated reader cursors for one trial."""

    def __init__(self) -> None:
        self._records: list[SharedRecord[PayloadT]] = []
        self._cursors: dict[str, int] = {}
        self._lock = Lock()

    @property
    def records(self) -> tuple[SharedRecord[PayloadT], ...]:
        """Return all records in authoritative append order."""
        return tuple(self._records)

    async def append(self, author_session_id: str, payload: PayloadT) -> SharedRecord[PayloadT]:
        """Append one opaque payload and assign its monotonic sequence."""
        _validate_identity(author_session_id, name="author_session_id")
        async with self._lock:
            record = SharedRecord(
                sequence=len(self._records),
                author_session_id=author_session_id,
                payload=payload,
            )
            self._records.append(record)
            return record

    async def read(
        self,
        session_id: str,
        *,
        max_chars: int,
        renderer: Callable[
            [tuple[SharedRecord[PayloadT], ...]],
            str | Awaitable[str],
        ],
    ) -> SharedPage[PayloadT]:
        """Render and advance through the oldest unread complete records.

        Selection, rendering, and cursor advancement share the communication
        lock. A first unread record is returned even when its complete rendering
        exceeds ``max_chars``; records are never split.
        """
        _validate_identity(session_id, name="session_id")
        if max_chars < 1:
            raise ValueError("max_chars must be positive")
        async with self._lock:
            cursor = self._cursors.get(session_id, 0)
            if cursor == len(self._records):
                return SharedPage(records=(), content="", more_available=False, next_cursor=cursor)

            selected: list[SharedRecord[PayloadT]] = []
            content = ""
            for record in self._records[cursor:]:
                candidate = (*selected, record)
                rendered = renderer(candidate)
                candidate_content = await rendered if not isinstance(rendered, str) else rendered
                if selected and len(candidate_content) > max_chars:
                    break
                selected.append(record)
                content = candidate_content

            next_cursor = cursor + len(selected)
            self._cursors[session_id] = next_cursor
            return SharedPage(
                records=tuple(selected),
                content=content,
                more_available=next_cursor < len(self._records),
                next_cursor=next_cursor,
            )


@dataclass(frozen=True)
class CompletedWork[ItemT, ResultT]:
    """One rolling-scheduler result in authoritative completion order."""

    item: ItemT
    result: ResultT
    admission_sequence: int
    completion_sequence: int


@dataclass(frozen=True)
class NotebookMutationResult:
    """Content-free result from one action-bound notebook mutation."""

    file: str
    status: Literal["created", "appended", "updated", "deleted", "not_found"]


class SessionNotebook:
    """Commit action-bound notebook edits to staged and canonical session stores."""

    def __init__(
        self,
        staged_store_resolver: Callable[[], InMemoryStore],
        canonical_store_resolver: Callable[[], InMemoryStore],
        snapshot_recorder: Callable[[str, str | None], None],
        scope: str,
    ) -> None:
        self._staged_store_resolver = staged_store_resolver
        self._canonical_store_resolver = canonical_store_resolver
        self._snapshot_recorder = snapshot_recorder
        self._scope = scope

    async def write(
        self,
        content: str,
        *,
        file: str = "MEMORY.md",
        old_text: str | None = None,
    ) -> NotebookMutationResult:
        """Append content or replace one unique passage in a session notebook file."""
        name = _normalize_memory_filename(file)
        if old_text is None and not content.strip():
            raise ModelRetry("Nothing to write; append non-empty text or provide old_text")
        staged_store = self._staged_store_resolver()
        canonical_store = self._canonical_store_resolver()
        path = f"{self._scope}/{name}"
        staged = await staged_store.read(path, max_chars=MEMORY_FILE_CHARS)
        if staged is not None and staged.truncated:
            raise ModelRetry(f"{name!r} exceeds the notebook edit limit")
        existing = None if staged is None else staged.content
        if old_text is None:
            updated = (
                f"{content.rstrip()}\n"
                if existing is None or not existing.strip()
                else f"{existing.rstrip()}\n{content.rstrip()}\n"
            )
            status: Literal["created", "appended", "updated"] = (
                "created" if existing is None or not existing.strip() else "appended"
            )
        else:
            if existing is None or not old_text:
                raise ModelRetry(f"old_text was not found in {name!r}")
            occurrences = existing.count(old_text)
            if occurrences == 0:
                raise ModelRetry(f"old_text was not found in {name!r}")
            if occurrences > 1:
                raise ModelRetry(f"old_text must occur exactly once in {name!r}")
            updated = existing.replace(old_text, content, 1)
            status = "updated"
        if len(updated) > MEMORY_FILE_CHARS:
            raise ModelRetry(
                f"{name!r} would grow to {len(updated)} characters; "
                f"the limit is {MEMORY_FILE_CHARS}"
            )
        canonical = await canonical_store.read(path, max_chars=MEMORY_FILE_CHARS)
        if canonical is not None and canonical.truncated:
            raise ModelRetry(f"{name!r} exceeds the notebook edit limit")
        self._snapshot_recorder(path, None if canonical is None else canonical.content)
        await staged_store.write(
            path,
            updated,
            expected_version=None if staged is None else staged.version,
        )
        if canonical is None or canonical.content != updated:
            await canonical_store.write(
                path,
                updated,
                expected_version=None if canonical is None else canonical.version,
            )
        return NotebookMutationResult(file=name, status=status)

    async def delete(self, file: str) -> NotebookMutationResult:
        """Delete one permitted session notebook file."""
        name = _normalize_memory_filename(file)
        if name == "MEMORY.md":
            raise ModelRetry("The main notebook cannot be deleted")
        staged_store = self._staged_store_resolver()
        canonical_store = self._canonical_store_resolver()
        path = f"{self._scope}/{name}"
        staged = await staged_store.read(path, max_chars=1)
        canonical = await canonical_store.read(path, max_chars=MEMORY_FILE_CHARS)
        if staged is not None or canonical is not None:
            self._snapshot_recorder(path, None if canonical is None else canonical.content)
        if staged is not None:
            await staged_store.delete(path, expected_version=staged.version)
        if canonical is not None:
            await canonical_store.delete(path, expected_version=canonical.version)
        if staged is None:
            return NotebookMutationResult(file=name, status="not_found")
        return NotebookMutationResult(file=name, status="deleted")


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
        self._active_stores: dict[str, InMemoryStore] = {}
        self._active_notebook_snapshots: dict[str, dict[str, str | None]] = {}
        self._usage = RunUsage()
        self._model_requests = 0
        self._cost_complete = True
        self._communication = SharedCommunication[SharedPayloadT]()

    @property
    def settings(self) -> AgentSettings:
        """Return the immutable protocol settings."""
        return self._settings

    @property
    def communication(self) -> SharedCommunication[SharedPayloadT]:
        """Return this trial's opaque shared communication component."""
        return self._communication

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

    def notebook(self, session_id: str) -> SessionNotebook:
        """Return the action-bound notebook editor for this session's active run."""
        _validate_identity(session_id, name="session_id")
        scope = f"{self._namespace(session_id)}/agent"
        return SessionNotebook(
            lambda: self._active_store(session_id),
            lambda: self._store,
            lambda path, content: self._record_notebook_snapshot(
                session_id,
                path,
                content,
            ),
            scope,
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
        one history. Failures propagate and leave the prior successful history
        and aggregate usage unchanged.
        """
        state = self._state(session_id)
        lock = self._locks.setdefault(session_id, Lock())
        async with lock:
            staged_store = self._staged_memory(session_id)
            model_tracker: _RequestTracker[AgentDepsT] = _RequestTracker()
            summary_tracker = _SummaryTracker()
            run_capabilities: tuple[AgentCapability[AgentDepsT], ...] = (
                self._compaction(summary_tracker),
                self._memory(session_id, staged_store),
                *capabilities,
                model_tracker,
            )
            self._active_stores[session_id] = staged_store
            self._active_notebook_snapshots[session_id] = {}
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
            except BaseException:
                await self._rollback_action_memory(session_id)
                raise
            finally:
                self._active_stores.pop(session_id, None)
                self._active_notebook_snapshots.pop(session_id, None)

    async def run_rolling[ItemT, ResultT](
        self,
        items: Iterable[ItemT],
        worker: Callable[[ItemT], Awaitable[ResultT]],
    ) -> tuple[CompletedWork[ItemT, ResultT], ...]:
        """Run queued work up to configured concurrency and return completion order."""
        iterator = iter(items)
        active: set[asyncio.Task[CompletedWork[ItemT, ResultT]]] = set()
        completed: list[CompletedWork[ItemT, ResultT]] = []
        next_admission = 0
        next_completion = 0

        async def execute(item: ItemT, admission: int) -> CompletedWork[ItemT, ResultT]:
            nonlocal next_completion
            result = await worker(item)
            completion = next_completion
            next_completion += 1
            return CompletedWork(
                item=item,
                result=result,
                admission_sequence=admission,
                completion_sequence=completion,
            )

        def admit_one() -> bool:
            nonlocal next_admission
            try:
                item = next(iterator)
            except StopIteration:
                return False
            active.add(asyncio.create_task(execute(item, next_admission)))
            next_admission += 1
            return True

        try:
            while len(active) < self._settings.concurrency and admit_one():
                pass
            while active:
                done, active = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                try:
                    finished = [task.result() for task in done]
                except BaseException:
                    active.update(done)
                    raise
                finished.sort(key=lambda item: item.completion_sequence)
                completed.extend(finished)
                while len(active) < self._settings.concurrency and admit_one():
                    pass
        finally:
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
        return tuple(completed)

    def _state(self, session_id: str) -> _SessionState:
        _validate_identity(session_id, name="session_id")
        return self._states.setdefault(session_id, _SessionState())

    def _active_store(self, session_id: str) -> InMemoryStore:
        try:
            return self._active_stores[session_id]
        except KeyError:
            raise RuntimeError("session notebook edits require an active Agent run") from None

    def _record_notebook_snapshot(
        self,
        session_id: str,
        path: str,
        content: str | None,
    ) -> None:
        try:
            snapshots = self._active_notebook_snapshots[session_id]
        except KeyError:
            raise RuntimeError("session notebook edits require an active Agent run") from None
        if path not in snapshots:
            snapshots[path] = content

    async def _rollback_action_memory(self, session_id: str) -> None:
        snapshots = self._active_notebook_snapshots[session_id]
        for path, original in reversed(snapshots.items()):
            current = await self._store.read(path, max_chars=MEMORY_FILE_CHARS)
            if original is None:
                if current is not None:
                    await self._store.delete(path, expected_version=current.version)
                continue
            if current is not None and not current.truncated and current.content == original:
                continue
            await self._store.write(
                path,
                original,
                expected_version=None if current is None else current.version,
            )

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


def _validate_identity(value: str, *, name: str) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty")


def _normalize_memory_filename(file: str) -> str:
    name = file.strip()
    if name and not name.endswith(".md"):
        name = f"{name}.md"
    if not _MEMORY_FILENAME.fullmatch(name) or ".." in name:
        raise ModelRetry("Use a short notebook filename with no slashes or parent traversal")
    return name


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
