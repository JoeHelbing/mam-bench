"""Ordered shared records and rolling admission of complete actor turns."""

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from anyio import Lock


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


async def run_rolling[ItemT, ResultT](
    items: Iterable[ItemT],
    worker: Callable[[ItemT], Awaitable[ResultT]],
    *,
    concurrency: int,
) -> tuple[CompletedWork[ItemT, ResultT], ...]:
    """Run queued work up to one concurrency limit in completion order."""
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
        while len(active) < concurrency and admit_one():
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
            while len(active) < concurrency and admit_one():
                pass
    finally:
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
    return tuple(completed)


def _validate_identity(value: str, *, name: str) -> None:
    if not value:
        raise ValueError(f"{name} must not be empty")
