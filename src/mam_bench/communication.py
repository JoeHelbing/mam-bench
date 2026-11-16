"""A world-owned append-only message board with private unread positions."""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime

from anyio import Lock
from pydantic_ai import ModelRetry


@dataclass(frozen=True)
class Message:
    sequence: int
    step: int
    time: str
    author_id: str
    text: str


@dataclass(frozen=True)
class MessagePage:
    content: str
    more_available: bool


class MessageBoard:
    """Publish authored messages and read complete records in chronological pages."""

    def __init__(self) -> None:
        self.step = 0  # The simulation step stamped on new posts.
        self._messages: list[Message] = []
        self._cursors: dict[str, int] = {}
        self._lock = Lock()

    async def post(self, author_id: str, text: str) -> int:
        if not author_id:
            raise ValueError("author_id must not be empty")
        if not text.strip() or len(text) > 4000:
            raise ModelRetry("Messages must contain 1 to 4000 characters of nonempty text.")
        async with self._lock:
            sequence = len(self._messages)
            time = datetime.now(UTC).strftime("%H:%M:%S")
            self._messages.append(Message(sequence, self.step, time, author_id, text))
            return sequence

    def unread(self, session_id: str) -> int:
        """Count other agents' messages this session has not read."""
        cursor = self._cursors.get(session_id, 0)
        return sum(m.author_id != session_id for m in self._messages[cursor:])

    async def read(self, session_id: str) -> MessagePage:
        """Return the next page of other agents' unread messages and mark it read."""
        if not session_id:
            raise ValueError("session_id must not be empty")
        async with self._lock:
            cursor = self._cursors.get(session_id, 0)
            content, taken = self._page(
                m for m in self._messages[cursor:] if m.author_id != session_id
            )
            # Own posts never count as unread, so the cursor skips past them too.
            for message in self._messages[cursor:]:
                if message.author_id != session_id:
                    if not taken:
                        break
                    taken -= 1
                cursor += 1
            self._cursors[session_id] = cursor
            return MessagePage(content, self.unread(session_id) > 0)

    async def search(
        self, *, since_step: int | None = None, author: str | None = None, last: int | None = None
    ) -> MessagePage:
        """Return matching messages, own posts included, without marking anything read."""
        async with self._lock:
            found = [
                m
                for m in self._messages
                if (since_step is None or m.step >= since_step)
                and (author is None or m.author_id == author)
            ]
            if last is not None:
                found = found[-last:] if last > 0 else []
            content, taken = self._page(found)
            return MessagePage(content, taken < len(found))

    @staticmethod
    def _page(messages: Iterable[Message]) -> tuple[str, int]:
        lines: list[str] = []
        length = 0
        for m in messages:
            line = f"[{m.sequence} step {m.step} {m.time} by {m.author_id}] {m.text}"
            if lines and length + 1 + len(line) > 40000:
                break
            lines.append(line)
            length += len(line) + bool(length)
        return "\n".join(lines), len(lines)
