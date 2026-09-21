"""A world-owned append-only message board with private unread positions."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from anyio import Lock
from pydantic_ai import ModelRetry

from mam_bench.diagnostics import ExecutionFailure


@dataclass(frozen=True)
class Message:
    sequence: int
    author_id: str
    text: str
    announcement: bool = False


@dataclass(frozen=True)
class MessagePage:
    content: str
    more_available: bool


class MessageBoard:
    """Publish authored messages and read complete records in chronological pages."""

    def __init__(self, *, archive_path: Path | None = None) -> None:
        self._messages: list[Message] = []
        self._cursors: dict[str, int] = {}
        self._lock = Lock()
        self._archive_path = archive_path
        if archive_path is not None:
            archive_path.touch(exist_ok=False)

    async def post(self, author_id: str, text: str, *, announcement: bool = False) -> int:
        if not author_id:
            raise ValueError("author_id must not be empty")
        if not text.strip() or len(text) > 4000:
            raise ModelRetry("Messages must contain 1 to 4000 characters of nonempty text.")
        async with self._lock:
            sequence = len(self._messages)
            message = Message(sequence, author_id, text, announcement)
            if self._archive_path is not None:
                try:
                    with self._archive_path.open("a", encoding="utf-8") as archive:
                        archive.write(json.dumps(asdict(message)) + "\n")
                except OSError as error:
                    raise ExecutionFailure(
                        "artifact_write", "message board write failed"
                    ) from error
            self._messages.append(message)
            return sequence

    async def read(self, session_id: str) -> MessagePage:
        if not session_id:
            raise ValueError("session_id must not be empty")
        async with self._lock:
            cursor = self._cursors.get(session_id, 0)
            lines: list[str] = []
            length = 0
            for message in self._messages[cursor:]:
                label = "announcement" if message.announcement else "message"
                line = f"[{message.sequence} {label} by {message.author_id}] {message.text}"
                if lines and length + 1 + len(line) > 40000:
                    break
                lines.append(line)
                length += len(line) + bool(length)
                cursor += 1
            self._cursors[session_id] = cursor
            return MessagePage("\n".join(lines), cursor < len(self._messages))
