"""Canonical benchmark evidence and model-request observation.

Messages are normalized and redacted before hashing. One recorder sequences
agent and domain events across all concurrent sessions in a trial.
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal, cast

from anyio import Lock
from pydantic_ai import ModelMessage
from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
from pydantic_ai.capabilities.abstract import ValidatedToolArgs, WrapToolExecuteHandler
from pydantic_ai.exceptions import FallbackExceptionGroup, ModelAPIError
from pydantic_ai.messages import (
    CompactionPart,
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
from pydantic_ai.models import KnownModelName, Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext, ToolDefinition

from mam_bench.benchmark import AgentInfrastructureFailure
from mam_bench.usage import request_usage_data

_MEMORY_TOOL_NAMES = frozenset({"write_memory", "read_memory", "search_memory", "delete_memory"})
type JSONValue = None | bool | int | float | str | list[JSONValue] | dict[str, JSONValue]


@dataclass(frozen=True)
class EvidenceEvent:
    """One normalized event in a trial-wide monotonic stream."""

    sequence: int
    timestamp: str
    kind: str
    data: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-ready event envelope."""
        return {
            "sequence": self.sequence,
            "timestamp": self.timestamp,
            "kind": self.kind,
            **self.data,
        }


class EvidenceRecorder:
    """Serialize concurrent benchmark evidence through one append-only lock."""

    def __init__(self) -> None:
        self._events: list[EvidenceEvent] = []
        self._message_ids: set[str] = set()
        self._compaction_message_ids: set[tuple[str, str]] = set()
        self._lock = Lock()

    @property
    def events(self) -> tuple[EvidenceEvent, ...]:
        """Return the current event stream in authoritative sequence order."""
        return tuple(self._events)

    async def record(self, kind: str, **data: object) -> EvidenceEvent:
        """Append one JSON-safe event and assign its monotonic sequence."""
        async with self._lock:
            event = EvidenceEvent(
                sequence=len(self._events),
                timestamp=datetime.now(UTC).isoformat(),
                kind=kind,
                data=cast("dict[str, object]", _json_value(data)),
            )
            self._events.append(event)
            return event

    async def record_request_start(
        self,
        kind: Literal["model_request_started", "summary_request_started"],
        **data: object,
    ) -> tuple[EvidenceEvent, str]:
        """Append a request start with an ID derived from its event sequence."""
        async with self._lock:
            sequence = len(self._events)
            request_id = f"request-{sequence}"
            event = EvidenceEvent(
                sequence=sequence,
                timestamp=datetime.now(UTC).isoformat(),
                kind=kind,
                data={
                    "request_id": request_id,
                    **cast("dict[str, object]", _json_value(data)),
                },
            )
            self._events.append(event)
            return event, request_id

    async def record_message(self, message: ModelMessage) -> str:
        """Store one allowlisted normalized message once and return its stable ID."""
        normalized = normalize_model_message(message)
        canonical = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
        message_id = f"message-{hashlib.sha256(canonical.encode()).hexdigest()}"
        async with self._lock:
            if message_id not in self._message_ids:
                self._message_ids.add(message_id)
                self._events.append(
                    EvidenceEvent(
                        sequence=len(self._events),
                        timestamp=datetime.now(UTC).isoformat(),
                        kind="message",
                        data={"message_id": message_id, "message": normalized},
                    )
                )
        return message_id

    async def record_context_message(self, session_id: str, message: ModelMessage) -> str:
        """Store a message and emit one typed reference for compaction evidence."""
        message_id = await self.record_message(message)
        event_kind = _compaction_event_kind(message)
        if event_kind is None:
            return message_id
        compaction_id = (session_id, message_id)
        async with self._lock:
            if compaction_id not in self._compaction_message_ids:
                self._compaction_message_ids.add(compaction_id)
                self._events.append(
                    EvidenceEvent(
                        sequence=len(self._events),
                        timestamp=datetime.now(UTC).isoformat(),
                        kind=event_kind,
                        data={"session_id": session_id, "message_id": message_id},
                    )
                )
        return message_id


def normalize_model_message(message: ModelMessage) -> dict[str, object]:
    """Return the explicit portable subset of one PydanticAI message."""
    if isinstance(message, ModelRequest):
        return {
            "kind": "request",
            "timestamp": message.timestamp.isoformat() if message.timestamp is not None else None,
            "instructions": message.instructions,
            "parts": [_normalize_request_part(part) for part in message.parts],
        }
    return {
        "kind": "response",
        "timestamp": message.timestamp.isoformat(),
        "model_name": message.model_name,
        "finish_reason": message.finish_reason,
        "parts": [_normalize_response_part(part) for part in message.parts],
        "usage": request_usage_data(message.usage),
    }


def _normalize_request_part(part: object) -> dict[str, object]:
    if isinstance(part, SystemPromptPart):
        return {"kind": part.part_kind, "content": part.content}
    if isinstance(part, UserPromptPart):
        content = (
            part.content
            if isinstance(part.content, str)
            else [
                item if isinstance(item, str) else item.content
                for item in part.content
                if isinstance(item, str | TextContent)
            ]
        )
        return {"kind": part.part_kind, "content": content}
    if isinstance(part, ToolReturnPart):
        return {
            "kind": part.part_kind,
            "tool_name": part.tool_name,
            "tool_call_id": part.tool_call_id,
            "content": _json_value(part.content),
        }
    if isinstance(part, RetryPromptPart):
        content: object
        if isinstance(part.content, str):
            content = part.content
        else:
            content = [
                {
                    "type": error.get("type"),
                    "location": list(error.get("loc", ())),
                    "message": error.get("msg"),
                }
                for error in part.content
            ]
        return {
            "kind": part.part_kind,
            "tool_name": part.tool_name,
            "tool_call_id": part.tool_call_id,
            "content": content,
        }
    raise TypeError(f"unsupported PydanticAI request part: {type(part).__name__}")


def _normalize_response_part(part: object) -> dict[str, object]:
    if isinstance(part, TextPart | ThinkingPart | CompactionPart):
        return {"kind": part.part_kind, "content": part.content}
    if isinstance(part, ToolCallPart):
        try:
            args = part.args_as_dict(raise_if_invalid=True)
        except AssertionError, ValueError:
            args = {"invalid_json": True}
        return {
            "kind": part.part_kind,
            "tool_name": part.tool_name,
            "tool_call_id": part.tool_call_id,
            "args": _json_value(args),
        }
    raise TypeError(f"unsupported PydanticAI response part: {type(part).__name__}")


def _compaction_event_kind(message: ModelMessage) -> str | None:
    if isinstance(message, ModelResponse):
        return (
            "compaction"
            if any(isinstance(part, CompactionPart) for part in message.parts)
            else None
        )
    for part in message.parts:
        if isinstance(part, SystemPromptPart) and part.content.startswith(
            "Summary of previous conversation:"
        ):
            return "compaction"
        if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
            continue
        receipts = [
            item.content
            for item in part.content
            if isinstance(item, TextContent) and item.content.startswith("[History")
        ]
        if receipts:
            return (
                "compaction_fallback"
                if any("dropped" in item for item in receipts)
                else "compaction_receipt"
            )
    return None


def _json_value(value: object) -> object:
    """Convert common model payloads to JSON values while redacting secret-shaped keys."""
    from pydantic_core import to_jsonable_python

    converted = cast("JSONValue", to_jsonable_python(value))
    return _redact_mapping_keys(converted)


def _redact_mapping_keys(value: JSONValue) -> JSONValue:
    if isinstance(value, dict):
        redacted: dict[str, JSONValue] = {}
        for key, item in value.items():
            normalized_key = str(key).lower().replace("-", "_")
            if normalized_key in {
                "access_token",
                "api_key",
                "authorization",
                "credential",
                "credentials",
                "password",
                "private_key",
                "refresh_token",
                "secret",
                "token",
            } or normalized_key.endswith(("_api_key", "_password", "_secret")):
                redacted[str(key)] = "[REDACTED]"
            else:
                redacted[str(key)] = _redact_mapping_keys(item)
        return redacted
    if isinstance(value, list | tuple):
        return [_redact_mapping_keys(item) for item in value]
    return value


@dataclass
class RequestTracker[AgentDepsT](AbstractCapability[AgentDepsT]):
    """Track evaluated-model responses after PydanticAI resolves their costs."""

    recorder: EvidenceRecorder
    session_id: str
    responses: list[ModelResponse] = field(default_factory=lambda: list[ModelResponse]())
    _pending: list[tuple[str, float]] = field(default_factory=list[tuple[str, float]])

    def get_ordering(self) -> CapabilityOrdering:
        """Observe prepared history and record responses before policy can reject them."""
        return CapabilityOrdering(position="innermost")

    @property
    def requests(self) -> int:
        return len(self.responses)

    @property
    def cost_complete(self) -> bool:
        return all(response.usage.cost is not None for response in self.responses)

    async def before_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        del ctx
        for message in request_context.messages:
            await self.recorder.record_context_message(self.session_id, message)
        request = request_context.messages[-1]
        if not isinstance(request, ModelRequest):
            raise RuntimeError("model request history must end with a ModelRequest")
        message_id = await self.recorder.record_message(request)
        started = time.perf_counter()
        _, request_id = await self.recorder.record_request_start(
            "model_request_started",
            session_id=self.session_id,
            message_id=message_id,
        )
        self._pending.append((request_id, started))
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        del ctx, request_context
        request_id, started = self._pending.pop(0)
        message_id = await self.recorder.record_context_message(self.session_id, response)
        await self.recorder.record(
            "model_request_completed",
            session_id=self.session_id,
            request_id=request_id,
            message_id=message_id,
            latency_seconds=time.perf_counter() - started,
            usage=request_usage_data(response.usage),
        )
        self.responses.append(response)
        return response

    async def wrap_tool_execute(
        self,
        ctx: RunContext[AgentDepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        del ctx, tool_def
        result = await handler(args)
        await self.recorder.record(
            "memory_operation" if call.tool_name in _MEMORY_TOOL_NAMES else "tool_completed",
            session_id=self.session_id,
            tool_name=call.tool_name,
            tool_call_id=call.tool_call_id,
        )
        return result


@dataclass
class SummaryTracker:
    responses: list[ModelResponse] = field(default_factory=lambda: list[ModelResponse]())
    failed: bool = False

    @property
    def requests(self) -> int:
        return len(self.responses)

    @property
    def cost_complete(self) -> bool:
        return not self.failed and all(
            response.usage.cost is not None for response in self.responses
        )

    def record(self, response: ModelResponse) -> None:
        self.responses.append(response)

    def record_failure(self) -> None:
        self.failed = True


class TrackedSummaryModel(WrapperModel):
    """Delegate to the evaluated model while observing summary response costs."""

    def __init__(
        self,
        wrapped: Model | KnownModelName,
        tracker: SummaryTracker,
        recorder: EvidenceRecorder,
        session_id: str,
    ) -> None:
        super().__init__(wrapped)
        self._tracker = tracker
        self._recorder = recorder
        self._session_id = session_id

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        for message in messages:
            await self._recorder.record_context_message(self._session_id, message)
        request = messages[-1]
        if not isinstance(request, ModelRequest):
            raise RuntimeError("summary request history must end with a ModelRequest")
        message_id = await self._recorder.record_message(request)
        started = time.perf_counter()
        _, request_id = await self._recorder.record_request_start(
            "summary_request_started",
            session_id=self._session_id,
            message_id=message_id,
        )
        try:
            response = await self.wrapped.request(
                messages,
                model_settings,
                model_request_parameters,
            )
        except ModelAPIError, FallbackExceptionGroup:
            self._tracker.record_failure()
            await self._recorder.record(
                "summary_request_failed",
                session_id=self._session_id,
                request_id=request_id,
                latency_seconds=time.perf_counter() - started,
            )
            raise
        except Exception as error:
            self._tracker.record_failure()
            await self._recorder.record(
                "summary_request_failed",
                session_id=self._session_id,
                request_id=request_id,
                latency_seconds=time.perf_counter() - started,
            )
            raise AgentInfrastructureFailure("compaction", str(error)) from error
        response_id = await self._recorder.record_context_message(
            self._session_id,
            response,
        )
        await self._recorder.record(
            "summary_request_completed",
            session_id=self._session_id,
            request_id=request_id,
            message_id=response_id,
            latency_seconds=time.perf_counter() - started,
            usage=request_usage_data(response.usage),
        )
        self._tracker.record(response)
        return response
