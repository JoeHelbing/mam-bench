"""PydanticAI implementation of Schelling Influence Actor interactions.

The runtime runs persistent, locally embodied actor turns with Schelling-owned
Public Document tools and terminal outputs over the package-level session
runtime. Simulation mechanics stay in ``runtime.py``.
"""

import asyncio
import json
import re
from dataclasses import dataclass, field, replace
from typing import Any, Never

import numpy as np
from anyio import Lock
from openai import APIConnectionError, APIError, APITimeoutError
from pydantic import ValidationError
from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput, UsageLimits
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.capabilities.abstract import (
    AgentNode,
    NodeResult,
    RawOutput,
    RawToolArgs,
    ValidatedToolArgs,
    WrapOutputProcessHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.exceptions import (
    ModelAPIError,
    ToolRetryError,
)
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.output import OutputContext
from pydantic_ai.result import FinalResult
from pydantic_ai.tools import ToolDefinition
from pydantic_ai_harness.memory import MemoryConflictError, MemoryOperationConflictError
from pydantic_graph import End

from mam_bench.agent import AgentSessionRuntime
from mam_bench.benchmark import AgentInfrastructureFailure, ModelRuntime
from mam_bench.communication import SharedCommunication, SharedRecord
from mam_bench.evidence import EvidenceRecorder
from mam_bench.usage import AgentSessionUsage

from .models import (
    ActorMemoryMutation,
    ActorNotebook,
    ActorTerminalAction,
    ActorTurnContext,
    ActorTurnCoordinator,
    ActorTurnDependencies,
    ActorTurnResult,
    AppendMemory,
    AuthoritativeRecord,
    DocumentRead,
    PostReceipt,
    PublicDocumentRecord,
    ReplaceMemory,
    Stay,
    SubmitMove,
    UnverifiedPost,
)
from .prompt import ACTOR_TURN_INSTRUCTIONS, actor_turn_prompt
from .reference import EMPTY_CELL
from .runtime import model_sampling_seed

ACTOR_TURN_PHASE_CODE = 2
PUBLIC_DOCUMENT_PAGE_CHARS = 40_000
PUBLIC_POST_CHARS = 4_000
MEMORY_TOOL_NAMES = frozenset({"write_memory", "read_memory", "search_memory", "delete_memory"})
PUBLIC_RUNTIME_RECORD_CHARS = 4_000
PUBLIC_RUNTIME_SOURCE_CHARS = 100


def _infrastructure_failure(error: BaseException) -> AgentInfrastructureFailure:
    if isinstance(error, AgentInfrastructureFailure):
        return error
    if isinstance(error, TimeoutError | APITimeoutError):
        kind = "timeout"
    elif isinstance(error, APIConnectionError):
        kind = "network"
    elif isinstance(error, ModelAPIError | APIError):
        kind = "provider"
    else:
        kind = "memory_store"
    return AgentInfrastructureFailure(kind, _error_text(error))


def _actor_turn_agent(
    runtime: ModelRuntime,
) -> Agent[ActorTurnDependencies, ActorTerminalAction]:
    agent: Agent[ActorTurnDependencies, ActorTerminalAction] = Agent(
        runtime.model,
        deps_type=ActorTurnDependencies,
        output_type=[
            ToolOutput(
                SubmitMove,
                name="submit_move",
                description="End this turn by requesting one destination row and column.",
                strict=True,
                max_retries=2,
            ),
            ToolOutput(
                Stay,
                name="stay",
                description="End this turn without moving.",
                strict=True,
                max_retries=2,
            ),
        ],
        instructions=ACTOR_TURN_INSTRUCTIONS,
        retries=2,
    )

    @agent.tool(name="read_document", retries=2, strict=True)
    async def _read_document(  # pyright: ignore[reportUnusedFunction]
        ctx: RunContext[ActorTurnDependencies],
    ) -> DocumentRead:
        """Read the oldest unread complete Public Document records."""
        return await ctx.deps.coordinator.read_document(ctx.deps.context.actor_id)

    @agent.tool(name="post_message", retries=2, strict=True)
    async def _post_message(  # pyright: ignore[reportUnusedFunction]
        ctx: RunContext[ActorTurnDependencies],
        text: str,
        memory: ActorMemoryMutation | None = None,
    ) -> PostReceipt:
        """Append one free-form, explicitly unverified Public Document post."""
        return await ctx.deps.coordinator.post_message(
            ctx.deps.context.actor_id,
            text,
            memory,
            _ActorNotebook(ctx),
        )

    return agent


@dataclass(frozen=True)
class _ActorNotebook:
    """Bind Harness editing to the current public action's operation identity."""

    ctx: RunContext[ActorTurnDependencies]

    async def write(
        self,
        content: str,
        *,
        file: str = "MEMORY.md",
        old_text: str | None = None,
    ) -> object:
        return await self.ctx.deps.notebook.write_memory(
            self.ctx, content, file=file, old_text=old_text
        )

    async def delete(self, file: str) -> object:
        return await self.ctx.deps.notebook.delete_memory(self.ctx, file)


class SchellingTurnCoordinator:
    """Interpret and render Schelling records over generic shared storage."""

    def __init__(
        self,
        communication: SharedCommunication[PublicDocumentRecord],
        evidence: EvidenceRecorder | None = None,
    ) -> None:
        self._communication = communication
        self._evidence = evidence
        self._lock = Lock()
        self._round_number: int | None = None
        self._round_vacancies: frozenset[int] | None = None
        self._reservations: dict[int, int] = {}

    async def start_round(self, round_number: int, cell_types: np.ndarray) -> None:
        """Reset reservation state from one frozen beginning-of-round board."""
        async with self._lock:
            self._round_number = round_number
            self._round_vacancies = frozenset(
                int(location) for location in np.flatnonzero(cell_types.ravel() == EMPTY_CELL)
            )
            self._reservations = {}
            if self._evidence is not None:
                await self._evidence.record(
                    "round_started",
                    round_number=round_number,
                    vacancy_count=len(self._round_vacancies),
                )

    async def remaining_unreserved_vacancies(self) -> int:
        """Return the current count of unclaimed beginning-round vacancies."""
        async with self._lock:
            if self._round_vacancies is None:
                raise RuntimeError("no Staged Round is active")
            return len(self._round_vacancies) - len(self._reservations)

    async def reserved_actor_moves(self) -> tuple[tuple[int, int], ...]:
        """Return actor reservations in coordinator commit order."""
        async with self._lock:
            return tuple(self._reservations.items())

    async def read_document(self, actor_id: int) -> DocumentRead:
        async with self._lock:
            page = await self._communication.read(
                str(actor_id),
                max_chars=PUBLIC_DOCUMENT_PAGE_CHARS,
                renderer=_render_public_document,
            )
            if self._evidence is not None:
                cursor_before = page.records[0].sequence if page.records else page.next_cursor
                await self._evidence.record(
                    "document_read",
                    actor_id=actor_id,
                    cursor_before=cursor_before,
                    cursor_after=page.next_cursor,
                    record_sequences=[record.sequence for record in page.records],
                    more_available=page.more_available,
                )
        return DocumentRead(content=page.content, more_available=page.more_available)

    async def post_message(
        self,
        actor_id: int,
        text: str,
        memory: ActorMemoryMutation | None = None,
        notebook: ActorNotebook | None = None,
    ) -> PostReceipt:
        if len(text) > PUBLIC_POST_CHARS:
            raise ValueError("Public Document posts must be at most 4,000 characters")
        async with self._lock:
            await _apply_memory_mutation(memory, notebook)
            record = await self._communication.append(str(actor_id), UnverifiedPost(text))
            if self._evidence is not None:
                if memory is not None:
                    await self._evidence.record(
                        "memory_operation",
                        actor_id=actor_id,
                        operation=memory.operation,
                        file=memory.file,
                        attached_to="post_message",
                    )
                await self._evidence.record(
                    "publication",
                    actor_id=actor_id,
                    document_sequence=record.sequence,
                    record_kind="unverified_post",
                    text=text,
                )
        return PostReceipt(sequence=record.sequence)

    async def commit_terminal(
        self,
        actor_id: int,
        action: ActorTerminalAction,
        notebook: ActorNotebook,
    ) -> ActorTerminalAction:
        """Commit one terminal action and any attached memory mutation."""
        async with self._lock:
            destination: int | None = None
            if isinstance(action, SubmitMove) and self._round_vacancies is not None:
                destination = action.row * 20 + action.column
                if destination not in self._round_vacancies:
                    raise ModelRetry("destination was not vacant at the beginning of the round")
                if destination in self._reservations.values():
                    raise ModelRetry("destination is already reserved")
            await _apply_memory_mutation(action.memory, notebook)
            if self._evidence is not None and action.memory is not None:
                await self._evidence.record(
                    "memory_operation",
                    actor_id=actor_id,
                    operation=action.memory.operation,
                    file=action.memory.file,
                    attached_to="submit_move" if destination is not None else "stay",
                )
            if destination is not None:
                self._reservations[actor_id] = destination
                row, column = divmod(destination, 20)
                record = await self._communication.append(
                    "schelling-runtime",
                    AuthoritativeRecord(
                        f"Round {self._round_number}: Influence Actor {actor_id} "
                        f"reserved ({row},{column})."
                    ),
                )
                if self._evidence is not None:
                    await self._evidence.record(
                        "reservation",
                        round_number=self._round_number,
                        actor_id=actor_id,
                        row=row,
                        column=column,
                        document_sequence=record.sequence,
                    )
            elif self._evidence is not None:
                await self._evidence.record(
                    "stay",
                    round_number=self._round_number,
                    actor_id=actor_id,
                )
        return action

    async def publish_authoritative(self, source: str, text: str) -> int:
        """Append one bounded runtime-authored fact and return its sequence."""
        if not source or len(source) > PUBLIC_RUNTIME_SOURCE_CHARS:
            raise ValueError("runtime record source must contain 1 to 100 characters")
        if len(text) > PUBLIC_RUNTIME_RECORD_CHARS:
            raise ValueError("runtime record text must be at most 4,000 characters")
        async with self._lock:
            record = await self._communication.append(source, AuthoritativeRecord(text))
            if self._evidence is not None:
                await self._evidence.record(
                    "publication",
                    source=source,
                    document_sequence=record.sequence,
                    record_kind="authoritative_runtime_record",
                    text=text,
                )
        return record.sequence


async def _apply_memory_mutation(
    mutation: ActorMemoryMutation | None,
    notebook: ActorNotebook | None,
) -> None:
    if mutation is None:
        return
    if notebook is None:
        raise ValueError("an attached memory mutation requires an active actor notebook")
    if isinstance(mutation, AppendMemory):
        await notebook.write(mutation.content, file=mutation.file)
    elif isinstance(mutation, ReplaceMemory):
        await notebook.write(
            mutation.content,
            file=mutation.file,
            old_text=mutation.old_text,
        )
    else:
        await notebook.delete(mutation.file)


def _render_public_document(
    records: tuple[SharedRecord[PublicDocumentRecord], ...],
) -> str:
    lines: list[str] = []
    for record in records:
        if isinstance(record.payload, UnverifiedPost):
            rendered = {
                "sequence": record.sequence,
                "kind": "unverified_post",
                "author_session_id": record.author_session_id,
                "text": record.payload.text,
            }
        else:
            rendered = {
                "sequence": record.sequence,
                "kind": "authoritative_runtime_record",
                "source": record.author_session_id,
                "text": record.payload.text,
            }
        lines.append(json.dumps(rendered, ensure_ascii=False, separators=(",", ":")))
    return "\n".join(lines)


class _ForcedStay(Exception):
    """Private control flow used when an actor exhausts its policy retries."""


@dataclass
class _TurnPolicy(AbstractCapability[ActorTurnDependencies]):
    """Enforce one Schelling Actor Turn before domain side effects occur."""

    evidence: EvidenceRecorder
    actor_id: int
    round_number: int
    accepted_calls: int = 0
    rejections: list[str] = field(default_factory=list[str])
    memory_operations: int = 0
    forced_stay: bool = False
    _pending_remaining: int | None = None

    async def prepare_tools(
        self,
        ctx: RunContext[ActorTurnDependencies],
        tool_defs: list[ToolDefinition],
    ) -> list[ToolDefinition]:
        del ctx
        return [] if self.accepted_calls >= 9 else tool_defs

    async def before_model_request(
        self,
        ctx: RunContext[ActorTurnDependencies],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        del ctx
        if self._pending_remaining is None:
            return request_context
        latest = request_context.messages[-1]
        if not isinstance(latest, ModelRequest):
            raise RuntimeError("model request history must end with a ModelRequest")
        reminder = UserPromptPart(
            f"{self._pending_remaining} successful calls remain in this Actor Turn."
        )
        request_context.messages[-1] = replace(latest, parts=[*latest.parts, reminder])
        self._pending_remaining = None
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[ActorTurnDependencies],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        del ctx
        calls = [part for part in response.parts if isinstance(part, ToolCallPart)]
        if len(calls) != 1:
            await self._reject(f"response contained {len(calls)} calls; exactly one is required")
        call = calls[0]
        parameters = request_context.model_request_parameters
        allowed = {tool.name for tool in (*parameters.function_tools, *parameters.output_tools)}
        if call.tool_name not in allowed:
            await self._reject(f"unknown or unavailable tool {call.tool_name!r}")
        return response

    async def after_tool_validate(
        self,
        ctx: RunContext[ActorTurnDependencies],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
    ) -> ValidatedToolArgs:
        del ctx, tool_def
        if call.tool_name == "post_message":
            text = args.get("text")
            if isinstance(text, str) and len(text) > PUBLIC_POST_CHARS:
                await self._reject("post_message text must be at most 4,000 characters")
        return args

    async def on_tool_validate_error(
        self,
        ctx: RunContext[ActorTurnDependencies],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: RawToolArgs,
        error: ValidationError | ModelRetry,
    ) -> ValidatedToolArgs:
        del ctx, tool_def, args
        await self._reject(f"malformed {call.tool_name} call: {_error_text(error)}")

    async def wrap_tool_execute(
        self,
        ctx: RunContext[ActorTurnDependencies],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> Any:  # noqa: ANN401 - PydanticAI's capability handler is dynamically typed
        del ctx, tool_def
        try:
            result = await handler(args)
        except ToolRetryError as error:
            await self._reject(_error_text(error))
        if call.tool_name in MEMORY_TOOL_NAMES or args.get("memory") is not None:
            self.memory_operations += 1
        self._accept_nonterminal()
        return result

    async def on_output_validate_error(
        self,
        ctx: RunContext[ActorTurnDependencies],
        *,
        output_context: OutputContext,
        output: RawOutput,
        error: ValidationError | ModelRetry,
    ) -> Any:  # noqa: ANN401 - PydanticAI's capability hook is dynamically typed
        del ctx, output_context, output
        await self._reject(f"malformed terminal action: {_error_text(error)}")

    async def wrap_output_process(
        self,
        ctx: RunContext[ActorTurnDependencies],
        *,
        output_context: OutputContext,
        output: Any,  # noqa: ANN401 - PydanticAI's capability handler is dynamically typed
        handler: WrapOutputProcessHandler,
    ) -> Any:  # noqa: ANN401 - PydanticAI's capability handler is dynamically typed
        del output_context
        result = await handler(output)
        if ctx.partial_output:
            return result
        if not isinstance(result, SubmitMove | Stay):
            raise RuntimeError("Schelling output processing returned an unknown action")
        try:
            committed = await ctx.deps.coordinator.commit_terminal(
                ctx.deps.context.actor_id,
                result,
                _ActorNotebook(ctx),
            )
        except (ModelRetry, ToolRetryError) as error:
            await self._reject(_error_text(error))
        if result.memory is not None:
            self.memory_operations += 1
        self.accepted_calls += 1
        return committed

    async def on_node_run_error(
        self,
        ctx: RunContext[ActorTurnDependencies],
        *,
        node: AgentNode[ActorTurnDependencies],
        error: Exception,
    ) -> NodeResult[ActorTurnDependencies]:
        del ctx, node
        if not isinstance(error, _ForcedStay):
            raise error
        return End(FinalResult(Stay(), tool_name="stay"))

    def _accept_nonterminal(self) -> None:
        self.accepted_calls += 1
        if self.accepted_calls >= 10:
            raise RuntimeError("a tenth successful nonterminal call must not be possible")
        self._pending_remaining = 10 - self.accepted_calls

    async def _reject(self, message: str) -> Never:
        self.rejections.append(message[:500])
        await self.evidence.record(
            "policy_retry",
            round_number=self.round_number,
            actor_id=self.actor_id,
            rejection_number=len(self.rejections),
        )
        if len(self.rejections) >= 3:
            self.forced_stay = True
            raise _ForcedStay
        raise ModelRetry(
            f"Policy rejection {len(self.rejections)} of 2: {message[:400]}. "
            "Return exactly one valid available tool call."
        )


class RuntimeInfluenceTeam:
    """Run the Schelling interaction protocol with PydanticAI agents."""

    def __init__(self, runtime: ModelRuntime) -> None:
        self.runtime = runtime
        self.runtime_info = runtime.info
        self._actor_turn_agent = _actor_turn_agent(runtime)
        self._actor_sessions: AgentSessionRuntime[
            ActorTurnDependencies,
            ActorTerminalAction,
            PublicDocumentRecord,
        ] = AgentSessionRuntime(self._actor_turn_agent, settings=runtime.info.agent_settings)

    @property
    def communication(self) -> SharedCommunication[PublicDocumentRecord]:
        """Return the v2 trial's generic Public Document storage."""
        return self._actor_sessions.communication

    @property
    def usage(self) -> AgentSessionUsage:
        """Return aggregate usage across the trial's persistent actor sessions."""
        return self._actor_sessions.usage

    @property
    def evidence(self) -> EvidenceRecorder:
        """Return the v2 trial's shared evidence recorder."""
        return self._actor_sessions.evidence

    async def record_final_notebooks(self) -> None:
        """Append final snapshots for all actor notebooks used in the trial."""
        await self._actor_sessions.record_final_notebooks()

    async def run_turn(
        self,
        context: ActorTurnContext,
        coordinator: ActorTurnCoordinator,
    ) -> ActorTurnResult:
        """Run one persistent, locally embodied v2 Influence Actor turn."""
        session_id = str(context.actor_id)
        policy = _TurnPolicy(
            evidence=self.evidence,
            actor_id=context.actor_id,
            round_number=context.round_number,
        )
        try:
            notebook = self._actor_sessions.notebook_tools(session_id)
            async with asyncio.timeout(self.runtime.info.agent_settings.timeout_seconds):
                result = await self._actor_sessions.run(
                    session_id,
                    actor_turn_prompt(context),
                    deps=ActorTurnDependencies(context, coordinator, notebook),
                    model_settings={
                        "seed": model_sampling_seed(
                            context.config,
                            round_number=context.round_number,
                            phase_code=ACTOR_TURN_PHASE_CODE,
                            actor_id=context.actor_id,
                        )
                    },
                    usage_limits=UsageLimits(request_limit=32, tool_calls_limit=15),
                    capabilities=(policy,),
                )
        except (
            AgentInfrastructureFailure,
            ModelAPIError,
            APIError,
            MemoryConflictError,
            MemoryOperationConflictError,
            OSError,
        ) as error:
            raise _infrastructure_failure(error) from None
        return ActorTurnResult(
            actor_id=context.actor_id,
            action=result.output,
            accepted_call_count=policy.accepted_calls,
            policy_rejections=tuple(policy.rejections),
            memory_operation_count=policy.memory_operations,
            forced_stay=policy.forced_stay,
        )


def _error_text(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"[:500]
    text = re.sub(r"(?i)Bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return re.sub(
        r"(?i)(api[_-]?key|authorization|password|token)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )
