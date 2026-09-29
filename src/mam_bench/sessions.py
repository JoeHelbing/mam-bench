"""Private agent sessions, memory, archives, and bounded individual turns."""

import asyncio
import hashlib
import logging
import re
from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import monotonic

from anyio import Lock
from openai import APIConnectionError, APIError, APITimeoutError
from pydantic_ai import (
    Agent,
    ModelMessage,
    RunContext,
    RunUsage,
    UsageLimits,
    UserContent,
    capture_run_messages,
)
from pydantic_ai.capabilities import AbstractCapability, AgentCapability
from pydantic_ai.capabilities.abstract import (
    ValidatedToolArgs,
    WrapModelRequestHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.exceptions import (
    IncompleteToolCall,
    ModelAPIError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai_harness import (
    FallbackCompaction,
    Memory,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.memory import (
    InMemoryStore,
    MemoryConflictError,
    MemoryOperationConflictError,
)
from pydantic_ai_harness.step_persistence import StepPersistence

from mam_bench.artifacts import AgentMessageArchive
from mam_bench.communication import MessageBoard, MessagePage
from mam_bench.config import AgentSettings
from mam_bench.diagnostics import ExecutionFailure, failure_trace

logger = logging.getLogger(__name__)


class _RequestLogging[DepsT](AbstractCapability[DepsT]):
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id

    async def wrap_model_request(
        self,
        ctx: RunContext[DepsT],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        started = monotonic()
        logger.debug(
            "request.start session=%s request=%d messages=%d",
            self.session_id,
            ctx.usage.requests,
            len(request_context.messages),
        )
        try:
            response = await handler(request_context)
        except BaseException as error:
            logger.debug(
                "request.failed session=%s elapsed_s=%.3f trace=%s",
                self.session_id,
                monotonic() - started,
                failure_trace(error),
            )
            raise
        logger.debug(
            "request.end session=%s elapsed_s=%.3f input_tokens=%d output_tokens=%d "
            "finish=%s tools=%s",
            self.session_id,
            monotonic() - started,
            response.usage.input_tokens,
            response.usage.output_tokens,
            response.finish_reason,
            [part.tool_name for part in response.parts if isinstance(part, ToolCallPart)],
        )
        return response

    async def wrap_tool_execute(
        self,
        ctx: RunContext[DepsT],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: WrapToolExecuteHandler,
    ) -> object:
        started = monotonic()
        logger.debug("tool.start session=%s tool=%s", self.session_id, call.tool_name)
        try:
            result = await handler(args)
        except BaseException as error:
            logger.debug(
                "tool.failed session=%s tool=%s exception=%s elapsed_s=%.3f",
                self.session_id,
                call.tool_name,
                type(error).__name__,
                monotonic() - started,
            )
            raise
        logger.debug(
            "tool.end session=%s tool=%s elapsed_s=%.3f",
            self.session_id,
            call.tool_name,
            monotonic() - started,
        )
        return result


class _Compaction[DepsT](FallbackCompaction[DepsT]):
    """Keep compaction failures distinct from recoverable agent tool retries."""

    async def compact(
        self, messages: list[ModelMessage], ctx: RunContext[DepsT]
    ) -> list[ModelMessage]:
        started = monotonic()
        logger.debug("compaction.start messages=%d", len(messages))
        try:
            result = await super().compact(messages, ctx)
        except Exception as error:
            logger.error("compaction.failed trace=%s", failure_trace(error))
            raise ExecutionFailure("compaction", "compaction failed") from error
        logger.debug(
            "compaction.end messages=%d elapsed_s=%.3f", len(result), monotonic() - started
        )
        return result


@dataclass
class _SessionState:
    history: list[ModelMessage] = field(default_factory=lambda: list[ModelMessage]())
    usage: RunUsage = field(default_factory=RunUsage)
    lock: Lock = field(default_factory=Lock)
    stop_reason: str | None = None


class AgentSessions[AgentDepsT, OutputDataT]:
    """Share one model and message board across isolated histories and notebooks."""

    def __init__(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
        *,
        settings: AgentSettings | None = None,
        artifact_directory: Path | None = None,
    ) -> None:
        self._agent = agent
        self._settings = settings or AgentSettings()
        self._store = InMemoryStore()
        self._states: dict[str, _SessionState] = {}
        self._failure: BaseException | None = None
        self._usage = RunUsage()
        self._archive = None
        if artifact_directory is not None:
            artifact_directory.mkdir(parents=True, exist_ok=True)
            self._archive = AgentMessageArchive(
                artifact_directory / "agent-messages", media_store=None
            )
        self.board = MessageBoard(
            archive_path=artifact_directory / "message-board.jsonl" if artifact_directory else None
        )

    @property
    def settings(self) -> AgentSettings:
        return self._settings

    @property
    def usage(self) -> RunUsage:
        """Return a snapshot of native usage across all sessions."""
        return deepcopy(self._usage)

    def history(self, session_id: str) -> tuple[ModelMessage, ...]:
        return tuple(self._state(session_id).history)

    def stop_reason(self, session_id: str) -> str | None:
        return self._state(session_id).stop_reason

    def _board_tools(self, session_id: str) -> FunctionToolset[AgentDepsT]:
        tools: FunctionToolset[AgentDepsT] = FunctionToolset()

        async def read_messages() -> MessagePage:
            """Read the next page of unread shared messages and simulation announcements."""
            return await self.board.read(session_id)

        async def post_message(text: str) -> int:
            """Post up to 4000 characters to the shared message board."""
            return await self.board.post(session_id, text)

        tools.add_function(read_messages, takes_ctx=False)
        tools.add_function(post_message, takes_ctx=False)
        return tools

    async def run(
        self,
        session_id: str,
        user_prompt: str | Sequence[UserContent] | None,
        *,
        deps: AgentDepsT,
        model_settings: ModelSettings | None = None,
        usage_limits: UsageLimits | None = None,
        capabilities: Sequence[AgentCapability[AgentDepsT]] = (),
    ) -> OutputDataT | None:
        """Run a bounded turn; native limit/retry stops preserve the session.

        None means no terminal output was accepted. The simulation decides its
        fallback action. Execution failures stop all sessions in this world.
        """
        if not self._settings.use_sampling_seed and model_settings is not None:
            model_settings = model_settings.copy()
            model_settings.pop("seed", None)
        state = self._state(session_id)
        async with state.lock:
            state.stop_reason = None
            self._check_active()
            usage = RunUsage()
            started = monotonic()
            outcome = "completed"
            deadline = asyncio.timeout(self._settings.timeout_seconds)
            logger.debug(
                "turn.start session=%s timeout_s=%s history_messages=%d",
                session_id,
                self._settings.timeout_seconds,
                len(state.history),
            )
            with capture_run_messages() as messages:
                try:
                    async with deadline:
                        result = await self._agent.run(
                            user_prompt,
                            conversation_id=session_id,
                            message_history=state.history,
                            deps=deps,
                            model_settings=model_settings,
                            usage=usage,
                            usage_limits=usage_limits
                            or UsageLimits(request_limit=25, tool_calls_limit=25),
                            capabilities=(
                                *(
                                    (StepPersistence(store=self._archive, agent_name=session_id),)
                                    if self._archive is not None
                                    else ()
                                ),
                                _RequestLogging[AgentDepsT](session_id),
                                self._compaction(),
                                self._memory(session_id),
                                *capabilities,
                            ),
                            toolsets=[self._board_tools(session_id)],
                        )
                except UsageLimitExceeded:
                    outcome = "usage_limit"
                    logger.info("turn.stop session=%s reason=usage_limit", session_id)
                    output = None

                except IncompleteToolCall:
                    outcome = "output_token_limit"
                    output = None
                except UnexpectedModelBehavior as error:
                    # These native PydanticAI errors have no dedicated subtype.
                    # Only exhausted turn budgets are recoverable; other failures abort.
                    if re.fullmatch(
                        r"Model token limit \([^)]+\) exceeded before any response "
                        r"was generated\..*",
                        error.message,
                    ):
                        outcome = "output_token_limit"
                    elif re.fullmatch(
                        r"Exceeded maximum output retries \(\d+\)|"
                        r"Tool .+ exceeded max retries count of \d+\..*",
                        error.message,
                    ):
                        outcome = "retry_exhaustion"
                    else:
                        self._failure = ExecutionFailure(
                            "provider", "model returned an unusable response"
                        )
                        raise self._failure from None
                    logger.info("turn.stop session=%s reason=%s", session_id, outcome)
                    output = None
                except (APITimeoutError, TimeoutError) as error:
                    scope = (
                        "turn_deadline"
                        if deadline.expired()
                        else (
                            "http_request"
                            if isinstance(error, APITimeoutError)
                            else "internal_operation"
                        )
                    )
                    logger.error(
                        "turn.timeout session=%s scope=%s timeout_s=%s elapsed_s=%.3f",
                        session_id,
                        scope,
                        self._settings.timeout_seconds,
                        monotonic() - started,
                    )
                    self._failure = ExecutionFailure("timeout", f"{scope} timed out")
                    raise self._failure from error
                except APIConnectionError as error:
                    self._failure = ExecutionFailure("network", "model connection failed")
                    raise self._failure from error
                except (APIError, ModelAPIError) as error:
                    self._failure = ExecutionFailure("provider", "model request failed")
                    raise self._failure from error
                except (MemoryConflictError, MemoryOperationConflictError, OSError) as error:
                    self._failure = ExecutionFailure("memory_store", "notebook failed")
                    raise self._failure from error
                except BaseException as error:
                    self._failure = error
                    raise
                else:
                    output = result.output
                finally:
                    if self._failure is not None:
                        outcome = "failed"
                        logger.error(
                            "turn.failed session=%s trace=%s",
                            session_id,
                            failure_trace(self._failure),
                        )
                    logger.debug(
                        "turn.end session=%s outcome=%s elapsed_s=%.3f requests=%d tools=%d "
                        "input_tokens=%d output_tokens=%d",
                        session_id,
                        outcome,
                        monotonic() - started,
                        usage.requests,
                        usage.tool_calls,
                        usage.input_tokens,
                        usage.output_tokens,
                    )
                if output is None:
                    state.stop_reason = outcome
                    self._close_stopped_turn(messages)
                state.history = list(messages)
                state.usage.incr(usage)
                self._usage.incr(usage)
                return output

    @staticmethod
    def _close_stopped_turn(messages: list[ModelMessage]) -> None:
        """Pair pending calls before submitting this history in the next round."""
        answered: set[str] = set()
        for message in reversed(messages):
            if isinstance(message, ModelRequest):
                answered.update(
                    part.tool_call_id
                    for part in message.parts
                    if isinstance(part, (ToolReturnPart, RetryPromptPart))
                )
            else:
                missing = [
                    ToolReturnPart(
                        tool_name=part.tool_name,
                        tool_call_id=part.tool_call_id,
                        content=(
                            "Turn ended at its usage or retry limit; no tool result was returned."
                        ),
                    )
                    for part in message.parts
                    if isinstance(part, ToolCallPart) and part.tool_call_id not in answered
                ]
                if missing:
                    messages.append(ModelRequest(parts=missing))
                break

    def _state(self, session_id: str) -> _SessionState:
        if not session_id:
            raise ValueError("session_id must not be empty")
        return self._states.setdefault(session_id, _SessionState())

    def _check_active(self) -> None:
        if self._failure is not None:
            raise self._failure

    def _memory(self, session_id: str) -> Memory[AgentDepsT]:
        self._state(session_id)
        return Memory(
            store=self._store,
            namespace=f"session-{hashlib.sha256(session_id.encode()).hexdigest()}",
            agent_name="agent",
            max_tokens=self._settings.memory_injection_tokens,
            injection_errors="raise",
        )

    def _compaction(self) -> TieredCompaction[AgentDepsT]:
        settings = self._settings
        model = self._agent.model
        if model is None:
            raise ExecutionFailure("compaction", "Session runtime requires an Agent with a model")
        summarizer: SummarizingCompaction[AgentDepsT] = SummarizingCompaction(
            model=model,
            max_tokens=1,
            keep_tokens=settings.compaction_tail_tokens,
            incremental=True,
            preserve_first_user_message=False,
            receipts=True,
            model_settings={"max_tokens": settings.summary_completion_tokens},
        )
        sliding: SlidingWindowCompaction[AgentDepsT] = SlidingWindowCompaction(
            max_tokens=1,
            keep_tokens=settings.compaction_tail_tokens,
            preserve_first_user_message=False,
            receipts=True,
        )
        fallback = _Compaction[AgentDepsT]([summarizer, sliding])
        return TieredCompaction(
            tiers=[fallback],
            target_fraction=settings.compaction_trigger_fraction,
            context_window=settings.context_window_tokens,
        )
