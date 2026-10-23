"""Private agent sessions, workspaces, and bounded individual turns."""

import asyncio
import json
import logging
import re
from collections.abc import AsyncIterable, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime
from time import monotonic
from types import TracebackType
from typing import cast

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
from pydantic_ai.agent import ModelRequestNode
from pydantic_ai.capabilities import AbstractCapability, AgentCapability
from pydantic_ai.capabilities.abstract import AgentNode, NodeResult
from pydantic_ai.exceptions import (
    IncompleteToolCall,
    ModelAPIError,
    UnexpectedModelBehavior,
    UsageLimitExceeded,
)
from pydantic_ai.messages import (
    AgentStreamEvent,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.settings import ModelSettings
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.workspaces import Workspace, WorkspaceError
from pydantic_ai_harness import (
    Coder,
    FallbackCompaction,
    SlidingWindowCompaction,
    SummarizingCompaction,
    TieredCompaction,
    modal_sandbox,
)

from mam_bench.communication import MessageBoard, MessagePage
from mam_bench.config import AgentSettings
from mam_bench.diagnostics import ExecutionFailure, failure_trace
from mam_bench.sandboxes import new_workspace

logger = logging.getLogger(__name__)


async def stream_responses(
    _ctx: RunContext[object], _events: AsyncIterable[AgentStreamEvent]
) -> None:
    """Enable native streaming without adding a second recorder."""


class _CallErrorLimitReached(Exception):
    def __init__(self, *, provider: bool) -> None:
        # Provider requests caused most of the failures, rather than the model's calls.
        self.provider = provider
        super().__init__("provider" if provider else "model")


class _CallErrors[DepsT](AbstractCapability[DepsT]):
    """Share one cumulative failure budget across tools and provider requests."""

    def __init__(self, history: Sequence[ModelMessage]) -> None:
        self._seen = self._tool_failures(history)
        self._count = 0
        self._provider_count = 0
        self._requests_before = 0

    def _limit(self) -> _CallErrorLimitReached:
        return _CallErrorLimitReached(provider=2 * self._provider_count > self._count)

    @staticmethod
    def _tool_failures(messages: Sequence[ModelMessage]) -> set[tuple[datetime | None, str]]:
        return {
            (message.timestamp, part.tool_call_id)
            for message in messages
            if isinstance(message, ModelRequest)
            for part in message.parts
            if (
                isinstance(part, RetryPromptPart)
                and part.tool_name is not None
                or isinstance(part, ToolReturnPart)
                and part.outcome == "failed"
            )
        }

    async def before_model_request(
        self, ctx: RunContext[DepsT], request_context: ModelRequestContext
    ) -> ModelRequestContext:
        # Settled tool batches retain all replies and accepted effects.
        failures = self._tool_failures(request_context.messages)
        self._count += len(failures - self._seen)
        self._seen.update(failures)
        if self._count >= 3:
            raise self._limit()
        self._requests_before = ctx.usage.requests
        return request_context

    async def on_node_run_error(
        self, ctx: RunContext[DepsT], *, node: AgentNode[DepsT], error: Exception
    ) -> NodeResult[DepsT]:
        # The node boundary also catches errors while consuming an open stream.
        if not isinstance(node, ModelRequestNode) or not isinstance(
            error, (APIError, ModelAPIError)
        ):
            raise error
        detail = error
        while isinstance(detail.__cause__, (APIError, ModelAPIError)):
            detail = detail.__cause__
        status = getattr(detail, "status_code", None)
        body = getattr(detail, "body", None)
        if status is None and isinstance(body, dict):
            status = cast(dict[str, object], body).get("code")
        if isinstance(status, int) and 400 <= status < 500 and status not in (408, 409, 429):
            raise error
        # Native streaming increments only after the stream opens; failed opens count too.
        if ctx.usage.requests == self._requests_before:
            ctx.usage.requests += 1
        self._count += 1
        self._provider_count += 1
        logger.warning(
            "request.failed session=%s failures=%d exception=%s status=%s error=%s body=%r",
            ctx.conversation_id,
            self._count,
            type(error).__name__,
            status if isinstance(status, int) else None,
            error,
            cast(object, body),
        )
        parts: list[ToolReturnPart | RetryPromptPart] = []
        if ctx.messages and isinstance(last := ctx.messages[-1], ModelResponse):
            # Partial streamed tool calls were never executed. Pair them before retrying.
            parts.extend(
                ToolReturnPart(
                    tool_name=part.tool_name,
                    tool_call_id=part.tool_call_id,
                    content="Provider request failed; this tool call was not executed.",
                )
                for part in last.parts
                if isinstance(part, ToolCallPart)
            )
        parts.append(RetryPromptPart(content="Provider request failed. Try again."))
        request = ModelRequest(parts=parts)
        if self._count >= 3:
            ctx.messages.append(request)
            raise self._limit() from error
        return ModelRequestNode(request)


class _Compaction[DepsT](FallbackCompaction[DepsT]):
    """Keep compaction failures distinct from recoverable agent tool retries."""

    async def compact(
        self, messages: list[ModelMessage], ctx: RunContext[DepsT]
    ) -> list[ModelMessage]:
        started = monotonic()
        logger.debug("compaction.start messages=%d", len(messages))
        try:
            result = await super().compact(messages, ctx)
        except UsageLimitExceeded:
            # Summarization shares the parent turn's budget and reserves its pending request.
            raise
        except Exception as error:
            logger.error("compaction.failed trace=%s", failure_trace(error))
            raise ExecutionFailure("compaction", "compaction failed") from error
        logger.debug(
            "compaction.end messages=%d elapsed_s=%.3f", len(result), monotonic() - started
        )
        return result


@dataclass(frozen=True)
class Completed[T]:
    value: T


@dataclass(frozen=True)
class Stopped:
    reason: str


@dataclass
class _SessionState:
    history: list[ModelMessage] = field(default_factory=lambda: list[ModelMessage]())
    lock: Lock = field(default_factory=Lock)
    workspace: modal_sandbox.ModalSandboxBackend | None = None
    preparation: asyncio.Task[None] | None = None


class AgentSessions[AgentDepsT, OutputDataT]:
    """Share one model and message board across private histories and sandboxes."""

    INSTRUCTIONS = """
Use read_messages and post_message to coordinate. Call observe() for your local
observation. In Python, use from observe import observe; observe() reads the same
snapshot from observation.json. Your private /workspace files persist across
your turns in this case. You may save notes, code, and results for later turns.
Posts are unverified; simulation announcements are marked separately.
Respond with tool calls, not plain text. Finish code and messaging before
submitting your action. A valid terminal action ends the turn; other calls in
the same response are skipped. Invalid actions return retry feedback without
reserving any part of the action.
Each turn allows up to 25 model requests. Three failed tool or transient API calls
end the turn; successful calls do not reset this failure count.
""".strip()

    def __init__(
        self,
        agent: Agent[AgentDepsT, OutputDataT],
        *,
        context_window_tokens: int,
        settings: AgentSettings | None = None,
    ) -> None:
        self._agent = agent
        self._settings = settings or AgentSettings()
        self._context_window_tokens = context_window_tokens
        self._states: dict[str, _SessionState] = {}
        self._closed = False
        self._failure: BaseException | None = None
        self._usage = RunUsage()
        self.board = MessageBoard()

    async def __aenter__(self) -> AgentSessions[AgentDepsT, OutputDataT]:
        self._check_active()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        cleanup = asyncio.create_task(self.aclose())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
        except Exception:
            if exc is None:
                raise
            logger.exception("sandbox.cleanup_failed while unwinding failed case")

    async def aclose(self) -> None:
        """Finish provisioning, then destroy every sandbox owned by this case."""
        self._closed = True

        async def close(state: _SessionState) -> None:
            if state.preparation is not None:
                # Creation is shielded from turn cancellation, so retain its eventual ref.
                await asyncio.gather(state.preparation, return_exceptions=True)
            if state.workspace is not None and (ref := state.workspace.ref) is not None:
                await modal_sandbox.ModalSandbox().destroy(ref)
                state.workspace = None

        results = await asyncio.gather(
            *(close(s) for s in self._states.values()), return_exceptions=True
        )
        failures = [r for r in results if isinstance(r, BaseException)]
        if failures:
            raise ExecutionFailure(
                "sandbox_cleanup", "could not terminate all case sandboxes"
            ) from failures[0]

    async def _prepare_workspace(self, session_id: str, state: _SessionState) -> None:
        assert state.workspace is not None
        await state.workspace.get_sandbox()
        logger.info("sandbox.ready session=%s ref=%s", session_id, state.workspace.ref)
        await Workspace(state.workspace).write_text(
            "observe.py",
            "import json\nfrom pathlib import Path\n\n"
            "def observe():\n"
            '    return json.loads(Path(__file__).with_name("observation.json").read_text())\n',
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

    def _board_tools(
        self, session_id: str, observation: dict[str, object] | None
    ) -> FunctionToolset[AgentDepsT]:
        tools: FunctionToolset[AgentDepsT] = FunctionToolset()

        async def read_messages() -> MessagePage:
            """Read the next page of world messages to coordinate with other model agents."""
            return await self.board.read(session_id)

        async def post_message(text: str) -> int:
            """Post up to 4000 characters for other model agents to read from the shared board."""
            return await self.board.post(session_id, text)

        tools.add_function(
            read_messages,
            takes_ctx=False,
            strict=self._settings.strict_parameterless_tools or None,
        )
        tools.add_function(post_message, takes_ctx=False)
        if observation is not None:

            def observe() -> dict[str, object]:
                """Read your local observation; in Python use from observe import observe."""
                return observation

            tools.add_function(
                observe, takes_ctx=False, strict=self._settings.strict_parameterless_tools or None
            )
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
        observation: dict[str, object] | None = None,
    ) -> Completed[OutputDataT] | Stopped:
        """Run a bounded turn; native limit/retry stops preserve the session.

        The simulation decides its fallback action for a stopped turn.
        Execution failures stop all sessions in this world.
        """
        if self._settings.max_completion_tokens is not None:
            model_settings = {
                "max_tokens": self._settings.max_completion_tokens,
                **(model_settings or {}),
            }
        state = self._state(session_id)
        async with state.lock:
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
                    if state.preparation is None:
                        state.workspace = new_workspace(self._settings.sandbox_timeout_seconds)
                        state.preparation = asyncio.create_task(
                            self._prepare_workspace(session_id, state)
                        )
                    await asyncio.shield(state.preparation)
                    assert state.workspace is not None
                    if observation is not None:
                        await Workspace(state.workspace).write_text(
                            "observation.json", json.dumps(observation)
                        )
                    async with deadline:
                        result = await self._agent.run(
                            user_prompt,
                            conversation_id=session_id,
                            message_history=state.history,
                            instructions=self.INSTRUCTIONS,
                            deps=deps,
                            workspace=state.workspace,
                            model_settings=model_settings,
                            usage=usage,
                            usage_limits=usage_limits or UsageLimits(request_limit=25),
                            capabilities=(
                                _CallErrors[AgentDepsT](state.history),
                                self._compaction(),
                                Coder(sub_agents=False, repo_context=False),
                                *capabilities,
                            ),
                            toolsets=[self._board_tools(session_id, observation)],
                        )
                except _CallErrorLimitReached as limit:
                    outcome = "provider_error_limit" if limit.provider else "call_error_limit"
                    logger.info("turn.stop session=%s reason=%s", session_id, outcome)
                    turn: Completed[OutputDataT] | Stopped = Stopped(outcome)
                except UsageLimitExceeded:
                    outcome = "usage_limit"
                    logger.info("turn.stop session=%s reason=usage_limit", session_id)
                    turn = Stopped(outcome)

                except IncompleteToolCall:
                    outcome = "output_token_limit"
                    turn = Stopped(outcome)
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
                        raise self._failure from error
                    logger.info("turn.stop session=%s reason=%s", session_id, outcome)
                    turn = Stopped(outcome)
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
                    logger.error(
                        "provider.failed session=%s error=%s body=%r",
                        session_id,
                        error,
                        getattr(error, "body", None),
                    )
                    self._failure = ExecutionFailure("provider", "model request failed")
                    raise self._failure from error
                except WorkspaceError as error:
                    self._failure = ExecutionFailure("sandbox", "agent workspace failed")
                    raise self._failure from error
                except BaseException as error:
                    self._failure = error
                    raise
                else:
                    turn = Completed(result.output)
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
                    self._usage.incr(usage)
                state.history = list(messages)
                return turn

    def _state(self, session_id: str) -> _SessionState:
        if not session_id:
            raise ValueError("session_id must not be empty")
        return self._states.setdefault(session_id, _SessionState())

    def _check_active(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise RuntimeError("agent sessions are closed")

    def _compaction(self) -> TieredCompaction[AgentDepsT]:
        settings = self._settings
        model = self._agent.model
        if model is None:
            raise ExecutionFailure("compaction", "Session runtime requires an Agent with a model")
        summarizer: SummarizingCompaction[AgentDepsT] = SummarizingCompaction(
            model=model,
            max_tokens=1,
            keep_tokens=settings.compaction_tail(self._context_window_tokens),
            incremental=True,
            preserve_first_user_message=False,
            receipts=True,
            model_settings={"max_tokens": settings.summary_completion_tokens}
            if settings.summary_completion_tokens is not None
            else None,
        )
        sliding: SlidingWindowCompaction[AgentDepsT] = SlidingWindowCompaction(
            max_tokens=1,
            keep_tokens=settings.compaction_tail(self._context_window_tokens),
            preserve_first_user_message=False,
            receipts=True,
        )
        return TieredCompaction(
            tiers=[_Compaction[AgentDepsT]([summarizer, sliding])],
            target_fraction=settings.compaction_trigger_fraction,
            context_window=self._context_window_tokens,
        )
