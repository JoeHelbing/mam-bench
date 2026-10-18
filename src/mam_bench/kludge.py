"""Collect temporary model/runtime workarounds and their removal conditions.

Keep normal provider construction in runtime.py. Changes here should identify
the broken upstream behavior, retain its evidence, and say when to remove them.
No simulation rules or output-schema relaxation belongs here.

Bolt also needs a serving-side ultra_v3 parser correction: pinned vLLM renders
reasoning_effort='none' as enable_thinking=False without updating the request
seen by NVIDIA's parser. That Python 3.12 plugin lives with the isolated PGX
recipe, outside this Python 3.14 package. Its original and regression probe are
under .scratch/model-runs/pgx-nemotron35-bolt-20261015/. Remove that correction
when NVIDIA's parser honors reasoning_effort or receives resolved render kwargs.

Qwen's strict, zero-argument XML tool grammar can emit unlimited whitespace
(vLLM #50989, XGrammar #802). The Python 3.12 serving plugin and matcher
regression live in serving/qwen_empty_tool_kludge.py and its adjacent test.
It replaces only closed empty argument sections, preserving native schemas,
tool extraction and request settings. Remove when the pinned serving image
passes that matcher regression without the plugin (upstream XGrammar #837).
"""

from collections.abc import AsyncGenerator, AsyncIterator
from contextlib import asynccontextmanager
from contextvars import ContextVar
from datetime import UTC, datetime
from typing import Literal, cast

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic_ai import RunContext
from pydantic_ai.messages import FinishReason, ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters, StreamedResponse
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIStreamedResponse,
)
from pydantic_ai.settings import ModelSettings

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import VLLMModel
from mam_bench.diagnostics import ExecutionFailure

# Task-local: concurrent sessions share a model, but never a capture directory.
vllm_stream_writer: ContextVar[ArtifactWriter | None] = ContextVar(
    "vllm_stream_writer", default=None
)


class _VLLMStreamedResponse(OpenAIStreamedResponse):
    capture_writer: ArtifactWriter | None = None
    capture_context: RunContext[object] | None = None

    async def _validate_response(self) -> AsyncIterator[ChatCompletionChunk]:
        async for chunk in super()._validate_response():
            if self.capture_writer is not None and self.capture_context is not None:
                ctx = self.capture_context
                try:
                    self.capture_writer.append(
                        f"{ctx.run_id}.jsonl",
                        {
                            "timestamp": datetime.now(UTC).isoformat(),
                            "conversation_id": ctx.conversation_id,
                            "run_step": ctx.run_step,
                            "chunk": chunk.model_dump(mode="json", exclude_unset=True),
                        },
                    )
                except OSError as error:
                    raise ExecutionFailure(
                        "artifact_write", "provider stream write failed"
                    ) from error
            yield chunk

    def _map_finish_reason(
        self, key: Literal["stop", "length", "tool_calls", "content_filter", "function_call"]
    ) -> FinishReason | None:
        return "length" if str(key) == "repetition" else super()._map_finish_reason(key)

    def _map_provider_details(self, chunk: ChatCompletionChunk) -> dict[str, str]:
        details = super()._map_provider_details(chunk) or {}
        if details.get("finish_reason") == "repetition":
            details["stop_reason"] = "repetition_detected"
        return details


class VLLMChatModel(OpenAIChatModel):
    """Accept vLLM's repetition stop while preserving its real cause.

    The native OpenAI adapter rejects finish_reason='repetition'. Translate it
    to its existing incomplete-output path, then restore the vendor cause in
    provider_details. Keep strict validation for every other finish reason.
    Evidence: docs/research-issue-log.md, R12; original fix 1716471.
    Remove when the native adapter accepts and preserves vLLM repetition stops.

    Archived runs also retain SDK chunks before native event mapping, including
    token IDs emitted while the server parser buffers text. Remove this capture
    hook when native events preserve those provider fields themselves.
    """

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        return _VLLMStreamedResponse

    @asynccontextmanager
    async def request_stream(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
        run_context: RunContext[object] | None = None,
    ) -> AsyncGenerator[StreamedResponse]:
        writer = vllm_stream_writer.get()
        if writer is not None:
            # Merge model defaults as well as per-call settings (e.g. summary caps).
            settings = {**(self.settings or {}), **(model_settings or {})}
            settings["extra_body"] = {
                **cast(dict[str, object], settings.get("extra_body") or {}),
                "return_token_ids": True,
                "include_reasoning": True,
            }
            model_settings = cast(ModelSettings, settings)
        async with super().request_stream(
            messages, model_settings, model_request_parameters, run_context
        ) as response:
            captured = cast(_VLLMStreamedResponse, response)
            captured.capture_writer = writer
            captured.capture_context = run_context
            yield response

    def _process_response(self, response: ChatCompletion | str) -> ModelResponse:
        repetition = isinstance(response, ChatCompletion) and any(
            str(choice.finish_reason) == "repetition" for choice in response.choices
        )
        if isinstance(response, ChatCompletion) and repetition:
            response = response.model_copy(
                update={
                    "choices": [
                        choice.model_copy(update={"finish_reason": "length"})
                        if str(choice.finish_reason) == "repetition"
                        else choice
                        for choice in response.choices
                    ]
                }
            )
        result = super()._process_response(response)
        if repetition:
            result.provider_details = {
                **(result.provider_details or {}),
                "finish_reason": "repetition",
                "stop_reason": "repetition_detected",
            }
        return result


def vllm_settings(selection: VLLMModel, settings: ModelSettings) -> OpenAIChatModelSettings:
    """Forward explicit thinking settings that native profile inference drops.

    Qwen3.8-27B: preserve the chat-template switch/effort from c5e1b26 on
    codex/qwen38-thinking. Native vLLM profiles deliberately exclude this
    family's different effort ladder. Off sends enable_thinking=False and no
    effort; on sends the requested effort inside chat_template_kwargs.
    Remove this case when the native provider implements that exact mapping.

    Bolt/new model names: PydanticAI drops unified 'thinking' unless its profile
    recognizes support. Forward an explicitly configured vLLM effort through
    the native OpenAI setting instead. The Bolt AgentSessions regression and
    PGX trial reproduce the omission. Remove when native profiles support the
    selected models without dropping their configured efforts.
    """
    resolved = OpenAIChatModelSettings(**settings)
    effort = selection.settings.reasoning_effort
    if effort is None:
        return resolved
    resolved.pop("thinking", None)
    if selection.model.rsplit("/", 1)[-1].lower().startswith("qwen3.8-27b"):
        chat_settings: dict[str, object] = {"enable_thinking": effort != "none"}
        if effort != "none":
            chat_settings["reasoning_effort"] = effort
        resolved["extra_body"] = {
            **cast(dict[str, object], resolved.get("extra_body") or {}),
            "chat_template_kwargs": chat_settings,
        }
    else:
        resolved["openai_reasoning_effort"] = effort
    return resolved
