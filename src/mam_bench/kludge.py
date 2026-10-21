"""Collect temporary model/runtime workarounds and their removal conditions.

Keep normal provider construction in runtime.py. Changes here should identify
the broken upstream behavior, retain its evidence, and say when to remove them.
No simulation rules or output-schema relaxation belongs here.

Only MAM-Bench client workarounds belong here. Server-side grammar and parser
corrections are documented in docs/vllm-serving-workarounds.md for operators
to apply in their own vLLM environment.
"""

from typing import Literal, cast

from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic_ai._json_schema import JsonSchema
from pydantic_ai.messages import FinishReason, ModelResponse
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIStreamedResponse,
)
from pydantic_ai.models.openrouter import OpenRouterReasoning
from pydantic_ai.profiles.openai import OpenAIJsonSchemaTransformer
from pydantic_ai.settings import ModelSettings

from mam_bench.config import ModelSelection


def modal_filesystem_error_workaround() -> None:
    """Avoid masking Modal failures with an absent SDK exception name.

    Harness 0.55 references FilesystemExecutionError, absent from stable Modal
    1.6.1 and 1.6.2. The live sandbox startup probe and the session startup-error
    test reproduce the resulting AttributeError. The nonexistent subclass cannot
    occur with these SDKs; aliasing its parent preserves their error mapping.
    Remove when Harness stops requiring the name or Modal supplies it.
    """
    from modal import exception

    if not hasattr(exception, "FilesystemExecutionError"):
        exception.FilesystemExecutionError = (  # pyright: ignore[reportAttributeAccessIssue]
            exception.ExecutionError
        )


class OpenAIFixedArrayTransformer(OpenAIJsonSchemaTransformer):
    """Encode fixed homogeneous tuples using OpenAI's accepted array subset.

    Pydantic's prefixItems is valid JSON Schema, but OpenAI rejects the move
    coordinate tool with 'array schema missing items'. Equal positional schemas
    and an exact length have an equivalent items encoding. Keep heterogeneous
    tuples and all other schemas native; validation and simulation types stay
    unchanged. Evidence: large-four-model-20261016/luna6-addition/diagnostic-http
    and schema-verification.json under .scratch/model-runs/.
    Remove when the native transformer or endpoint supports this tuple form.
    """

    def transform(self, schema: JsonSchema) -> JsonSchema:
        value = schema.get("prefixItems")
        prefix = cast(list[object], value) if isinstance(value, list) else []
        if (
            schema.get("type") == "array"
            and prefix
            and schema.get("minItems") == schema.get("maxItems") == len(prefix)
            and all(item == prefix[0] for item in prefix)
            and "items" not in schema
        ):
            schema["items"] = prefix[0]
            schema.pop("prefixItems")
        return super().transform(schema)


class _VLLMStreamedResponse(OpenAIStreamedResponse):
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
    """

    @property
    def _streamed_response_cls(self) -> type[OpenAIStreamedResponse]:
        return _VLLMStreamedResponse

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


def openrouter_reasoning(effort: str) -> OpenRouterReasoning:
    """Send the configured OpenRouter effort instead of PydanticAI's lossy map.

    PydanticAI 2.55 maps unified 'xhigh' to OpenRouter 'high' and has no 'max',
    though OpenRouter accepts both. DeepSeek V4.1 Flash on DeepInfra advertises
    max/high/low/none; six probes each on 2026-10-21 averaged 15.5k reasoning
    tokens at max versus 10.3k at high. Remove when the native map forwards
    every OpenRouter effort unchanged.
    """
    return cast(OpenRouterReasoning, {"effort": effort})


def vllm_settings(selection: ModelSelection, settings: ModelSettings) -> OpenAIChatModelSettings:
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
