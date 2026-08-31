"""PydanticAI/OpenAI implementation of the provider-neutral Model Runtime."""

import os
import re
import time
from typing import Any, Literal, Self, cast

from openai import APIError
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator
from pydantic_ai.exceptions import ModelAPIError, UnexpectedModelBehavior
from pydantic_ai.messages import (
    ModelMessage,
    SystemPromptPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.messages import (
    ModelRequest as PydanticModelRequest,
)
from pydantic_ai.messages import (
    ModelResponse as PydanticModelResponse,
)
from pydantic_ai.messages import (
    TextPart as PydanticTextPart,
)
from pydantic_ai.messages import (
    ToolCallPart as PydanticToolCallPart,
)
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.tools import ToolDefinition as PydanticToolDefinition

from mam_bench.runtime import (
    ModelRequest,
    ModelResponse,
    ModelUsage,
    RuntimeCapability,
    RuntimeDescriptor,
    RuntimeInfrastructureError,
    RuntimeMessage,
    RuntimeProtocolError,
    SamplingControl,
    TextOutput,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)

_OPENROUTER_ENVIRONMENT = "OPENROUTER" + "_API_KEY"
_ENVIRONMENT_NAME_PATTERN = r"^[A-Z_][A-Z0-9_]*$"


class PydanticRuntimeSettings(BaseModel):
    """Provider binding and routing policy for the PydanticAI adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = "qwen-qwen3.8-27b"
    provider: Literal["openrouter", "openai-compatible"] = "openrouter"
    base_url: str = "https://openrouter.ai/api/v1"
    model_name: str = "qwen/qwen3.8-27b"
    credential_environment: str = Field(
        default=_OPENROUTER_ENVIRONMENT,
        pattern=_ENVIRONMENT_NAME_PATTERN,
    )
    openrouter_provider_slug: str = "phala"
    openrouter_allow_fallbacks: bool = False
    openrouter_require_parameters: bool = True
    max_concurrent_requests: int = Field(default=16, ge=1)

    @model_validator(mode="after")
    def validate_openrouter_endpoint(self) -> Self:
        if self.provider == "openrouter" and self.base_url != "https://openrouter.ai/api/v1":
            raise ValueError("OpenRouter runtime requires its fixed API endpoint")
        if self.provider == "openrouter" and self.credential_environment != _OPENROUTER_ENVIRONMENT:
            raise ValueError("OpenRouter runtime requires OPENROUTER_API_KEY")
        if self.openrouter_allow_fallbacks:
            raise ValueError("OpenRouter provider fallbacks must remain disabled")
        if not self.openrouter_require_parameters:
            raise ValueError("OpenRouter must require every declared parameter")
        return self


def pydantic_runtime_descriptor(settings: PydanticRuntimeSettings) -> RuntimeDescriptor:
    """Describe one binding without constructing its provider client."""

    routing_provider: str | None = None
    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    if settings.provider == "openrouter":
        routing_provider = settings.openrouter_provider_slug
        allow_fallbacks = settings.openrouter_allow_fallbacks
        require_parameters = settings.openrouter_require_parameters
    return RuntimeDescriptor(
        model_id=settings.model_id,
        runtime="pydantic-ai-openai",
        provider=settings.provider,
        model=settings.model_name,
        adapter_version="pydantic-runtime-v1",
        capabilities=frozenset(RuntimeCapability),
        sampling_controls=frozenset(SamplingControl),
        max_concurrent_requests=settings.max_concurrent_requests,
        endpoint=settings.base_url,
        routing_provider=routing_provider,
        allow_fallbacks=allow_fallbacks,
        require_parameters=require_parameters,
        required_environment=(settings.credential_environment,),
    )


class PydanticRuntimeFactory:
    """Create a fresh Pydantic Model Runtime after Compatibility Preflight."""

    def __init__(self, settings: PydanticRuntimeSettings) -> None:
        self.settings = settings
        self.descriptor = pydantic_runtime_descriptor(settings)

    def create(self) -> "PydanticModelRuntime":
        return PydanticModelRuntime(self.settings)


class PydanticModelRuntime:
    """Translate repository-owned requests to one PydanticAI model call."""

    def __init__(
        self,
        settings: PydanticRuntimeSettings,
        *,
        model: Model | None = None,
    ) -> None:
        self.settings = settings
        self.descriptor = pydantic_runtime_descriptor(settings)
        self._model = model or _build_model(settings)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        """Execute one exact provider request without dropping requested controls."""

        started = time.perf_counter()
        try:
            response = await self._model.request(
                _to_pydantic_messages(request.messages),
                _model_settings(request, self.settings),
                _request_parameters(request),
            )
        except (ModelAPIError, APIError, OSError, TimeoutError) as exc:
            raise RuntimeInfrastructureError(_error_text(exc)) from exc
        except UnexpectedModelBehavior as exc:
            raise RuntimeProtocolError(_error_text(exc)) from exc
        _validate_response_limits(request, response)
        messages = _from_pydantic_response(response)
        output = _response_output(request, response)
        usage = response.usage
        return ModelResponse(
            request_id=request.request_id,
            messages=messages,
            output=output,
            usage=ModelUsage(
                requests=1,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
            ),
            latency_seconds=time.perf_counter() - started,
            provider_response_id=response.provider_response_id,
        )


def _build_model(settings: PydanticRuntimeSettings) -> OpenAIChatModel:
    api_key = os.environ.get(settings.credential_environment)
    if not api_key:
        raise RuntimeInfrastructureError(
            "required credential environment variable is missing: "
            f"{settings.credential_environment}"
        )
    if settings.provider == "openrouter":
        provider = OpenRouterProvider(api_key=api_key, app_title="MAM-Bench")
    else:
        provider = OpenAIProvider(base_url=settings.base_url, api_key=api_key)
    return OpenAIChatModel(settings.model_name, provider=provider)


def _model_settings(
    request: ModelRequest,
    runtime: PydanticRuntimeSettings,
) -> OpenAIChatModelSettings:
    extra_body: dict[str, object] = {"top_k": request.sampling.top_k}
    if runtime.provider == "openrouter":
        extra_body["provider"] = {
            "only": [runtime.openrouter_provider_slug],
            "allow_fallbacks": runtime.openrouter_allow_fallbacks,
            "require_parameters": runtime.openrouter_require_parameters,
        }
    settings = OpenAIChatModelSettings(
        temperature=request.sampling.temperature,
        top_p=request.sampling.top_p,
        max_tokens=request.sampling.max_output_tokens,
        timeout=request.limits.timeout_seconds,
        openai_reasoning_effort=cast(Any, request.sampling.reasoning_effort),
        extra_body=extra_body,
    )
    if request.sampling.request_seed is not None:
        settings["seed"] = request.sampling.request_seed
    if runtime.provider == "openai-compatible":
        settings["parallel_tool_calls"] = False
    return settings


def _request_parameters(request: ModelRequest) -> ModelRequestParameters:
    function_tools = [
        PydanticToolDefinition(
            name=tool.name,
            description=tool.description,
            parameters_json_schema=cast(Any, tool.input_schema),
            strict=tool.strict,
        )
        for tool in request.tools
    ]
    if isinstance(request.output, TextOutput):
        return ModelRequestParameters(
            function_tools=function_tools,
            output_mode="text",
            allow_text_output=True,
        )
    return ModelRequestParameters(
        function_tools=function_tools,
        output_mode="tool",
        output_tools=[
            PydanticToolDefinition(
                name=request.output.name,
                description=request.output.description,
                parameters_json_schema=cast(Any, request.output.json_schema),
                strict=request.output.strict,
                kind="output",
            )
        ],
        allow_text_output=False,
    )


def _to_pydantic_messages(messages: tuple[RuntimeMessage, ...]) -> list[ModelMessage]:
    translated: list[ModelMessage] = []
    for message in messages:
        if message.role == "assistant":
            response_parts: list[PydanticTextPart | PydanticToolCallPart] = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    response_parts.append(PydanticTextPart(content=part.text))
                elif isinstance(part, ToolCallPart):
                    response_parts.append(
                        PydanticToolCallPart(
                            tool_name=part.name,
                            args=cast(dict[str, Any], part.arguments),
                            tool_call_id=part.call_id,
                        )
                    )
                else:
                    raise RuntimeProtocolError("assistant messages cannot contain tool results")
            translated.append(PydanticModelResponse(parts=response_parts))
        else:
            request_parts: list[SystemPromptPart | UserPromptPart | ToolReturnPart] = []
            for part in message.parts:
                if isinstance(part, TextPart):
                    if message.role == "system":
                        request_parts.append(SystemPromptPart(content=part.text))
                    else:
                        request_parts.append(UserPromptPart(content=part.text))
                elif isinstance(part, ToolResultPart):
                    request_parts.append(
                        ToolReturnPart(
                            tool_name=part.name,
                            content=cast(Any, part.result),
                            tool_call_id=part.call_id,
                        )
                    )
                else:
                    raise RuntimeProtocolError(
                        "request messages cannot contain assistant tool calls"
                    )
            translated.append(PydanticModelRequest(parts=request_parts))
    return translated


def _validate_response_limits(
    request: ModelRequest,
    response: PydanticModelResponse,
) -> None:
    tool_call_count = sum(isinstance(part, PydanticToolCallPart) for part in response.parts)
    if tool_call_count > request.limits.tool_call_limit:
        raise RuntimeProtocolError("response exceeded the tool-call limit")


def _from_pydantic_response(
    response: PydanticModelResponse,
) -> tuple[RuntimeMessage, ...]:
    parts: list[TextPart | ToolCallPart] = []
    for part in response.parts:
        if isinstance(part, PydanticTextPart):
            parts.append(TextPart(text=part.content))
        elif isinstance(part, PydanticToolCallPart):
            parts.append(
                ToolCallPart(
                    call_id=part.tool_call_id,
                    name=part.tool_name,
                    arguments=cast(dict[str, JsonValue], part.args_as_dict()),
                )
            )
        else:
            raise RuntimeProtocolError(f"unsupported PydanticAI response part: {part.part_kind}")
    if not parts:
        raise RuntimeProtocolError("PydanticAI response contains no auditable output")
    return (RuntimeMessage(role="assistant", parts=tuple(parts)),)


def _response_output(
    request: ModelRequest,
    response: PydanticModelResponse,
) -> JsonValue:
    if isinstance(request.output, TextOutput):
        text_parts = [part.content for part in response.parts if isinstance(part, PydanticTextPart)]
        if not text_parts:
            raise RuntimeProtocolError("text request returned no text output")
        return "".join(text_parts)
    calls = [
        part
        for part in response.parts
        if isinstance(part, PydanticToolCallPart) and part.tool_name == request.output.name
    ]
    if len(calls) != 1:
        raise RuntimeProtocolError(
            f"structured request requires one {request.output.name} output call"
        )
    return cast(dict[str, JsonValue], calls[0].args_as_dict())


def _error_text(error: BaseException) -> str:
    text = f"{type(error).__name__}: {error}"[:500]
    text = re.sub(r"(?i)Bearer\s+[^\s,;]+", "Bearer [REDACTED]", text)
    text = re.sub(r"sk-[A-Za-z0-9_-]+", "[REDACTED]", text)
    return re.sub(
        r"(?i)(api[_-]?key|authorization|password|token)\s*[:=]\s*[^\s,;]+",
        r"\1=[REDACTED]",
        text,
    )
