"""Discover serving constraints before running cases; YAML owns request choices."""

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import quote

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai.models import Model, ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from mam_bench.config import AgentSettings, ModelSelection
from mam_bench.diagnostics import ExecutionFailure
from mam_bench.runtime import OpenRouterSettings, VLLMSettings, create_model

logger = logging.getLogger(__name__)


class EndpointMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    name: str = ""
    tag: str
    provider_name: str
    status: int
    context_length: int | None = Field(default=None, gt=0)
    max_completion_tokens: int | None = Field(default=None, gt=0)
    max_prompt_tokens: int | None = Field(default=None, gt=0)
    supported_parameters: list[str] | None = None
    supports_tool_choice: dict[str, bool] | None = None


class ModelEndpointData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    endpoints: list[EndpointMetadata]


class EndpointResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    data: ModelEndpointData


class AdvertisedReasoning(BaseModel):
    """Catalog reasoning options, kept whole as evidence; never enforced."""

    model_config = ConfigDict(extra="allow")
    supported_efforts: list[str] | None = None


class CatalogEntry(BaseModel):
    model_config = ConfigDict(extra="ignore")
    id: str
    reasoning: AdvertisedReasoning | None = None


class CatalogResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    data: list[CatalogEntry]


class VLLMModelMetadata(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True)

    id: str
    max_model_len: int | None = Field(default=None, gt=0)


@dataclass(frozen=True)
class ResolvedModelSetup:
    """Requested selection, discovered context and JSON-safe serving evidence."""

    selection: ModelSelection
    context_window_tokens: int
    metadata: dict[str, object]


async def _fetch_endpoints(model: str) -> tuple[EndpointResponse, AdvertisedReasoning | None]:
    """Endpoint limits plus the model's advertised reasoning options, if any."""
    connection = OpenRouterSettings()  # pyright: ignore[reportCallIssue]
    async with AsyncOpenAI(
        base_url=connection.base_url,
        api_key=connection.api_key.get_secret_value(),
        max_retries=0,
        timeout=30,
    ) as client:
        payload = await client.get(
            f"/models/{quote(model, safe='/')}/endpoints", cast_to=dict[str, object]
        )
        # Only the catalog listing carries reasoning options; record, never enforce.
        catalog = CatalogResponse.model_validate(
            await client.get("/models", cast_to=dict[str, object])
        )
        reasoning = next((e.reasoning for e in catalog.data if e.id == model), None)
        return EndpointResponse.model_validate(payload), reasoning


async def probe_request(model: Model, settings: AgentSettings) -> None:
    """Send one real request with the run's settings so a provider refusal stops setup."""
    request_settings = ModelSettings()
    if settings.max_completion_tokens is not None:
        request_settings["max_tokens"] = settings.max_completion_tokens
    try:
        await model.request(
            [ModelRequest(parts=[UserPromptPart("Reply with OK.")])],
            request_settings,
            ModelRequestParameters(),
        )
    except ModelAPIError as error:
        raise ExecutionFailure(
            "provider_probe", f"Provider refused the startup request: {error}"
        ) from error


async def _fetch_vllm_model(selection: ModelSelection) -> VLLMModelMetadata:
    async with AsyncOpenAI(
        base_url=selection.base_url,
        api_key=VLLMSettings().api_key.get_secret_value(),
        max_retries=0,
        timeout=30,
    ) as client:
        models = await client.models.list()
        for model in models.data:
            if model.id == selection.model:
                return VLLMModelMetadata.model_validate(model.model_dump())
    raise ValueError(f"Selected model {selection.model!r} is not served by the vLLM endpoint")


def _matches_provider(endpoint: EndpointMetadata, provider: str) -> bool:
    selected = provider.casefold()
    tag = endpoint.tag.casefold()
    return selected in (tag, endpoint.provider_name.casefold()) or (
        "/" not in selected and tag.startswith(selected + "/")
    )


def _resolve_limits(
    settings: AgentSettings,
    context: int,
    output_limits: list[int],
    prompt_context: int | None = None,
) -> AgentSettings:
    output_fields = ("max_completion_tokens", "summary_completion_tokens")
    default_output = min([max(1, context // 4), *output_limits])
    settings = settings.model_copy(
        update={
            field: default_output for field in output_fields if getattr(settings, field) is None
        }
    )
    settings.compaction_tail(prompt_context if prompt_context is not None else context)
    for field in output_fields:
        requested = getattr(settings, field)
        if requested is not None and requested > min([context, *output_limits]):
            raise ValueError(f"settings.{field} exceeds the selected serving limit")
    return settings


async def resolve_model_setup(selection: ModelSelection) -> ResolvedModelSetup:
    """Discover serving limits and default output caps to a quarter of context."""
    if selection.runtime in ("openai", "anthropic"):
        context = create_model(selection).context_window
        if context is None:
            service = "OpenAI" if selection.runtime == "openai" else "Anthropic"
            raise ValueError(f"Selected {service} model has no PydanticAI context metadata")
        selection.settings.compaction_tail(context)
        for cap in (
            selection.settings.max_completion_tokens,
            selection.settings.summary_completion_tokens,
        ):
            if cap is not None and cap > context:
                raise ValueError("Completion token limit exceeds the selected model context")
        return ResolvedModelSetup(
            selection,
            context,
            {
                "source": "pydantic_ai_model_profile",
                "model": selection.model,
                "context_window_tokens": context,
                "output_limit_policy": "Omitted caps use native provider defaults",
                "advertised_reasoning": None,
            },
        )

    if selection.runtime == "vllm":
        model = await _fetch_vllm_model(selection)
        if model.max_model_len is None:
            raise ValueError("Selected vLLM model does not report its serving context limit")
        selection = selection.model_copy(
            update={"settings": _resolve_limits(selection.settings, model.max_model_len, [])}
        )
        return ResolvedModelSetup(
            selection,
            model.max_model_len,
            {
                "source": "vllm_models",
                "fetched_at": datetime.now(UTC).isoformat(),
                "model": model.model_dump(mode="json"),
                "context_window_tokens": model.max_model_len,
                "advertised_reasoning": None,
            },
        )

    requested_provider = selection.provider
    assert requested_provider is not None
    response, advertised = await _fetch_endpoints(selection.model)
    endpoints = [
        endpoint
        for endpoint in response.data.endpoints
        if _matches_provider(endpoint, requested_provider) and endpoint.status == 0
    ]
    if not endpoints:
        raise ValueError(f"No active endpoints for selected provider {selection.provider!r}")
    if any(endpoint.context_length is None for endpoint in endpoints):
        raise ValueError("Selected OpenRouter endpoints do not report their serving context limits")
    context = min(
        endpoint.context_length for endpoint in endpoints if endpoint.context_length is not None
    )
    prompt_limits = [e.max_prompt_tokens for e in endpoints if e.max_prompt_tokens is not None]
    prompt_context = min([context, *prompt_limits])
    output_limits = [
        e.max_completion_tokens for e in endpoints if e.max_completion_tokens is not None
    ]
    selection = selection.model_copy(
        update={
            "settings": _resolve_limits(selection.settings, context, output_limits, prompt_context)
        }
    )

    known_parameters = [
        set(endpoint.supported_parameters)
        for endpoint in endpoints
        if endpoint.supported_parameters is not None
    ]
    supported = set[str].intersection(*known_parameters) if known_parameters else None
    # Explicit sampling and reasoning values are the provider's to accept or refuse.
    if supported is not None and not {"tools", "tool_choice"} <= supported:
        raise ValueError("Selected provider endpoints do not support tools and tool_choice")
    effort = selection.settings.reasoning_effort
    efforts = advertised.supported_efforts if advertised is not None else None
    if effort is not None and efforts is not None and effort not in efforts:
        # OpenRouter remaps valid-but-unadvertised efforts without an error.
        logger.warning("reasoning_effort.unadvertised effort=%s advertised=%s", effort, efforts)

    completion_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    if supported is not None and "max_tokens" not in supported:
        if "max_completion_tokens" not in supported:
            raise ValueError("Selected endpoints have no shared completion token parameter")
        completion_parameter = "max_completion_tokens"

    required_support = None
    if any(
        e.supports_tool_choice is not None and e.supports_tool_choice.get("required") is False
        for e in endpoints
    ):
        required_support = False
    if required_support is False and any(
        e.supports_tool_choice is not None and e.supports_tool_choice.get("auto") is False
        for e in endpoints
    ):
        raise ValueError("Selected endpoints have no shared usable tool_choice mode")
    if selection.settings.tool_choice is not None and any(
        e.supports_tool_choice is not None
        and e.supports_tool_choice.get(selection.settings.tool_choice) is False
        for e in endpoints
    ):
        raise ValueError("Explicit tool_choice is unsupported by selected endpoints")
    effective = selection.with_setup(completion_parameter, required_support)
    # Provider routing takes slugs; metadata may also match a display name.
    if requested_provider.casefold() == endpoints[0].provider_name.casefold():
        matching_tags = {endpoint.tag.split("/", 1)[0] for endpoint in endpoints}
        if len(matching_tags) != 1:
            raise ValueError("Provider display name matches multiple slugs; choose a provider tag")
        effective = effective.model_copy(update={"provider": matching_tags.pop()})
    return ResolvedModelSetup(
        effective,
        prompt_context,
        {
            "source": "openrouter_model_endpoints",
            "fetched_at": datetime.now(UTC).isoformat(),
            "model": selection.model,
            "provider": effective.provider,
            "requested_provider": selection.provider,
            "endpoints": [endpoint.model_dump(mode="json") for endpoint in endpoints],
            "context_window_tokens": prompt_context,
            "context_length": context,
            "supported_parameters": sorted(supported) if supported is not None else None,
            "parameter_metadata_complete": len(known_parameters) == len(endpoints),
            "completion_token_parameter": completion_parameter,
            "endpoint_status_policy": "Only status=0 accepted; not a health guarantee",
            "advertised_reasoning": advertised.model_dump(mode="json")
            if advertised is not None
            else None,
        },
    )
