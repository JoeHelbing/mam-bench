"""Resolve OpenRouter endpoint capabilities once, before creating a benchmark run."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import quote

from openai import APIError, AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from mam_bench.config import AgentSettings, ModelSelection, OpenAICompatibleModel, OpenRouterModel
from mam_bench.runtime import OpenRouterSettings


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
    supports_implicit_caching: bool = False


class ModelEndpointData(BaseModel):
    model_config = ConfigDict(extra="ignore")
    endpoints: list[EndpointMetadata]


class EndpointResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")
    data: ModelEndpointData


@dataclass(frozen=True)
class ResolvedModelSetup:
    """Effective runtime selection and JSON-safe evidence for the run artifact."""

    selection: ModelSelection
    metadata: dict[str, object]


async def _fetch_endpoints(model: str) -> EndpointResponse:
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
        return EndpointResponse.model_validate(payload)


def _matches_provider(endpoint: EndpointMetadata, provider: str) -> bool:
    selected = provider.casefold()
    tag = endpoint.tag.casefold()
    return selected in (tag, endpoint.provider_name.casefold()) or (
        "/" not in selected and tag.startswith(selected + "/")
    )


def _budgets(
    settings: AgentSettings, endpoints: list[EndpointMetadata]
) -> tuple[dict[str, object], dict[str, object]]:
    explicit = settings.model_fields_set
    values: dict[str, object] = {}
    provenance: dict[str, object] = {}

    def limit(field: str, endpoint_field: str) -> int:
        found = [getattr(endpoint, endpoint_field) for endpoint in endpoints]
        supplied = int(getattr(settings, field))
        if not found or any(value is None for value in found):
            if field not in explicit:
                raise ValueError(
                    f"Missing endpoint {endpoint_field}; set settings.{field} explicitly"
                )
            known = [value for value in found if isinstance(value, int)]
            if known and supplied > min(known):
                raise ValueError(f"settings.{field} exceeds the selected provider limit")
            return supplied
        ceiling = min(found)
        if field in explicit and supplied > ceiling:
            raise ValueError(f"settings.{field} exceeds the selected provider limit {ceiling}")
        if field == "context_window_tokens" and field in explicit:
            return supplied
        return ceiling

    context = limit("context_window_tokens", "context_length")
    output_limit = limit("max_completion_tokens", "max_completion_tokens")
    headroom = max(1024, context // 20)
    output = (
        settings.max_completion_tokens
        if "max_completion_tokens" in explicit
        else min(32768, output_limit, max(1, context // 8))
    )
    summary = (
        settings.summary_completion_tokens
        if "summary_completion_tokens" in explicit
        else min(16000, output, max(1, context // 16))
    )
    memory = (
        settings.memory_injection_tokens
        if "memory_injection_tokens" in explicit
        else min(4000, max(1, context // 64))
    )
    # Reserve both normal and summary output. The extra 5% (at least 1024 tokens)
    # covers instructions, tools and a new turn; this is an estimate, not a tokenizer.
    prompt_limits = [e.max_prompt_tokens for e in endpoints if e.max_prompt_tokens is not None]
    prompt_limit = min([context, *prompt_limits])
    available = min(prompt_limit, context - max(output, summary)) - headroom - memory
    if available <= 1 or output > output_limit or summary > output_limit:
        raise ValueError("Token budgets leave no safe prompt room or exceed the output limit")
    fraction = (
        settings.compaction_trigger_fraction
        if "compaction_trigger_fraction" in explicit
        else (available / context)
    )
    trigger = int(context * fraction)
    if trigger > available:
        raise ValueError("compaction_trigger_fraction leaves insufficient output/headroom reserve")
    tail = (
        settings.compaction_tail_tokens
        if "compaction_tail_tokens" in explicit
        else min(40000, trigger // 4)
    )
    if tail + summary + memory >= trigger:
        raise ValueError("Compaction tail, summary and memory must fit below the trigger")
    values.update(
        context_window_tokens=context,
        max_completion_tokens=output,
        summary_completion_tokens=summary,
        memory_injection_tokens=memory,
        compaction_trigger_fraction=fraction,
        compaction_tail_tokens=tail,
    )
    for field in values:
        provenance[field] = "yaml" if field in explicit else "derived_from_endpoint_limits"
    return values, {
        "fields": provenance,
        "headroom_tokens": headroom,
        "prompt_limit_tokens": prompt_limit,
        "output_limit_tokens": output_limit,
        "trigger_tokens": trigger,
    }


def _cache_strategy(
    selection: OpenRouterModel, endpoints: list[EndpointMetadata]
) -> Literal["explicit", "implicit", "unverified", "off"]:
    if selection.prompt_cache == "off":
        return "off"
    if endpoints and all(endpoint.supports_implicit_caching for endpoint in endpoints):
        return "implicit"
    # PydanticAI's native OpenRouter breakpoints support Claude across its upstream
    # providers and Gemini. Require endpoint evidence, not just a model-name guess.
    family = selection.model.removeprefix("~").split("/", 1)[0]
    explicit = bool(endpoints) and all(
        (
            family == "anthropic"
            and endpoint.tag.split("/", 1)[0]
            in {"anthropic", "amazon-bedrock", "google-vertex", "azure"}
        )
        or (
            family == "google"
            and endpoint.tag.split("/", 1)[0] in {"google-ai-studio", "google-vertex"}
        )
        for endpoint in endpoints
    )
    if explicit:
        return "explicit"
    if selection.prompt_cache == "required":
        raise ValueError(
            "prompt_cache required, but selected endpoints have no verified cache support"
        )
    return "unverified"


async def resolve_model_setup(selection: ModelSelection) -> ResolvedModelSetup:
    """Fetch metadata once; never make inference calls or cache model responses.

    Explicit YAML settings are constraints. Unsupported omitted sampling defaults
    are removed from requests, not silently presented as supported configuration.
    """
    if isinstance(selection, OpenAICompatibleModel):
        return ResolvedModelSetup(selection, {"source": "yaml", "runtime": selection.runtime})

    endpoints: list[EndpointMetadata] = []
    metadata_error: str | None = None
    try:
        response = await _fetch_endpoints(selection.model)
    except (APIError, ValidationError) as error:
        # Do not persist exception messages: HTTP errors can contain private URLs.
        metadata_error = type(error).__name__
    else:
        endpoints = [
            endpoint
            for endpoint in response.data.endpoints
            if _matches_provider(endpoint, selection.provider) and endpoint.status == 0
        ]
        if not endpoints:
            raise ValueError(f"No active endpoints for selected provider {selection.provider!r}")

    settings = selection.settings
    explicit = settings.model_fields_set
    supported: set[str] | None = None
    if endpoints and all(endpoint.supported_parameters is not None for endpoint in endpoints):
        supported = set[str].intersection(
            *(set(endpoint.supported_parameters or []) for endpoint in endpoints)
        )
    values, budgets = _budgets(settings, endpoints)
    omitted: set[str] = set()
    decisions: list[str] = []
    completion_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"
    if supported is not None:
        if "tools" not in supported:
            raise ValueError("Selected provider endpoints do not support tools")
        for field, parameter in {
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "reasoning_effort": "reasoning",
            "use_sampling_seed": "seed",
        }.items():
            if parameter not in supported:
                disabled = (field == "use_sampling_seed" and not settings.use_sampling_seed) or (
                    field == "reasoning_effort" and settings.reasoning_effort == "none"
                )
                if field in explicit and not disabled:
                    raise ValueError(
                        f"Explicit settings.{field} is unsupported by selected endpoints"
                    )
                omitted.add(parameter)
                if field == "use_sampling_seed":
                    values[field] = False
                decisions.append(f"Omitted unsupported default {field}")
        if "max_tokens" not in supported:
            if "max_completion_tokens" not in supported:
                raise ValueError("Selected endpoints have no shared completion token parameter")
            completion_parameter = "max_completion_tokens"
        if "tool_choice" not in supported:
            raise ValueError("Selected endpoints do not support tool_choice")
        required_supported = "tool_choice" in supported and all(
            endpoint.supports_tool_choice is not None
            and endpoint.supports_tool_choice.get("required", False)
            for endpoint in endpoints
        )
        if settings.tool_choice == "required" and not required_supported:
            if "tool_choice" in explicit:
                raise ValueError(
                    "Explicit required tool_choice is unsupported by selected endpoints"
                )
            values["tool_choice"] = "auto"
            decisions.append("Default tool_choice changed to auto")
    else:
        if not {"context_window_tokens", "max_completion_tokens"} <= explicit:
            raise ValueError(
                "Endpoint capabilities are missing; set context_window_tokens and "
                "max_completion_tokens explicitly to proceed with unknown capabilities"
            )
        decisions.append("Parameter capabilities unknown; requests retain configured defaults")

    # OpenRouter otherwise silently drops Claude reasoning when forcing a tool.
    if (
        selection.model.removeprefix("~").startswith("anthropic/")
        and "reasoning" not in omitted
        and settings.reasoning_effort != "none"
        and values.get("tool_choice", settings.tool_choice) == "required"
    ):
        if "tool_choice" in explicit:
            raise ValueError(
                "Claude reasoning conflicts with explicit required tool_choice; use auto"
            )
        values["tool_choice"] = "auto"
        decisions.append("Default tool_choice changed to auto to preserve Claude reasoning")

    if values.get("tool_choice", settings.tool_choice) == "auto" and any(
        endpoint.supports_tool_choice is not None
        and not endpoint.supports_tool_choice.get("auto", False)
        for endpoint in endpoints
    ):
        raise ValueError("Selected endpoints do not support auto tool_choice")

    field_sources = {
        field: "yaml"
        if field in explicit
        else ("derived_from_endpoint_metadata" if field in values else "default")
        for field in type(settings).model_fields
    }
    for field, parameter in {
        "temperature": "temperature",
        "top_p": "top_p",
        "top_k": "top_k",
        "reasoning_effort": "reasoning",
        "use_sampling_seed": "seed",
    }.items():
        if parameter in omitted:
            field_sources[field] = "omitted_unsupported"
    effective_settings = AgentSettings.model_validate({**settings.model_dump(), **values})
    effective = selection.with_setup(
        effective_settings,
        frozenset(omitted),
        _cache_strategy(selection, endpoints),
        completion_parameter,
    )
    if endpoints:
        # Provider routing takes slugs, while endpoint metadata also exposes display names.
        matching_tags = {endpoint.tag.split("/", 1)[0] for endpoint in endpoints}
        if selection.provider.casefold() == endpoints[0].provider_name.casefold():
            if len(matching_tags) != 1:
                raise ValueError(
                    "Provider display name matches multiple slugs; choose a provider tag"
                )
            effective = effective.model_copy(update={"provider": matching_tags.pop()})
    return ResolvedModelSetup(
        effective,
        {
            "source": "openrouter_model_endpoints"
            if metadata_error is None
            else "explicit_limits_fallback",
            "endpoint_path": f"/models/{selection.model}/endpoints",
            "fetched_at": datetime.now(UTC).isoformat(),
            "metadata_error": metadata_error,
            "model": selection.model,
            "provider": effective.provider,
            "requested_provider": selection.provider,
            "endpoints": [endpoint.model_dump(mode="json") for endpoint in endpoints],
            "supported_parameters": sorted(supported) if supported is not None else None,
            "omitted_parameters": sorted(omitted),
            "completion_token_parameter": completion_parameter,
            "decisions": decisions,
            "budgets": budgets,
            "field_sources": field_sources,
            "endpoint_status_policy": "Only status=0 accepted; not a health guarantee",
            "null_prompt_limit_policy": (
                "Use shared context limit when no separate prompt limit is published"
            ),
            "prompt_cache": {
                "requested": selection.prompt_cache,
                "strategy": effective.cache_strategy,
                "note": "off disables explicit hints, not provider-managed implicit caching; "
                "metadata support neither guarantees nor rules out actual hits",
            },
            "effective_settings": effective_settings.model_dump(mode="json"),
        },
    )
