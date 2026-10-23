"""Provider construction and settings shared by every simulation."""

import hashlib
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import cast

import anthropic
import httpx2 as httpx
import openai
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from openai.types.shared import ReasoningEffort
from pydantic import Field, SecretStr
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
    OpenRouterReasoning,
)
from pydantic_ai.profiles.anthropic import AnthropicEffort
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.vllm import VLLMProvider
from pydantic_ai.settings import ModelSettings
from pydantic_settings import BaseSettings, SettingsConfigDict

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import AgentSettings, ModelSelection
from mam_bench.kludge import OpenAIFixedArrayTransformer, VLLMChatModel, vllm_settings
from mam_bench.qualification import FallbackLedger

# Each provider client retries HTTP 408/409/429, server errors, and connection
# failures itself with exponential backoff (0.5 s doubling to 8 s, honoring
# Retry-After), about 80-100 s per request, before a session-level retry.
PROVIDER_MAX_RETRIES = 16

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaseRuntime:
    """Model access, agent settings, and artifacts for exactly one test case."""

    model: Model
    settings: AgentSettings
    writer: ArtifactWriter
    context_window_tokens: int
    # A benchmark run shares one ledger across its cases; standalone cases have none.
    fallbacks: FallbackLedger | None = None


class VLLMSettings(BaseSettings):
    """vLLM credentials loaded from the environment or `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
        hide_input_in_errors=True,
    )

    # Match VLLMProvider's internal placeholder for the OpenAI client's credential check.
    api_key: SecretStr = SecretStr("api-key-not-set")


class OpenRouterSettings(BaseSettings):
    """OpenRouter connection settings loaded from the environment or `.env`."""

    model_config = SettingsConfigDict(
        env_prefix="OPENROUTER_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    base_url: str = "https://openrouter.ai/api/v1"
    api_key: SecretStr = Field(min_length=1)


def _wire_recorder() -> Callable[[httpx.Request], Awaitable[None]]:
    """Log each distinct request body once; conversation content stays in traces."""
    seen: set[str] = set()

    async def record(request: httpx.Request) -> None:
        try:
            body = json.loads(request.content)
        except ValueError:
            return
        if not isinstance(body, dict):
            return
        shape = json.dumps(
            {
                k: v
                for k, v in cast(dict[str, object], body).items()
                if k not in ("messages", "input")
            },
            sort_keys=True,
        )
        key = hashlib.sha256(shape.encode()).hexdigest()
        if key not in seen:
            seen.add(key)
            logger.info("request.wire method=%s url=%s body=%s", request.method, request.url, shape)

    return record


def create_model(selection: ModelSelection) -> Model:
    """Apply shared request overrides; sessions own action and summary output caps."""

    agent_settings = selection.settings
    profile = OpenAIModelProfile()
    if agent_settings.tool_choice == "auto":
        # Tool-only output implies forcing. Opt out without asserting support
        # or disabling the providers' conditional safety checks.
        profile["openai_supports_tool_choice_required"] = False
    request_settings = ModelSettings(timeout=agent_settings.timeout_seconds)
    if agent_settings.temperature is not None:
        request_settings["temperature"] = agent_settings.temperature
    if agent_settings.top_p is not None:
        request_settings["top_p"] = agent_settings.top_p
    if (effort := agent_settings.reasoning_effort) is not None:
        # Unified thinking only switches reasoning; each runtime's native field
        # below carries the configured effort verbatim for the provider to judge.
        request_settings["thinking"] = effort != "none"
    hooks = {"request": [_wire_recorder()]}
    if agent_settings.top_k is not None:
        request_settings["extra_body"] = {"top_k": agent_settings.top_k}

    if selection.runtime == "anthropic":
        request_settings.pop("extra_body", None)
        if agent_settings.top_k is not None:
            request_settings["top_k"] = agent_settings.top_k
        if agent_settings.tool_choice is not None:
            request_settings["tool_choice"] = agent_settings.tool_choice
        anthropic_settings = AnthropicModelSettings(
            **request_settings,
            anthropic_cache=True,
            anthropic_cache_instructions=True,
            anthropic_cache_tool_definitions=True,
        )
        if effort is not None and effort != "none":
            anthropic_settings["anthropic_effort"] = cast(AnthropicEffort, effort)
        return AnthropicModel(
            selection.model,
            provider=AnthropicProvider(
                anthropic_client=AsyncAnthropic(
                    max_retries=PROVIDER_MAX_RETRIES,
                    http_client=anthropic.DefaultAsyncHttpxClient(event_hooks=hooks),
                )
            ),
            settings=anthropic_settings,
        )

    if selection.runtime == "openai":
        if agent_settings.top_k is not None:
            raise ValueError("OpenAI does not support settings.top_k")
        profile["json_schema_transformer"] = OpenAIFixedArrayTransformer
        openai_settings = OpenAIResponsesModelSettings(
            **request_settings,
            openai_store=False,
            openai_prompt_cache_key=f"mam-bench:{selection.model}",
        )
        if effort is not None:
            openai_settings["openai_reasoning_effort"] = cast(ReasoningEffort, effort)
        return OpenAIResponsesModel(
            selection.model,
            provider=OpenAIProvider(
                openai_client=AsyncOpenAI(
                    max_retries=PROVIDER_MAX_RETRIES,
                    http_client=openai.DefaultAsyncHttpxClient(event_hooks=hooks),
                )
            ),
            profile=profile,
            settings=openai_settings,
        )

    if selection.runtime == "vllm":
        if selection.repetition_detection is not None:
            request_settings["extra_body"] = {
                **cast(dict[str, object], request_settings.get("extra_body") or {}),
                "repetition_detection": selection.repetition_detection.model_dump(),
            }
        model = VLLMChatModel(
            selection.model,
            profile=profile,
            provider=VLLMProvider(
                openai_client=AsyncOpenAI(
                    base_url=selection.base_url,
                    api_key=VLLMSettings().api_key.get_secret_value(),
                    max_retries=PROVIDER_MAX_RETRIES,
                    http_client=openai.DefaultAsyncHttpxClient(event_hooks=hooks),
                )
            ),
            settings=vllm_settings(selection, request_settings),
        )
    else:
        assert selection.provider is not None
        settings = OpenRouterSettings()  # pyright: ignore[reportCallIssue]
        if selection.model.startswith("openai/"):
            profile["json_schema_transformer"] = OpenAIFixedArrayTransformer
        if selection.required_tool_choice_supported is False:
            profile["openai_supports_tool_choice_required"] = False
        router_settings = OpenRouterModelSettings(
            **request_settings,
            openrouter_provider={
                "only": [selection.provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        )
        if effort is not None:
            router_settings["openrouter_reasoning"] = cast(OpenRouterReasoning, {"effort": effort})
        router_settings["openrouter_cache_instructions"] = True
        router_settings["openrouter_cache_messages"] = True
        router_settings["openrouter_cache_tool_definitions"] = True
        profile["openai_chat_supports_max_completion_tokens"] = (
            selection.completion_token_parameter == "max_completion_tokens"
        )
        model = OpenRouterModel(
            selection.model,
            profile=profile,
            provider=OpenRouterProvider(
                openai_client=AsyncOpenAI(
                    base_url=settings.base_url,
                    api_key=settings.api_key.get_secret_value(),
                    default_headers={"X-Title": "MAM-Bench"},
                    max_retries=PROVIDER_MAX_RETRIES,
                    http_client=openai.DefaultAsyncHttpxClient(event_hooks=hooks),
                )
            ),
            settings=router_settings,
        )

    return model
