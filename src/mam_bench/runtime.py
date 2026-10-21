"""Provider construction and settings shared by every simulation."""

from dataclasses import dataclass
from typing import cast

from openai import AsyncOpenAI
from pydantic import Field, SecretStr
from pydantic_ai.models import Model
from pydantic_ai.models.anthropic import AnthropicModel, AnthropicModelSettings
from pydantic_ai.models.openai import OpenAIResponsesModel, OpenAIResponsesModelSettings
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
)
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.vllm import VLLMProvider
from pydantic_ai.settings import ModelSettings
from pydantic_settings import BaseSettings, SettingsConfigDict

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import AgentSettings, ModelSelection
from mam_bench.kludge import (
    OpenAIFixedArrayTransformer,
    VLLMChatModel,
    openrouter_reasoning,
    vllm_settings,
)


@dataclass(frozen=True)
class CaseRuntime:
    """Model access, agent settings, and artifacts for exactly one test case."""

    model: Model
    settings: AgentSettings
    writer: ArtifactWriter
    context_window_tokens: int


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
        request_settings["thinking"] = (
            False if effort == "none" else "xhigh" if effort == "max" else effort
        )
    if agent_settings.top_k is not None:
        request_settings["extra_body"] = {"top_k": agent_settings.top_k}

    if selection.runtime == "anthropic":
        request_settings.pop("extra_body", None)
        if agent_settings.top_k is not None:
            request_settings["top_k"] = agent_settings.top_k
        if agent_settings.tool_choice is not None:
            request_settings["tool_choice"] = agent_settings.tool_choice
        return AnthropicModel(
            selection.model,
            provider=AnthropicProvider(),
            settings=AnthropicModelSettings(
                **request_settings,
                anthropic_cache=True,
                anthropic_cache_instructions=True,
                anthropic_cache_tool_definitions=True,
            ),
        )

    if selection.runtime == "openai":
        if agent_settings.top_k is not None:
            raise ValueError("OpenAI does not support settings.top_k")
        profile["json_schema_transformer"] = OpenAIFixedArrayTransformer
        return OpenAIResponsesModel(
            selection.model,
            provider=OpenAIProvider(),
            profile=profile,
            settings=OpenAIResponsesModelSettings(
                **request_settings,
                openai_store=False,
                openai_prompt_cache_key=f"mam-bench:{selection.model}",
            ),
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
                base_url=selection.base_url,
                api_key=VLLMSettings().api_key.get_secret_value(),
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
            router_settings["openrouter_reasoning"] = openrouter_reasoning(effort)
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
                )
            ),
            settings=router_settings,
        )

    return model
