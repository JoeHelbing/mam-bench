"""Provider construction and settings shared by every simulation."""

import os
from dataclasses import dataclass
from typing import Literal

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
)
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterModelProfile, OpenRouterProvider
from pydantic_ai.settings import ModelSettings
from pydantic_settings import BaseSettings, SettingsConfigDict

from mam_bench.config import AgentSettings, ModelSelection, OpenAICompatibleModel


class RuntimeInfo(BaseModel):
    """Model, provider, and agent settings saved with benchmark results."""

    model_config = ConfigDict(frozen=True)

    model_id: str
    provider: Literal["openrouter", "openai-compatible"]
    model: str
    endpoint: str
    routing_provider: str | None = None
    agent_settings: AgentSettings = AgentSettings()


@dataclass(frozen=True)
class ModelRuntime:
    """A PydanticAI model paired with its benchmark metadata."""

    info: RuntimeInfo
    model: Model


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


def create_runtime(selection: ModelSelection) -> ModelRuntime:
    """Create a model whose request defaults also apply to history summaries."""

    agent_settings = selection.settings
    # Provider-level forcing must retain PydanticAI's terminal output tools.
    # ModelSettings.tool_choice='required' instead forces function tools only.
    force_tools = agent_settings.tool_choice == "required"
    request_settings = ModelSettings(
        temperature=agent_settings.temperature,
        top_p=agent_settings.top_p,
        max_tokens=agent_settings.max_completion_tokens,
        timeout=agent_settings.timeout_seconds,
        extra_body={"top_k": agent_settings.top_k},
    )

    if isinstance(selection, OpenAICompatibleModel):
        endpoint = selection.base_url
        provider_name = "openai-compatible"
        routing_provider = None
        model = OpenAIChatModel(
            selection.model,
            # Local chat templates can require exactly one leading system message.
            # Let PydanticAI combine instructions without changing their role or text.
            profile=OpenAIModelProfile(
                openai_chat_supports_multiple_system_messages=False,
                openai_supports_tool_choice_required=force_tools,
                openai_supports_forced_tool_choice_with_thinking=force_tools,
            ),
            provider=OpenAIProvider(
                base_url=endpoint,
                api_key=os.environ[selection.api_key_env],
            ),
            settings=OpenAIChatModelSettings(
                **request_settings,
                openai_reasoning_effort=agent_settings.reasoning_effort,
            ),
        )
    else:
        settings = OpenRouterSettings()  # pyright: ignore[reportCallIssue]
        endpoint = settings.base_url
        provider_name = "openrouter"
        routing_provider = selection.provider
        model = OpenRouterModel(
            selection.model,
            profile=OpenRouterModelProfile(
                openai_supports_tool_choice_required=force_tools,
                openai_supports_forced_tool_choice_with_thinking=force_tools,
                openrouter_supports_forced_tool_choice_with_thinking=force_tools,
            ),
            provider=OpenRouterProvider(
                openai_client=AsyncOpenAI(
                    base_url=settings.base_url,
                    api_key=settings.api_key.get_secret_value(),
                    default_headers={"X-Title": "MAM-Bench"},
                )
            ),
            settings=OpenRouterModelSettings(
                **request_settings,
                openrouter_provider={
                    "only": [selection.provider],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                openrouter_reasoning={"effort": agent_settings.reasoning_effort},
            ),
        )

    return ModelRuntime(
        info=RuntimeInfo(
            model_id=selection.id,
            provider=provider_name,
            model=selection.model,
            endpoint=endpoint,
            routing_provider=routing_provider,
            agent_settings=agent_settings,
        ),
        model=model,
    )
