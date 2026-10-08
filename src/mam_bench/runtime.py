"""Provider construction and settings shared by every simulation."""

import json
import os
from dataclasses import dataclass
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import Field, SecretStr
from pydantic_ai.messages import ModelResponse
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

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import AgentSettings, ModelSelection, OpenAICompatibleModel


class _RouterModel(OpenRouterModel):
    """Preserve whether cache telemetry was actually reported by the provider."""

    def _process_response(self, response: ChatCompletion | str) -> ModelResponse:
        result = super()._process_response(response)
        reported = False
        if isinstance(response, ChatCompletion) and response.usage is not None:
            details = response.usage.prompt_tokens_details
            reported = details is not None and (
                details.cached_tokens is not None
                or getattr(details, "cache_write_tokens", None) is not None
            )
        result.provider_details = {
            **(result.provider_details or {}),
            "cache_usage_reported": reported,
        }
        return result


@dataclass(frozen=True)
class CaseRuntime:
    """Model access, agent settings, and artifacts for exactly one test case."""

    model: Model
    settings: AgentSettings
    writer: ArtifactWriter


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


class _CompatibleChatModel(OpenAIChatModel):
    """Normalize SGLang telemetry while retaining normal completion validation."""

    def _process_response(self, response: ChatCompletion | str) -> ModelResponse:
        if isinstance(response, ChatCompletion):
            metadata = cast(dict[str, object] | None, response.metadata)
            if isinstance(metadata, dict) and isinstance(metadata.get("weight_versions"), list):
                # SGLang emits spans here; OpenAI metadata permits only string values.
                response = response.model_copy(
                    update={
                        "metadata": {
                            **metadata,
                            "weight_versions": json.dumps(metadata["weight_versions"]),
                        }
                    }
                )
        return super()._process_response(response)


def create_model(selection: ModelSelection) -> Model:
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
        model = _CompatibleChatModel(
            selection.model,
            # Local chat templates can require exactly one leading system message.
            # Let PydanticAI combine instructions without changing their role or text.
            profile=OpenAIModelProfile(
                openai_chat_supports_multiple_system_messages=False,
                openai_supports_tool_choice_required=force_tools,
                openai_supports_forced_tool_choice_with_thinking=force_tools,
            ),
            provider=OpenAIProvider(
                base_url=selection.base_url,
                api_key=os.environ[selection.api_key_env],
            ),
            settings=OpenAIChatModelSettings(
                **request_settings,
                openai_reasoning_effort=agent_settings.reasoning_effort,
            ),
        )
    else:
        settings = OpenRouterSettings()  # pyright: ignore[reportCallIssue]
        for parameter in selection.omitted_parameters:
            if parameter in ("temperature", "top_p"):
                request_settings.pop(parameter, None)
            elif parameter == "top_k":
                request_settings.pop("extra_body", None)
        router_settings = OpenRouterModelSettings(
            **request_settings,
            openrouter_provider={
                "only": [selection.provider],
                "allow_fallbacks": False,
                "require_parameters": True,
            },
        )
        if "reasoning" not in selection.omitted_parameters:
            router_settings["openrouter_reasoning"] = {"effort": agent_settings.reasoning_effort}
        if selection.cache_strategy == "explicit":
            router_settings["openrouter_cache_instructions"] = True
            router_settings["openrouter_cache_messages"] = True
            router_settings["openrouter_cache_tool_definitions"] = True
        model = _RouterModel(
            selection.model,
            profile=OpenRouterModelProfile(
                openai_chat_supports_max_completion_tokens=(
                    selection.completion_token_parameter == "max_completion_tokens"
                ),
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
            settings=router_settings,
        )

    return model
