"""Provider construction and settings shared by every simulation."""

import os
from dataclasses import dataclass
from typing import cast

from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion
from pydantic import Field, SecretStr
from pydantic_ai.messages import ModelResponse
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import (
    OpenRouterModel,
    OpenRouterModelSettings,
)
from pydantic_ai.profiles.openai import OpenAIModelProfile
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.providers.vllm import VLLMProvider
from pydantic_ai.settings import ModelSettings
from pydantic_settings import BaseSettings, SettingsConfigDict

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import AgentSettings, ModelSelection, VLLMModel


@dataclass(frozen=True)
class CaseRuntime:
    """Model access, agent settings, and artifacts for exactly one test case."""

    model: Model
    settings: AgentSettings
    writer: ArtifactWriter
    context_window_tokens: int


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


class _VLLMChatModel(OpenAIChatModel):
    """Normalize vLLM repetition stops until the native model accepts them."""

    def _process_response(self, response: ChatCompletion | str) -> ModelResponse:
        repetition = isinstance(response, ChatCompletion) and any(
            str(choice.finish_reason) == "repetition" for choice in response.choices
        )
        if isinstance(response, ChatCompletion) and repetition:
            # Native early termination uses the same bounded incomplete-output
            # handling as a length stop; preserve its actual cause below.
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
    if agent_settings.reasoning_effort is not None:
        request_settings["thinking"] = (
            False if agent_settings.reasoning_effort == "none" else agent_settings.reasoning_effort
        )
    if agent_settings.top_k is not None:
        request_settings["extra_body"] = {"top_k": agent_settings.top_k}

    if isinstance(selection, VLLMModel):
        if selection.repetition_detection is not None:
            request_settings["extra_body"] = {
                **cast(dict[str, object], request_settings.get("extra_body") or {}),
                "repetition_detection": selection.repetition_detection.model_dump(),
            }
        model = _VLLMChatModel(
            selection.model,
            profile=profile,
            provider=VLLMProvider(
                base_url=selection.base_url,
                api_key=os.environ[selection.api_key_env],
            ),
            settings=request_settings,
        )
    else:
        settings = OpenRouterSettings()  # pyright: ignore[reportCallIssue]
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
