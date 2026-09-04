"""Benchmark interfaces, results, and PydanticAI model construction."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_settings import BaseSettings, SettingsConfigDict

from mam_bench.config import ModelSelection, OpenAICompatibleModel


class AgentSettings(BaseModel):
    """How MAM-Bench calls PydanticAI agents."""

    model_config = ConfigDict(frozen=True)

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    reasoning_effort: str = "medium"
    max_completion_tokens: int = 32_768
    concurrency: int = 4
    timeout_seconds: float = 3_600.0


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
    """Create the PydanticAI model selected in YAML."""

    if isinstance(selection, OpenAICompatibleModel):
        endpoint = selection.base_url
        provider_name = "openai-compatible"
        routing_provider = None
        provider = OpenAIProvider(
            base_url=endpoint,
            api_key=os.environ[selection.api_key_env],
        )
    else:
        settings = OpenRouterSettings()  # pyright: ignore[reportCallIssue]
        endpoint = settings.base_url
        provider_name = "openrouter"
        routing_provider = selection.provider
        provider = OpenRouterProvider(
            openai_client=AsyncOpenAI(
                base_url=settings.base_url,
                api_key=settings.api_key.get_secret_value(),
                default_headers={"X-Title": "MAM-Bench"},
            )
        )

    return ModelRuntime(
        info=RuntimeInfo(
            model_id=selection.id,
            provider=provider_name,
            model=selection.model,
            endpoint=endpoint,
            routing_provider=routing_provider,
        ),
        model=OpenAIChatModel(selection.model, provider=provider),
    )


class PrimaryScore(BaseModel):
    """The main score produced by one simulation run."""

    name: str
    value: float
    unit: str
    higher_is_better: bool = True


class ToplineEntry(BaseModel):
    """The score and model details for one simulation-model run."""

    simulation_id: str
    simulation_version: str
    model_id: str
    provider: str
    model: str
    primary_score: PrimaryScore


class BenchmarkTopline(BaseModel):
    """All scores from a benchmark run, in execution order."""

    entries: tuple[ToplineEntry, ...]


class BenchmarkSimulation(Protocol):
    """Interface every built-in simulation implements."""

    simulation_id: str
    simulation_version: str

    async def run(
        self,
        runtime: ModelRuntime,
        output_directory: Path,
    ) -> PrimaryScore: ...
