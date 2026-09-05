"""Benchmark interfaces, results, and PydanticAI model construction."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol, Self

from openai import AsyncOpenAI
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_ai.models import Model
from pydantic_ai.models.openai import OpenAIChatModel, OpenAIChatModelSettings
from pydantic_ai.models.openrouter import OpenRouterModel, OpenRouterModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.settings import ModelSettings
from pydantic_settings import BaseSettings, SettingsConfigDict

from mam_bench.config import ModelSelection, OpenAICompatibleModel


class AgentSettings(BaseModel):
    """How MAM-Bench calls PydanticAI agents."""

    model_config = ConfigDict(frozen=True)

    temperature: float = 1.0
    top_p: float = 0.95
    top_k: int = 20
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium"
    max_completion_tokens: int = 32_768
    concurrency: int = Field(default=4, gt=0)
    timeout_seconds: float = 3_600.0
    context_window_tokens: int = Field(default=260_000, gt=0)
    compaction_trigger_fraction: float = Field(default=0.7, gt=0, lt=1)
    compaction_tail_tokens: int = Field(default=40_000, ge=0)
    summary_completion_tokens: int = Field(default=16_000, gt=0)
    memory_injection_tokens: int = Field(default=4_000, gt=0)

    @model_validator(mode="after")
    def validate_compaction_tail(self) -> Self:
        """Keep the verbatim tail below the threshold that triggers compaction."""
        trigger_tokens = self.context_window_tokens * self.compaction_trigger_fraction
        if self.compaction_tail_tokens >= trigger_tokens:
            raise ValueError("compaction_tail_tokens must be below the compaction trigger")
        return self


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

    agent_settings = AgentSettings()
    request_settings = ModelSettings(
        temperature=agent_settings.temperature,
        top_p=agent_settings.top_p,
        max_tokens=agent_settings.max_completion_tokens,
        timeout=agent_settings.timeout_seconds,
        parallel_tool_calls=False,
        extra_body={"top_k": agent_settings.top_k},
    )

    if isinstance(selection, OpenAICompatibleModel):
        endpoint = selection.base_url
        provider_name = "openai-compatible"
        routing_provider = None
        model = OpenAIChatModel(
            selection.model,
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


type PairFailureKind = Literal[
    "provider",
    "network",
    "timeout",
    "memory_store",
    "unsupported_context",
    "compaction",
    "evidence_publication",
]


class AgentInfrastructureFailure(RuntimeError):
    """Abort a pair after an unrecoverable agent runtime failure."""

    def __init__(self, kind: PairFailureKind, message: str) -> None:
        self.kind: PairFailureKind = kind
        super().__init__(message)


class PairFailure(BaseModel):
    """Redacted identity and failure category for one unscored pair."""

    model_config = ConfigDict(frozen=True)

    simulation_id: str
    simulation_version: str
    model_id: str
    provider: str
    model: str
    kind: PairFailureKind


class PairInfrastructureFailure(RuntimeError):
    """Signal that one pair failed after publishing its failure evidence."""

    def __init__(self, failure: PairFailure) -> None:
        self.failure = failure
        super().__init__(f"{failure.simulation_id} / {failure.model_id} ({failure.kind})")


class BenchmarkRunFailure(RuntimeError):
    """Report all pair failures after the matrix and topline are complete."""

    def __init__(self, failures: tuple[PairFailure, ...]) -> None:
        self.failures = failures
        pairs = ", ".join(
            f"{failure.simulation_id} / {failure.model_id} ({failure.kind})" for failure in failures
        )
        super().__init__(f"benchmark failed pairs: {pairs}")


class BenchmarkSimulation(Protocol):
    """Interface every built-in simulation implements."""

    simulation_id: str
    simulation_version: str

    async def run(
        self,
        runtime: ModelRuntime,
        output_directory: Path,
    ) -> PrimaryScore: ...
