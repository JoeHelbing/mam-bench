"""Strict YAML configuration for benchmark runs."""

from pathlib import Path, PureWindowsPath
from typing import Annotated, Literal, Self, cast

import yaml
from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator

# YAML loading


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load YAML, resolving output paths against the caller's working directory."""

    with path.open("rb") as config_file:
        document = cast(
            object,
            yaml.safe_load(config_file),  # pyright: ignore[reportUnknownMemberType]
        )
    config = BenchmarkConfig.model_validate(document)
    return config.model_copy(update={"output_directory": config.output_directory.resolve()})


# Model selections


def _validate_model_id(value: str) -> str:
    """Keep a user-defined model ID within one output-directory component."""
    windows_path = PureWindowsPath(value)
    if (
        not value
        or value in {".", ".."}
        or "\x00" in value
        or Path(value).name != value
        or windows_path.name != value
        or windows_path.drive
    ):
        raise ValueError("model id must be a single output-directory name")
    return value


ModelId = Annotated[str, AfterValidator(_validate_model_id)]


class AgentSettings(BaseModel):
    """How MAM-Bench calls PydanticAI agents."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    temperature: float = Field(default=1.0, ge=0, le=2)
    top_p: float = Field(default=0.95, gt=0, le=1)
    top_k: int = Field(default=20, ge=0)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] = "medium"
    max_completion_tokens: int = Field(default=32_768, gt=0)
    concurrency: int = Field(default=4, gt=0)
    use_sampling_seed: bool = True
    tool_choice: Literal["required", "auto"] = "required"
    timeout_seconds: float = Field(default=3_600.0, gt=0)
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


class OpenRouterModel(BaseModel):
    """One selected model through the built-in OpenRouter adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: ModelId
    runtime: Literal["openrouter"]
    model: str
    provider: str
    settings: AgentSettings = AgentSettings()


class OpenAICompatibleModel(BaseModel):
    """One selected model through the built-in OpenAI-compatible adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: ModelId
    runtime: Literal["openai-compatible"]
    model: str
    base_url: str
    api_key_env: str
    settings: AgentSettings = AgentSettings()


ModelSelection = Annotated[
    OpenRouterModel | OpenAICompatibleModel,
    Field(discriminator="runtime"),
]


# Benchmark configuration

SimulationSelection = str


class BenchmarkConfig(BaseModel):
    """Configuration for choosing which simulations and models to run."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    simulations: tuple[SimulationSelection, ...] = Field(min_length=1)
    models: tuple[ModelSelection, ...] = Field(min_length=1)
    output_directory: Path = Field(
        description=(
            "Results root relative to the working directory; each invocation creates a unique child"
        )
    )

    @model_validator(mode="after")
    def require_unique_selections(self) -> BenchmarkConfig:
        if len(set(self.simulations)) != len(self.simulations):
            raise ValueError("simulation selections must be unique")
        model_ids = tuple(model.id for model in self.models)
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("model selections must have unique ids")
        return self
