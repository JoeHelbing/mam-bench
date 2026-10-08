"""Strict YAML configuration for benchmark runs."""

from pathlib import Path
from typing import Annotated, Literal, Self, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from mam_bench.simulations.civil_violence.settings import CivilViolenceSettings
from mam_bench.simulations.schelling.settings import SchellingSettings


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
    strict_parameterless_tools: bool = Field(default=False, strict=True)
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
        if (
            self.compaction_tail_tokens >= trigger_tokens
            and "compaction_tail_tokens" in self.model_fields_set
        ):
            raise ValueError("compaction_tail_tokens must be below the compaction trigger")
        return self


class OpenRouterModel(BaseModel):
    """One selected model through the built-in OpenRouter adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: Literal["openrouter"]
    model: str = Field(min_length=1, pattern=r"^\S+/\S+$")
    provider: str = Field(min_length=1, pattern=r"^\S(?:.*\S)?$")
    prompt_cache: Literal["auto", "required", "off"] = "auto"
    settings: AgentSettings = AgentSettings()

    # Derived by preflight, never accepted as user configuration.
    _omitted_parameters: frozenset[str] = PrivateAttr(default_factory=frozenset)
    _cache_strategy: Literal["explicit", "implicit", "unverified", "off"] = PrivateAttr(
        default="unverified"
    )

    _completion_token_parameter: Literal["max_tokens", "max_completion_tokens"] = PrivateAttr(
        default="max_tokens"
    )

    @property
    def completion_token_parameter(self) -> Literal["max_tokens", "max_completion_tokens"]:
        return self._completion_token_parameter

    @property
    def omitted_parameters(self) -> frozenset[str]:
        return self._omitted_parameters

    @property
    def cache_strategy(self) -> Literal["explicit", "implicit", "unverified", "off"]:
        return self._cache_strategy

    def with_setup(
        self,
        settings: AgentSettings,
        omitted_parameters: frozenset[str],
        cache_strategy: Literal["explicit", "implicit", "unverified", "off"],
        completion_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
    ) -> Self:
        """Attach verified runtime decisions without adding YAML configuration fields."""
        resolved = self.model_copy(update={"settings": settings})
        resolved._omitted_parameters = omitted_parameters
        resolved._cache_strategy = cache_strategy
        resolved._completion_token_parameter = completion_token_parameter
        return resolved


class OpenAICompatibleModel(BaseModel):
    """One selected model through the built-in OpenAI-compatible adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: Literal["openai-compatible"]
    model: str
    base_url: str
    api_key_env: str
    settings: AgentSettings = AgentSettings()

    @model_validator(mode="after")
    def validate_compaction_tail(self) -> Self:
        settings = self.settings
        if settings.compaction_tail_tokens >= (
            settings.context_window_tokens * settings.compaction_trigger_fraction
        ):
            raise ValueError("compaction_tail_tokens must be below the compaction trigger")
        return self


ModelSelection = Annotated[
    OpenRouterModel | OpenAICompatibleModel,
    Field(discriminator="runtime"),
]


# Both shipped and custom suites use the same discriminated case schema.
CaseSettings = Annotated[
    SchellingSettings | CivilViolenceSettings, Field(discriminator="simulation")
]


class BenchmarkConfig(BaseModel):
    """One explicit model and an ordered, fully specified suite."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)
    model: ModelSelection
    cases: tuple[CaseSettings, ...] = Field(min_length=1)
    output_directory: Path
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"


DEFAULT_SUITE = Path(__file__).with_name("default-suite.yaml")


def load_benchmark_config(
    model_path: Path, suite_path: Path = DEFAULT_SUITE, *, output_directory: Path
) -> BenchmarkConfig:
    """Validate every case before constructing a provider or creating output."""
    with model_path.open("rb") as source:
        model = cast(object, yaml.safe_load(source))  # pyright: ignore[reportUnknownMemberType]
    with suite_path.open("rb") as source:
        suite = cast(object, yaml.safe_load(source))  # pyright: ignore[reportUnknownMemberType]
    if not isinstance(suite, dict):
        raise ValueError("suite YAML must be a mapping containing cases")
    return BenchmarkConfig.model_validate(
        {
            **cast(dict[str, object], suite),
            "model": model,
            "output_directory": output_directory.resolve(),
        }
    )
