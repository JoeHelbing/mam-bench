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

    temperature: float | None = Field(default=None, ge=0, le=2)
    top_p: float | None = Field(default=None, gt=0, le=1)
    top_k: int | None = Field(default=None, ge=0)
    reasoning_effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    max_completion_tokens: int | None = Field(default=None, gt=0)
    concurrency: int = Field(default=4, gt=0)
    tool_choice: Literal["required", "auto"] | None = None
    strict_parameterless_tools: bool = Field(default=False, strict=True)
    timeout_seconds: float = Field(default=3_600.0, gt=0)
    compaction_trigger_fraction: float = Field(default=0.7, gt=0, lt=1)
    compaction_tail_tokens: int | None = Field(default=None, ge=0)
    summary_completion_tokens: int | None = Field(default=None, gt=0)
    memory_injection_tokens: int = Field(default=4_000, gt=0)

    def compaction_tail(self, context_window_tokens: int) -> int:
        """Resolve the retained history against the actual serving context."""
        trigger_tokens = int(context_window_tokens * self.compaction_trigger_fraction)
        tail = self.compaction_tail_tokens
        if tail is None:
            tail = min(40_000, trigger_tokens // 4)
        if tail >= trigger_tokens:
            raise ValueError("compaction_tail_tokens must be below the compaction trigger")
        return tail


class OpenRouterModel(BaseModel):
    """One selected model through the built-in OpenRouter adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: Literal["openrouter"]
    model: str = Field(min_length=1, pattern=r"^\S+/\S+$")
    provider: str = Field(min_length=1, pattern=r"^\S(?:.*\S)?$")
    settings: AgentSettings = AgentSettings()

    # Derived by preflight, never accepted as user configuration.
    _completion_token_parameter: Literal["max_tokens", "max_completion_tokens"] = PrivateAttr(
        default="max_tokens"
    )
    _required_tool_choice_supported: bool | None = PrivateAttr(default=None)

    @property
    def completion_token_parameter(self) -> Literal["max_tokens", "max_completion_tokens"]:
        return self._completion_token_parameter

    @property
    def required_tool_choice_supported(self) -> bool | None:
        return self._required_tool_choice_supported

    def with_setup(
        self,
        completion_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
        required_tool_choice_supported: bool | None = None,
    ) -> Self:
        """Attach verified runtime decisions without adding YAML configuration fields."""
        resolved = self.model_copy()
        resolved._completion_token_parameter = completion_token_parameter
        resolved._required_tool_choice_supported = required_tool_choice_supported
        return resolved


class RepetitionDetection(BaseModel):
    """Stop consecutive repeated token patterns on supporting vLLM endpoints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_pattern_size: int = Field(default=16, ge=1, strict=True)
    max_pattern_size: int = Field(default=128, ge=1, strict=True)
    min_count: int = Field(default=8, ge=2, strict=True)

    @model_validator(mode="after")
    def validate_pattern_sizes(self) -> Self:
        if self.min_pattern_size > self.max_pattern_size:
            raise ValueError("min_pattern_size must not exceed max_pattern_size")
        return self


class VLLMModel(BaseModel):
    """One selected model through PydanticAI's native vLLM provider."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    runtime: Literal["vllm"]
    model: str
    base_url: str
    api_key_env: str
    settings: AgentSettings = AgentSettings()
    repetition_detection: RepetitionDetection | None = None


ModelSelection = Annotated[
    OpenRouterModel | VLLMModel,
    Field(discriminator="runtime"),
]


# Both shipped and custom suites use the same discriminated case schema.
CaseSettings = Annotated[
    SchellingSettings | CivilViolenceSettings, Field(discriminator="simulation")
]


class BenchmarkConfig(BaseModel):
    """One explicit model and an ordered, fully specified suite."""

    model_config = ConfigDict(frozen=True, extra="forbid")
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
