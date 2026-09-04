"""Strict selection-only YAML configuration for benchmark runs."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

_SAFE_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
_ENVIRONMENT_NAME_PATTERN = r"^[A-Z_][A-Z0-9_]*$"


class OpenRouterModel(BaseModel):
    """One selected model through the built-in OpenRouter adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=_SAFE_ID_PATTERN)
    runtime: Literal["openrouter"]
    model: str = Field(min_length=1)
    provider: str = Field(pattern=_SAFE_ID_PATTERN)


class OpenAICompatibleModel(BaseModel):
    """One selected model through the built-in OpenAI-compatible adapter."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=_SAFE_ID_PATTERN)
    runtime: Literal["openai-compatible"]
    model: str = Field(min_length=1)
    base_url: str
    api_key_env: str = Field(pattern=_ENVIRONMENT_NAME_PATTERN)

    @field_validator("base_url")
    @classmethod
    def validate_base_url(cls, value: str) -> str:
        """Allow only credential-free HTTP(S) endpoints."""

        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("base_url must not contain a query or fragment")
        return value.rstrip("/")


ModelSelection = Annotated[
    OpenRouterModel | OpenAICompatibleModel,
    Field(discriminator="runtime"),
]
SimulationSelection = Annotated[str, Field(pattern=_SAFE_ID_PATTERN)]


class BenchmarkConfig(BaseModel):
    """Configuration for choosing which simulations and models to run."""

    model_config = ConfigDict(frozen=True, extra="forbid", hide_input_in_errors=True)

    simulations: tuple[SimulationSelection, ...] = Field(min_length=1)
    models: tuple[ModelSelection, ...] = Field(min_length=1)
    output_directory: Path

    @model_validator(mode="after")
    def require_unique_selections(self) -> BenchmarkConfig:
        if len(set(self.simulations)) != len(self.simulations):
            raise ValueError("simulation selections must be unique")
        model_ids = tuple(model.id for model in self.models)
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("model selections must have unique ids")
        return self


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load one YAML document without constructing runtime objects."""

    with path.open("rb") as config_file:
        document = yaml.safe_load(config_file)
    config = BenchmarkConfig.model_validate(document)
    output_directory = config.output_directory
    if not output_directory.is_absolute():
        output_directory = path.resolve().parent / output_directory
    return config.model_copy(update={"output_directory": output_directory.resolve()})
