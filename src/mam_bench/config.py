"""Strict selection-only YAML configuration for benchmark runs."""

from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator
from pathlib import Path
from typing import Annotated, Any, Literal, Protocol, cast
from urllib.parse import urlsplit

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
from yaml import parse as parse_yaml_untyped  # pyright: ignore[reportUnknownVariableType]
from yaml.events import (
    AliasEvent,
    CollectionStartEvent,
    DocumentStartEvent,
    Event,
    NodeEvent,
    ScalarEvent,
)
from yaml.nodes import MappingNode, Node

_SAFE_ID_PATTERN = r"^[a-z0-9][a-z0-9._-]*$"
_ENVIRONMENT_NAME_PATTERN = r"^[A-Z_][A-Z0-9_]*$"


class BenchmarkConfigError(ValueError):
    """One or more fail-closed YAML configuration errors."""

    def __init__(self, errors: tuple[str, ...]) -> None:
        self.errors = errors
        super().__init__("benchmark configuration failed:\n" + "\n".join(errors))


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


class OutputSelection(BaseModel):
    """Cross-simulation output policy only."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    directory: Path
    retain_diagnostic_artifacts: bool = False


class BenchmarkConfig(BaseModel):
    """Closed YAML v1 contract selecting one Cartesian benchmark matrix."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["mam-bench.run.v1"]
    simulations: tuple[SimulationSelection, ...] = Field(min_length=1)
    models: tuple[ModelSelection, ...] = Field(min_length=1)
    output: OutputSelection

    @model_validator(mode="after")
    def require_unique_selections(self) -> BenchmarkConfig:
        if len(set(self.simulations)) != len(self.simulations):
            raise ValueError("simulation selections must be unique")
        model_ids = tuple(model.id for model in self.models)
        if len(set(model_ids)) != len(model_ids):
            raise ValueError("model selections must have unique ids")
        return self


class _StrictSafeLoader(yaml.SafeLoader):
    """SafeLoader with duplicate and merge-key rejection."""


class _ObjectConstructor(Protocol):
    def construct_object(self, node: Node, deep: bool = False) -> object: ...


def _construct_unique_mapping(
    loader: _StrictSafeLoader,
    node: MappingNode,
    deep: bool = False,
) -> dict[Hashable, Any]:
    mapping: dict[Hashable, Any] = {}
    items = cast(list[tuple[Node, Node]], node.value)
    construct_object = cast(_ObjectConstructor, loader).construct_object
    for key_node, value_node in items:
        if key_node.tag == "tag:yaml.org,2002:merge":
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "YAML merge keys are not allowed",
                key_node.start_mark,
            )
        key = construct_object(key_node, deep)
        if not isinstance(key, Hashable):
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                "mapping keys must be scalar",
                key_node.start_mark,
            )
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key is not allowed: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = construct_object(value_node, deep)
    return mapping


_StrictSafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def _reject_unsafe_yaml_features(content: str) -> None:
    document_count = 0
    parse_yaml = cast(Callable[..., Iterator[Event]], parse_yaml_untyped)
    events = parse_yaml(content, Loader=yaml.SafeLoader)
    for event in events:
        if isinstance(event, DocumentStartEvent):
            document_count += 1
            if document_count > 1:
                raise yaml.YAMLError("multiple YAML documents are not allowed")
        if isinstance(event, AliasEvent):
            raise yaml.YAMLError("YAML aliases are not allowed")
        if isinstance(event, NodeEvent) and event.anchor is not None:
            raise yaml.YAMLError("YAML anchors are not allowed")
        if (
            isinstance(event, (CollectionStartEvent, ScalarEvent))
            and event.tag is not None
            and not event.tag.startswith("tag:yaml.org,2002:")
        ):
            raise yaml.YAMLError("custom YAML tags are not allowed")


def _validation_errors(error: ValidationError) -> tuple[str, ...]:
    messages: list[str] = []
    for detail in error.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in detail["loc"])
        messages.append(f"{location}: {detail['msg']}")
    return tuple(messages)


def load_benchmark_config(path: Path) -> BenchmarkConfig:
    """Load one strict YAML document without constructing runtime objects."""

    try:
        content = path.read_text(encoding="utf-8")
        _reject_unsafe_yaml_features(content)
        document = yaml.load(content, Loader=_StrictSafeLoader)
        config = BenchmarkConfig.model_validate(document)
        directory = config.output.directory
        if not directory.is_absolute():
            directory = path.resolve().parent / directory
        directory = directory.resolve()
        if directory.exists() and (not directory.is_dir() or any(directory.iterdir())):
            raise ValueError("output.directory must be new or empty")
        return config.model_copy(
            update={"output": config.output.model_copy(update={"directory": directory})}
        )
    except BenchmarkConfigError:
        raise
    except ValidationError as exc:
        raise BenchmarkConfigError(_validation_errors(exc)) from exc
    except (OSError, UnicodeError, yaml.YAMLError, ValueError) as exc:
        raise BenchmarkConfigError((str(exc),)) from exc
