"""Provider-neutral Model Runtime interface for Benchmark Simulations."""

from enum import StrEnum
from typing import Annotated, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class RuntimeCapability(StrEnum):
    """Model Runtime behavior a Model Interaction Protocol may require."""

    TEXT_OUTPUT = "text-output"
    STRICT_STRUCTURED_OUTPUT = "strict-structured-output"
    TOOL_RESULT_CONTINUATION = "tool-result-continuation"
    REQUEST_SEED = "request-seed"
    AUDITABLE_MESSAGES = "auditable-messages"


class SamplingControl(StrEnum):
    """Sampling controls a Model Runtime can translate without approximation."""

    TEMPERATURE = "temperature"
    TOP_P = "top-p"
    TOP_K = "top-k"
    REASONING_EFFORT = "reasoning-effort"
    OUTPUT_LIMIT = "max-output-tokens"


class RuntimeDescriptor(BaseModel):
    """Provider-independent facts available during Compatibility Preflight."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    runtime: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    model: str = Field(min_length=1)
    adapter_version: str = Field(min_length=1)
    capabilities: frozenset[RuntimeCapability]
    sampling_controls: frozenset[SamplingControl]
    max_concurrent_requests: int = Field(ge=1)
    endpoint: str | None = None
    routing_provider: str | None = None
    allow_fallbacks: bool | None = None
    require_parameters: bool | None = None
    required_environment: tuple[str, ...] = ()


class RuntimeRequirements(BaseModel):
    """Model Runtime behavior required by one prepared Benchmark Simulation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: frozenset[RuntimeCapability]
    sampling_controls: frozenset[SamplingControl]
    minimum_concurrent_requests: int = Field(ge=1)


class TextPart(BaseModel):
    """Text in a provider-neutral model message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["text"] = "text"
    text: str


class RuntimeReasoningPart(BaseModel):
    """Auditable provider reasoning returned before an answer or tool call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["reasoning"] = "reasoning"
    text: str
    id: str | None = None
    signature: str | None = None
    provider_name: str | None = None
    provider_details: dict[str, JsonValue] | None = None


class ToolCallPart(BaseModel):
    """One provider-neutral model tool call."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["tool-call"] = "tool-call"
    call_id: str
    name: str
    arguments: dict[str, JsonValue]


class ToolResultPart(BaseModel):
    """One provider-neutral tool result."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["tool-result"] = "tool-result"
    call_id: str
    name: str
    result: JsonValue


type RuntimePart = Annotated[
    TextPart | RuntimeReasoningPart | ToolCallPart | ToolResultPart,
    Field(discriminator="kind"),
]


class RuntimeMessage(BaseModel):
    """One auditable provider-neutral message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    parts: tuple[RuntimePart, ...]


class ToolDefinition(BaseModel):
    """A simulation-owned tool contract translated by a Model Runtime."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    input_schema: dict[str, JsonValue]
    strict: bool = True


class TextOutput(BaseModel):
    """Free-text model output."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["text"] = "text"


class StructuredOutput(BaseModel):
    """Strict structured model output owned by a Model Interaction Protocol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["structured"] = "structured"
    name: str
    description: str
    json_schema: dict[str, JsonValue]
    strict: bool = True


type OutputContract = Annotated[
    TextOutput | StructuredOutput,
    Field(discriminator="kind"),
]


class SamplingSettings(BaseModel):
    """Simulation-owned sampling controls for one model request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float = Field(ge=0.0)
    top_p: float = Field(gt=0.0, le=1.0)
    top_k: int = Field(ge=0)
    reasoning_effort: str
    max_output_tokens: int = Field(gt=0)
    request_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class RequestLimits(BaseModel):
    """Operational limits for one Model Runtime request."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_limit: int = Field(ge=1)
    tool_call_limit: int = Field(ge=0)
    timeout_seconds: float = Field(gt=0.0)


class ModelRequest(BaseModel):
    """One provider-neutral request assembled by a Model Interaction Protocol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    messages: tuple[RuntimeMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()
    output: OutputContract
    sampling: SamplingSettings
    limits: RequestLimits


class ModelUsage(BaseModel):
    """Provider-reported usage for one runtime operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    requests: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ModelResponse(BaseModel):
    """Auditable provider-neutral result of one Model Runtime operation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    messages: tuple[RuntimeMessage, ...]
    output: JsonValue
    usage: ModelUsage
    latency_seconds: float = Field(ge=0.0)
    provider_response_id: str | None = None


class RuntimeInfrastructureError(RuntimeError):
    """A transport, timeout, provider, or remote-service failure."""


class RuntimeProtocolError(RuntimeError):
    """A Model Runtime response violated its repository-owned contract."""


class ModelRuntime(Protocol):
    """Provider execution used by Benchmark Simulation implementations."""

    @property
    def descriptor(self) -> RuntimeDescriptor: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...


class ModelRuntimeFactory(Protocol):
    """Create a fresh Model Runtime for one Benchmark Simulation pairing."""

    @property
    def descriptor(self) -> RuntimeDescriptor: ...

    def create(self) -> ModelRuntime: ...
