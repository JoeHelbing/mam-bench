"""Shared Benchmark Simulation results and interfaces."""

from pathlib import Path, PurePosixPath
from typing import Literal, Protocol, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from mam_bench.runtime import ModelRuntime, RuntimeDescriptor, RuntimeRequirements


class SimulationDescriptor(BaseModel):
    """Stable identity and topline contract for one Benchmark Simulation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    simulation_version: str = Field(min_length=1)
    title: str = Field(min_length=1)
    primary_score_name: str = Field(min_length=1)


class PreflightIssue(BaseModel):
    """One deterministic Compatibility Preflight failure."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_id: str
    model_id: str | None
    code: str
    message: str


class PrimaryScore(BaseModel):
    """A simulation-native, higher-is-better topline score."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    value: float = Field(allow_inf_nan=False)
    higher_is_better: Literal[True] = True
    objective: str = Field(min_length=1)
    meaning: str = Field(min_length=1)
    unit: str = Field(min_length=1)
    semantics_version: str = Field(min_length=1)


class EvidenceReceipt(BaseModel):
    """Location and hash of simulation-validated Run Evidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_directory: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @field_validator("relative_directory", "manifest_path")
    @classmethod
    def validate_relative_path(cls, value: str, info: ValidationInfo) -> str:
        del cls
        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("evidence paths must be relative and cannot escape the run")
        if value == "" or (value == "." and info.field_name == "manifest_path"):
            raise ValueError("evidence paths cannot be empty")
        return value


class SimulationResult(BaseModel):
    """Validated scored result returned by one Benchmark Simulation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation: SimulationDescriptor
    runtime: RuntimeDescriptor
    primary_score: PrimaryScore
    evidence: EvidenceReceipt
    diagnostic_artifacts_retained: bool

    @model_validator(mode="after")
    def validate_primary_score_name(self) -> Self:
        if self.primary_score.name != self.simulation.primary_score_name:
            raise ValueError("Primary Score name does not match the simulation descriptor")
        return self


class ToplineEntry(BaseModel):
    """One Benchmark Simulation-Model Runtime score in the topline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_id: str
    simulation_version: str
    model_id: str
    runtime: str
    provider: str
    model: str
    primary_score: PrimaryScore
    evidence: EvidenceReceipt


class BenchmarkTopline(BaseModel):
    """Complete non-aggregated benchmark results in deterministic order."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["mam-bench.topline.v1"] = "mam-bench.topline.v1"
    status: Literal["complete"] = "complete"
    entries: tuple[ToplineEntry, ...]


class PreparedSimulation(Protocol):
    """Model-independent prepared state for one Benchmark Simulation."""

    @property
    def descriptor(self) -> SimulationDescriptor: ...

    @property
    def runtime_requirements(self) -> RuntimeRequirements: ...

    async def execute(
        self,
        *,
        runtime: ModelRuntime,
        output_directory: Path,
        retain_diagnostic_artifacts: bool,
    ) -> None: ...


class BenchmarkSimulation(Protocol):
    """Deep simulation seam used by the benchmark runner."""

    @property
    def descriptor(self) -> SimulationDescriptor: ...

    def prepare(self) -> PreparedSimulation: ...

    def validate(
        self,
        *,
        output_directory: Path,
        runtime: RuntimeDescriptor,
    ) -> SimulationResult: ...
