"""Generic benchmark results and execution contracts."""

from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from mam_bench.model import ModelRuntime


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
    "artifact_write",
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
    """Signal that one pair failed without a score."""

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
