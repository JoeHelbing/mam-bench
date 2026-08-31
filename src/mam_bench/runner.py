"""Compatibility Preflight and deterministic benchmark execution."""

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, field_validator

from mam_bench.benchmark import (
    BenchmarkSimulation,
    BenchmarkTopline,
    PreflightIssue,
    PreparedSimulation,
    SimulationResult,
    ToplineEntry,
)
from mam_bench.runtime import (
    ModelRuntimeFactory,
    RuntimeDescriptor,
    RuntimeRequirements,
)


class BenchmarkError(Exception):
    """Base class for benchmark configuration and execution failures."""


class BenchmarkConfigurationError(BenchmarkError, ValueError):
    """The benchmark selection cannot be resolved safely."""


class CompatibilityPreflightError(BenchmarkError):
    """One or more Benchmark Simulation-Model Runtime pairings are incompatible."""

    def __init__(self, issues: tuple[PreflightIssue, ...]) -> None:
        self.issues = issues
        super().__init__(f"{len(issues)} Benchmark Simulation-Model Runtime compatibility issue(s)")


class OutputConflictError(BenchmarkError, FileExistsError):
    """A planned benchmark output path already contains material."""


class EvidenceValidationError(BenchmarkError, ValueError):
    """A Benchmark Simulation did not return usable validated evidence."""


class BenchmarkSelection(BaseModel):
    """Already-parsed simulation, Model Runtime, and output selections."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    simulations: tuple[str, ...]
    runtimes: tuple[str, ...]
    output_directory: Path
    retain_diagnostic_artifacts: bool = False

    @field_validator("simulations", "runtimes")
    @classmethod
    def validate_unique_non_empty(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if not values:
            raise ValueError("benchmark selections cannot be empty")
        if len(set(values)) != len(values):
            raise ValueError("benchmark selections cannot contain duplicates")
        return values


@dataclass(frozen=True)
class PreparedPairing:
    """One resolved Benchmark Simulation-Model Runtime pairing."""

    simulation: BenchmarkSimulation
    prepared_simulation: PreparedSimulation
    runtime_factory: ModelRuntimeFactory
    relative_output_directory: Path


@dataclass(frozen=True)
class PreparedBenchmark:
    """Opaque complete plan produced only after full Compatibility Preflight."""

    selection: BenchmarkSelection
    pairings: tuple[PreparedPairing, ...]


def preflight_benchmark(
    selection: BenchmarkSelection,
    *,
    simulations: Mapping[str, BenchmarkSimulation],
    runtimes: Mapping[str, ModelRuntimeFactory],
    environment: Mapping[str, str] | None = None,
) -> PreparedBenchmark:
    """Resolve and validate the complete matrix without calls or output writes."""

    _validate_output_root(selection.output_directory)
    environment = os.environ if environment is None else environment
    selected_simulations: list[tuple[BenchmarkSimulation, PreparedSimulation]] = []
    for simulation_id in selection.simulations:
        try:
            simulation = simulations[simulation_id]
        except KeyError as exc:
            raise BenchmarkConfigurationError(
                f"unknown Benchmark Simulation: {simulation_id}"
            ) from exc
        if simulation.descriptor.simulation_id != simulation_id:
            raise BenchmarkConfigurationError(
                f"simulation registry key does not match descriptor: {simulation_id}"
            )
        prepared = simulation.prepare()
        if prepared.descriptor != simulation.descriptor:
            raise BenchmarkConfigurationError(
                f"prepared simulation descriptor changed: {simulation_id}"
            )
        selected_simulations.append((simulation, prepared))

    selected_runtimes: list[ModelRuntimeFactory] = []
    for runtime_id in selection.runtimes:
        try:
            factory = runtimes[runtime_id]
        except KeyError as exc:
            raise BenchmarkConfigurationError(
                f"unknown Model Runtime selection: {runtime_id}"
            ) from exc
        if factory.descriptor.model_id != runtime_id:
            raise BenchmarkConfigurationError(
                f"runtime registry key does not match descriptor: {runtime_id}"
            )
        selected_runtimes.append(factory)

    issues: list[PreflightIssue] = []
    pairings: list[PreparedPairing] = []
    for simulation, prepared in selected_simulations:
        for factory in selected_runtimes:
            descriptor = factory.descriptor
            requirements = prepared.runtime_requirements
            issues.extend(
                _compatibility_issues(
                    simulation_id=simulation.descriptor.simulation_id,
                    descriptor=descriptor,
                    requirements=requirements,
                    environment=environment,
                )
            )
            pairings.append(
                PreparedPairing(
                    simulation=simulation,
                    prepared_simulation=prepared,
                    runtime_factory=factory,
                    relative_output_directory=(
                        Path("runs") / simulation.descriptor.simulation_id / descriptor.model_id
                    ),
                )
            )
    if issues:
        raise CompatibilityPreflightError(tuple(issues))
    return PreparedBenchmark(selection=selection, pairings=tuple(pairings))


async def run_benchmark(prepared: PreparedBenchmark) -> BenchmarkTopline:
    """Execute every prepared pairing and publish one complete topline last."""

    output_root = prepared.selection.output_directory
    _validate_output_root(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    entries: list[ToplineEntry] = []
    for pairing in prepared.pairings:
        output_directory = output_root / pairing.relative_output_directory
        if output_directory.exists() and any(output_directory.iterdir()):
            raise OutputConflictError(
                f"run output directory must be new or empty: {output_directory}"
            )
        output_directory.mkdir(parents=True, exist_ok=True)
        runtime = pairing.runtime_factory.create()
        if runtime.descriptor != pairing.runtime_factory.descriptor:
            raise BenchmarkConfigurationError(
                "Model Runtime descriptor changed after Compatibility Preflight"
            )
        await pairing.prepared_simulation.execute(
            runtime=runtime,
            output_directory=output_directory,
            retain_diagnostic_artifacts=(prepared.selection.retain_diagnostic_artifacts),
        )
        result = pairing.simulation.validate(
            output_directory=output_directory,
            runtime=runtime.descriptor,
        )
        result = _validate_result(pairing, result, output_root, output_directory)
        entries.append(_topline_entry(result))

    topline = BenchmarkTopline(entries=tuple(entries))
    _atomic_write_text(
        output_root / "topline.json",
        f"{topline.model_dump_json(indent=2)}\n",
    )
    return topline


def _compatibility_issues(
    *,
    simulation_id: str,
    descriptor: RuntimeDescriptor,
    requirements: RuntimeRequirements,
    environment: Mapping[str, str],
) -> tuple[PreflightIssue, ...]:
    required = requirements.capabilities
    required_sampling = requirements.sampling_controls
    minimum_concurrency = requirements.minimum_concurrent_requests
    issues: list[PreflightIssue] = []
    for capability in sorted(
        required - descriptor.capabilities,
        key=lambda value: value.value,
    ):
        issues.append(
            PreflightIssue(
                simulation_id=simulation_id,
                model_id=descriptor.model_id,
                code=f"missing-capability:{capability.value}",
                message=(f"{descriptor.model_id} lacks required capability {capability.value}"),
            )
        )
    for control in sorted(
        required_sampling - descriptor.sampling_controls,
        key=lambda value: value.value,
    ):
        issues.append(
            PreflightIssue(
                simulation_id=simulation_id,
                model_id=descriptor.model_id,
                code=f"missing-sampling-control:{control.value}",
                message=(
                    f"{descriptor.model_id} cannot translate required sampling "
                    f"control {control.value}"
                ),
            )
        )
    if descriptor.max_concurrent_requests < minimum_concurrency:
        issues.append(
            PreflightIssue(
                simulation_id=simulation_id,
                model_id=descriptor.model_id,
                code="insufficient-concurrency",
                message=(
                    f"{descriptor.model_id} admits "
                    f"{descriptor.max_concurrent_requests} concurrent request(s); "
                    f"{minimum_concurrency} required"
                ),
            )
        )
    for environment_name in sorted(descriptor.required_environment):
        if environment_name not in environment:
            issues.append(
                PreflightIssue(
                    simulation_id=simulation_id,
                    model_id=descriptor.model_id,
                    code=f"missing-environment:{environment_name}",
                    message=(
                        f"{descriptor.model_id} requires environment variable {environment_name}"
                    ),
                )
            )
    return tuple(issues)


def _validate_output_root(output_root: Path) -> None:
    if output_root.exists():
        if not output_root.is_dir():
            raise OutputConflictError(f"benchmark output is not a directory: {output_root}")
        if any(output_root.iterdir()):
            raise OutputConflictError(
                f"benchmark output directory must be new or empty: {output_root}"
            )


def _validate_result(
    pairing: PreparedPairing,
    result: SimulationResult,
    output_root: Path,
    output_directory: Path,
) -> SimulationResult:
    if result.simulation != pairing.simulation.descriptor:
        raise EvidenceValidationError(
            "validated result simulation does not match the prepared simulation"
        )
    if result.runtime != pairing.runtime_factory.descriptor:
        raise EvidenceValidationError(
            "validated result Model Runtime does not match the prepared Model Runtime"
        )
    manifest_path = (output_directory / result.evidence.manifest_path).resolve()
    run_root = output_directory.resolve()
    if not manifest_path.is_relative_to(run_root):
        raise EvidenceValidationError("evidence manifest escapes the run directory")
    if not manifest_path.is_file():
        raise EvidenceValidationError(
            f"evidence manifest does not exist: {result.evidence.manifest_path}"
        )
    actual_hash = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_hash != result.evidence.manifest_sha256:
        raise EvidenceValidationError("evidence manifest hash does not match")
    relative_directory = output_directory.relative_to(output_root).as_posix()
    return result.model_copy(
        update={
            "evidence": result.evidence.model_copy(
                update={"relative_directory": relative_directory}
            )
        }
    )


def _topline_entry(result: SimulationResult) -> ToplineEntry:
    return ToplineEntry(
        simulation_id=result.simulation.simulation_id,
        simulation_version=result.simulation.simulation_version,
        model_id=result.runtime.model_id,
        runtime=result.runtime.runtime,
        provider=result.runtime.provider,
        model=result.runtime.model,
        primary_score=result.primary_score,
        evidence=result.evidence,
    )


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
