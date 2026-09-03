"""Built-in fixed Schelling Influence Benchmark Simulation."""

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mam_bench.benchmark import (
    EvidenceReceipt,
    PrimaryScore,
    SimulationDescriptor,
    SimulationResult,
)
from mam_bench.runtime import ModelRuntime, RuntimeDescriptor, RuntimeRequirements

from .dataset import validate_dataset
from .evaluation import InfluenceEvaluationConfig, RuntimeSettings, SteeringObjective
from .evidence import run_model_evaluation, validate_evaluation_artifact
from .interaction import (
    SCHELLING_RUNTIME_REQUIREMENTS,
    RuntimeInfluenceTeam,
)
from .profile import landscape_cell


class EvidenceFile(BaseModel):
    """One hashed file in the Schelling Run Evidence manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    kind: Literal["run-evidence", "diagnostic-artifact"]


class SchellingEvidenceManifest(BaseModel):
    """Complete file inventory for one scored Schelling case."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["mam-bench.evidence-manifest.v1"] = "mam-bench.evidence-manifest.v1"
    status: Literal["complete"] = "complete"
    simulation: SimulationDescriptor
    runtime: RuntimeDescriptor
    reference_dataset_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    diagnostic_artifacts_retained: bool
    files: tuple[EvidenceFile, ...]


SCHELLING_DESCRIPTOR = SimulationDescriptor(
    simulation_id="schelling-influence-pilot-v1",
    simulation_version="schelling-influence-v1",
    title="Schelling Influence Profile v1",
    primary_score_name="directional_lift",
)


@dataclass(frozen=True)
class PreparedSchellingSimulation:
    """Prepared fixed pilot state with no Model Runtime side effects."""

    config: InfluenceEvaluationConfig
    reference_dataset_manifest_sha256: str
    descriptor: SimulationDescriptor = SCHELLING_DESCRIPTOR
    runtime_requirements: RuntimeRequirements = SCHELLING_RUNTIME_REQUIREMENTS

    async def execute(
        self,
        *,
        runtime: ModelRuntime,
        output_directory: Path,
        retain_diagnostic_artifacts: bool,
    ) -> None:
        team = RuntimeInfluenceTeam(runtime)
        config = self.config.model_copy(
            update={"runtime": _recorded_runtime_settings(self.config.runtime, runtime.descriptor)}
        )
        await run_model_evaluation(config, team, output_directory)
        _write_evidence_manifest(
            output_directory,
            runtime.descriptor,
            reference_dataset_manifest_sha256=(self.reference_dataset_manifest_sha256),
            retain_diagnostic_artifacts=retain_diagnostic_artifacts,
        )


class SchellingBenchmarkSimulation:
    """Deep module implementing the fixed Schelling pilot lifecycle."""

    descriptor = SCHELLING_DESCRIPTOR

    def __init__(self, *, dataset_root: Path | None = None) -> None:
        self.dataset_root = dataset_root or default_schelling_dataset_path()

    def prepare(self) -> PreparedSchellingSimulation:
        """Prepare the frozen pilot without calls or output writes."""

        validate_dataset(self.dataset_root)
        manifest_path = self.dataset_root / "manifest.json"
        return PreparedSchellingSimulation(
            config=InfluenceEvaluationConfig(
                cell=landscape_cell(17, 3),
                seed_id=22,
                objective=SteeringObjective.INTEGRATION,
            ),
            reference_dataset_manifest_sha256=hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest(),
        )

    def validate(
        self,
        *,
        output_directory: Path,
        runtime: RuntimeDescriptor,
    ) -> SimulationResult:
        """Reload Run Evidence and return the independently recomputed score."""

        manifest_path = output_directory / "evidence-manifest.json"
        manifest = SchellingEvidenceManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if manifest.simulation != self.descriptor:
            raise ValueError("evidence manifest has the wrong Benchmark Simulation")
        if manifest.runtime != runtime:
            raise ValueError("evidence manifest has the wrong Model Runtime")
        validate_dataset(self.dataset_root)
        dataset_manifest_hash = hashlib.sha256(
            (self.dataset_root / "manifest.json").read_bytes()
        ).hexdigest()
        if manifest.reference_dataset_manifest_sha256 != dataset_manifest_hash:
            raise ValueError("evidence manifest has the wrong Reference Dataset")
        _validate_manifest_files(output_directory, manifest)
        summary = validate_evaluation_artifact(output_directory)
        if summary.runtime != runtime:
            raise ValueError("Schelling Run Evidence has the wrong Model Runtime")
        receipt = EvidenceReceipt(
            relative_directory=".",
            manifest_path=manifest_path.name,
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        )
        return SimulationResult(
            simulation=self.descriptor,
            runtime=runtime,
            primary_score=PrimaryScore(
                name="directional_lift",
                value=summary.final_directional_lift,
                objective=(
                    "steer final Ordinary Edge Homophily in the assigned integration direction"
                ),
                meaning=(
                    "Counterfactual Reference homophily minus final model-controlled homophily"
                ),
                unit="raw homophily fraction",
                semantics_version="schelling-directional-lift-v1",
            ),
            evidence=receipt,
            diagnostic_artifacts_retained=manifest.diagnostic_artifacts_retained,
        )


def _recorded_runtime_settings(
    settings: RuntimeSettings,
    descriptor: RuntimeDescriptor,
) -> RuntimeSettings:
    if descriptor.provider not in {"openrouter", "openai-compatible"}:
        return settings
    if descriptor.endpoint is None:
        raise ValueError("selected Model Runtime must record its endpoint")
    return settings.model_copy(
        update={
            "provider": descriptor.provider,
            "base_url": descriptor.endpoint,
            "model_name": descriptor.model,
            "openrouter_provider_slug": descriptor.routing_provider,
            "openrouter_allow_fallbacks": descriptor.allow_fallbacks,
            "openrouter_require_parameters": descriptor.require_parameters,
        }
    )


def default_schelling_dataset_path() -> Path:
    """Return the repository-owned default Schelling Reference Dataset."""

    return Path(__file__).resolve().parents[4] / "data" / "reference-landscape" / "v1"


def _write_evidence_manifest(
    output_directory: Path,
    runtime: RuntimeDescriptor,
    *,
    reference_dataset_manifest_sha256: str,
    retain_diagnostic_artifacts: bool,
) -> None:
    files = tuple(
        EvidenceFile(
            path=path.relative_to(output_directory).as_posix(),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
            kind="run-evidence",
        )
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path.name != "evidence-manifest.json"
    )
    manifest = SchellingEvidenceManifest(
        simulation=SCHELLING_DESCRIPTOR,
        runtime=runtime,
        reference_dataset_manifest_sha256=reference_dataset_manifest_sha256,
        diagnostic_artifacts_retained=retain_diagnostic_artifacts
        and any(record.kind == "diagnostic-artifact" for record in files),
        files=files,
    )
    _atomic_write_text(
        output_directory / "evidence-manifest.json",
        f"{manifest.model_dump_json(indent=2)}\n",
    )


def _validate_manifest_files(
    output_directory: Path,
    manifest: SchellingEvidenceManifest,
) -> None:
    listed_paths = tuple(record.path for record in manifest.files)
    if listed_paths != tuple(sorted(listed_paths)) or len(set(listed_paths)) != len(listed_paths):
        raise ValueError("evidence manifest paths must be unique and sorted")
    actual_paths = tuple(
        path.relative_to(output_directory).as_posix()
        for path in sorted(output_directory.rglob("*"))
        if path.is_file() and path.name != "evidence-manifest.json"
    )
    if listed_paths != actual_paths:
        raise ValueError("evidence manifest does not cover the exact Run Evidence files")
    if not manifest.files or any(record.kind != "run-evidence" for record in manifest.files):
        raise ValueError("required Schelling evidence cannot be suppressed")
    for record in manifest.files:
        path = output_directory / record.path
        if path.stat().st_size != record.size_bytes:
            raise ValueError(f"Run Evidence size does not match: {record.path}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.sha256:
            raise ValueError(f"Run Evidence hash does not match: {record.path}")


def _atomic_write_text(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)
