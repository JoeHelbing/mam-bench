"""Explicit built-in Benchmark Simulation and Model Runtime registries."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from mam_bench.benchmark import BenchmarkSimulation, PreflightIssue
from mam_bench.config import BenchmarkConfig, OpenAICompatibleModel, OpenRouterModel
from mam_bench.pydantic_runtime import (
    PydanticRuntimeFactory,
    PydanticRuntimeSettings,
)
from mam_bench.runner import BenchmarkSelection
from mam_bench.runtime import ModelRuntimeFactory
from mam_bench.simulations.schelling import SchellingBenchmarkSimulation

BUILTIN_SIMULATIONS: Mapping[str, BenchmarkSimulation] = MappingProxyType(
    {
        "schelling-influence-pilot-v1": SchellingBenchmarkSimulation(),
    }
)


class RegistryPreflightError(ValueError):
    """Every unknown built-in selection found before preparation."""

    def __init__(self, issues: tuple[PreflightIssue, ...]) -> None:
        self.issues = issues
        super().__init__("benchmark registry preflight failed:\n" + _issue_text(issues))


@dataclass(frozen=True)
class ResolvedBenchmarkPlan:
    """Resolved built-ins ready for scientific Compatibility Preflight."""

    selection: BenchmarkSelection
    simulation_registry: Mapping[str, BenchmarkSimulation]
    runtime_factories: Mapping[str, ModelRuntimeFactory]

    @property
    def simulations(self) -> tuple[str, ...]:
        return self.selection.simulations

    @property
    def runtimes(self) -> tuple[str, ...]:
        return self.selection.runtimes


def resolve_benchmark_plan(
    config: BenchmarkConfig,
    *,
    simulations: Mapping[str, BenchmarkSimulation] = BUILTIN_SIMULATIONS,
) -> ResolvedBenchmarkPlan:
    """Resolve all selected built-ins without preparing or constructing runtimes."""

    issues = tuple(
        PreflightIssue(
            simulation_id=simulation_id,
            model_id=None,
            code="unknown-simulation",
            message=f"unknown built-in Benchmark Simulation: {simulation_id}",
        )
        for simulation_id in config.simulations
        if simulation_id not in simulations
    )
    if issues:
        raise RegistryPreflightError(issues)
    selected_simulations = {
        simulation_id: simulations[simulation_id] for simulation_id in config.simulations
    }
    runtime_factories = {model.id: _runtime_factory(model) for model in config.models}
    return ResolvedBenchmarkPlan(
        selection=BenchmarkSelection(
            simulations=config.simulations,
            runtimes=tuple(model.id for model in config.models),
            output_directory=config.output.directory,
            retain_diagnostic_artifacts=config.output.retain_diagnostic_artifacts,
        ),
        simulation_registry=selected_simulations,
        runtime_factories=runtime_factories,
    )


def _runtime_factory(
    model: OpenRouterModel | OpenAICompatibleModel,
) -> PydanticRuntimeFactory:
    if isinstance(model, OpenRouterModel):
        settings = PydanticRuntimeSettings(
            model_id=model.id,
            provider="openrouter",
            model_name=model.model,
            openrouter_provider_slug=model.provider,
        )
    else:
        settings = PydanticRuntimeSettings(
            model_id=model.id,
            provider="openai-compatible",
            base_url=model.base_url,
            model_name=model.model,
            credential_environment=model.api_key_env,
        )
    return PydanticRuntimeFactory(settings)


def _issue_text(issues: tuple[PreflightIssue, ...]) -> str:
    return "\n".join(f"[{issue.code}] {issue.simulation_id}: {issue.message}" for issue in issues)
