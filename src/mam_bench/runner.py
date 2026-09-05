"""Run every simulation and model selected in a benchmark YAML file."""

import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path

from mam_bench.benchmark import (
    BenchmarkRunFailure,
    BenchmarkSimulation,
    BenchmarkTopline,
    ModelRuntime,
    PairFailure,
    PairInfrastructureFailure,
    ToplineEntry,
    create_runtime,
)
from mam_bench.config import BenchmarkConfig, ModelSelection
from mam_bench.simulations.schelling.simulation import SchellingBenchmarkSimulation

SIMULATIONS: Mapping[str, BenchmarkSimulation] = {
    "schelling-influence-pilot-v1": SchellingBenchmarkSimulation(),
}
RuntimeBuilder = Callable[[ModelSelection], ModelRuntime]


async def run_benchmark(
    config: BenchmarkConfig,
    *,
    simulations: Mapping[str, BenchmarkSimulation] = SIMULATIONS,
    runtime_builder: RuntimeBuilder = create_runtime,
) -> BenchmarkTopline:
    """Run the selected simulation-model combinations and save their scores."""

    selected_simulations = tuple(simulations[simulation_id] for simulation_id in config.simulations)
    config.output_directory.mkdir(parents=True)

    entries: list[ToplineEntry] = []
    failures: list[PairFailure] = []
    for simulation in selected_simulations:
        for model in config.models:
            runtime = runtime_builder(model)
            output_directory = (
                config.output_directory / "runs" / simulation.simulation_id / runtime.info.model_id
            )
            try:
                score = await simulation.run(runtime, output_directory)
            except PairInfrastructureFailure as error:
                failures.append(error.failure)
                continue
            entries.append(
                ToplineEntry(
                    simulation_id=simulation.simulation_id,
                    simulation_version=simulation.simulation_version,
                    model_id=runtime.info.model_id,
                    provider=runtime.info.provider,
                    model=runtime.info.model,
                    primary_score=score,
                )
            )

    topline = BenchmarkTopline(entries=tuple(entries))
    _write_topline(config.output_directory / "topline.json", topline)
    if failures:
        raise BenchmarkRunFailure(tuple(failures))
    return topline


def _write_topline(path: Path, topline: BenchmarkTopline) -> None:
    """Atomically publish the completed matrix's success-only topline."""
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".topline.",
            suffix=".json",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary.write(f"{topline.model_dump_json(indent=2)}\n")
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
