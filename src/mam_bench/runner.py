"""Run every simulation and model selected in a benchmark YAML file."""

from collections.abc import Callable, Mapping

from mam_bench.benchmark import (
    BenchmarkSimulation,
    BenchmarkTopline,
    ModelRuntime,
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
    for simulation in selected_simulations:
        for model in config.models:
            runtime = runtime_builder(model)
            output_directory = (
                config.output_directory / "runs" / simulation.simulation_id / runtime.info.model_id
            )
            score = await simulation.run(runtime, output_directory)
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
    (config.output_directory / "topline.json").write_text(
        f"{topline.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    return topline
