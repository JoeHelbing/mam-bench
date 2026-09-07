"""Run every simulation and model selected in a benchmark YAML file."""

import logging
import tempfile
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from time import monotonic

from mam_bench.benchmark import (
    BenchmarkRunFailure,
    BenchmarkSimulation,
    BenchmarkTopline,
    PairFailure,
    PairInfrastructureFailure,
    ToplineEntry,
)
from mam_bench.config import BenchmarkConfig, ModelSelection
from mam_bench.model import ModelRuntime, create_runtime
from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim
from mam_bench.simulations.schelling.simulation import SchellingSim

logger = logging.getLogger(__name__)

SIMULATIONS = (
    "schelling-influence-pilot-v1",
    "civil-violence-citizens-v1",
    "civil-violence-police-v1",
)
RuntimeBuilder = Callable[[ModelSelection], ModelRuntime]


async def run_benchmark(
    config: BenchmarkConfig,
    *,
    simulations: Mapping[str, BenchmarkSimulation] | None = None,
    runtime_builder: RuntimeBuilder = create_runtime,
    on_output_directory: Callable[[Path], None] | None = None,
) -> BenchmarkTopline:
    """Run a fresh matrix in a unique child of the configured output root.

    Notify the caller of the allocated directory before starting simulations,
    so its output remains discoverable even if execution fails.
    """

    available = SIMULATIONS if simulations is None else simulations
    unknown = tuple(name for name in config.simulations if name not in available)
    if unknown:
        raise ValueError(
            f"Unknown simulations: {', '.join(map(repr, unknown))}. "
            f"Available simulations: {', '.join(sorted(available)) or '(none)'}"
        )
    if simulations is None:
        simulations = {
            "schelling-influence-pilot-v1": SchellingSim(settings=config.schelling),
            "civil-violence-citizens-v1": CivilViolenceSim(config.civil_violence_citizens),
            "civil-violence-police-v1": CivilViolenceSim(config.civil_violence_police, "police"),
        }
    selected_simulations = tuple(simulations[simulation_id] for simulation_id in config.simulations)

    runtimes: list[ModelRuntime] = []
    for model in config.models:
        try:
            runtime = runtime_builder(model)
        except Exception as error:
            # Provider exceptions can include credentials or response bodies.
            raise ValueError(
                f"Runtime initialization failed for model {model.id!r} "
                f"({type(error).__name__}); check credentials, endpoint, model and provider."
            ) from None
        runtimes.append(runtime)
    config.output_directory.mkdir(parents=True, exist_ok=True)
    # mkdtemp reserves the persistent attempt directory atomically and retries collisions.
    attempt_directory = Path(
        tempfile.mkdtemp(
            prefix=f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-",
            dir=config.output_directory,
        )
    )
    if on_output_directory is not None:
        on_output_directory(attempt_directory)

    started = monotonic()
    logger.info(
        "matrix.start directory=%s pairs=%d",
        attempt_directory,
        len(selected_simulations) * len(runtimes),
    )
    entries: list[ToplineEntry] = []
    failures: list[PairFailure] = []
    for simulation in selected_simulations:
        for runtime in runtimes:
            output_directory = (
                attempt_directory / "runs" / simulation.simulation_id / runtime.info.model_id
            )
            pair_started = monotonic()
            logger.info(
                "pair.start simulation=%s model=%s", simulation.simulation_id, runtime.info.model_id
            )
            try:
                score = await simulation.run(runtime, output_directory)
            except PairInfrastructureFailure as error:
                logger.error(
                    "pair.failed simulation=%s model=%s kind=%s elapsed_s=%.3f",
                    simulation.simulation_id,
                    runtime.info.model_id,
                    error.failure.kind,
                    monotonic() - pair_started,
                )
                failures.append(error.failure)
                continue
            logger.info(
                "pair.end simulation=%s model=%s score=%s elapsed_s=%.3f",
                simulation.simulation_id,
                runtime.info.model_id,
                score.value,
                monotonic() - pair_started,
            )
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
    _write_topline(attempt_directory / "topline.json", topline)
    logger.info(
        "matrix.end successes=%d failures=%d elapsed_s=%.3f",
        len(entries),
        len(failures),
        monotonic() - started,
    )
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
