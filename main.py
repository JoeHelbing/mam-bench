"""Run one MAM-Bench YAML configuration."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml
from pydantic import ValidationError

from mam_bench.benchmark import BenchmarkTopline
from mam_bench.config import load_benchmark_config
from mam_bench.registry import resolve_benchmark_plan
from mam_bench.runner import BenchmarkError, preflight_benchmark, run_benchmark
from mam_bench.runtime import RuntimeInfrastructureError
from mam_bench.simulations.schelling.evaluation import InfluenceInfrastructureError


def format_topline(topline: BenchmarkTopline) -> str:
    """Format simulation-local scores without cross-simulation aggregation."""

    lines: list[str] = []
    current_simulation: str | None = None
    for entry in topline.entries:
        if entry.simulation_id != current_simulation:
            if lines:
                lines.append("")
            lines.append(entry.simulation_id)
            current_simulation = entry.simulation_id
        lines.append(
            f"  {entry.model_id}  {entry.primary_score.name} = "
            f"{entry.primary_score.value:.6f} {entry.primary_score.unit}"
        )
    return "\n".join(lines)


def yaml_path(value: str) -> Path:
    """Parse a benchmark configuration path with the required suffix."""

    path = Path(value)
    if path.suffix.lower() != ".yaml":
        raise argparse.ArgumentTypeError("benchmark configuration must be a .yaml file")
    return path


def main(arguments: Sequence[str] | None = None) -> int:
    """Validate and run the single YAML path supplied by the user."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=yaml_path, help="path to a benchmark YAML file")
    args = parser.parse_args(arguments)

    try:
        config = load_benchmark_config(args.config)
        resolved = resolve_benchmark_plan(config)
        prepared = preflight_benchmark(
            resolved.selection,
            simulations=resolved.simulation_registry,
            runtimes=resolved.runtime_factories,
        )
        topline = asyncio.run(run_benchmark(prepared))
    except (
        BenchmarkError,
        InfluenceInfrastructureError,
        OSError,
        RuntimeInfrastructureError,
        ValidationError,
        ValueError,
        yaml.YAMLError,
    ) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2

    print(format_topline(topline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
