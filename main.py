"""Run one MAM-Bench YAML configuration."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from mam_bench.benchmark import BenchmarkRunFailure, BenchmarkTopline
from mam_bench.config import load_benchmark_config
from mam_bench.runner import run_benchmark


def format_topline(topline: BenchmarkTopline) -> str:
    """Format scores for the terminal."""

    return "\n".join(
        f"{entry.simulation_id} / {entry.model_id}: "
        f"{entry.primary_score.value:.6f} {entry.primary_score.unit}"
        for entry in topline.entries
    )


def yaml_path(value: str) -> Path:
    path = Path(value)
    if path.suffix.lower() != ".yaml":
        raise argparse.ArgumentTypeError("benchmark configuration must be a .yaml file")
    return path


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=yaml_path)
    config = load_benchmark_config(parser.parse_args(arguments).config)
    try:
        topline = asyncio.run(run_benchmark(config))
    except BenchmarkRunFailure as error:
        print(error, file=sys.stderr)
        return 1
    print(format_topline(topline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
