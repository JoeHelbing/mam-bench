"""Run one MAM-Bench YAML configuration."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from pathlib import Path

from mam_bench.benchmark import BenchmarkRunFailure, BenchmarkTopline
from mam_bench.config import load_benchmark_config
from mam_bench.diagnostics import configure_logging
from mam_bench.runner import run_benchmark


def format_topline(topline: BenchmarkTopline) -> str:
    """Format scores for the terminal."""

    return "\n".join(
        f"{entry.simulation_id} / {entry.model_id}: "
        f"{entry.primary_score.value:.6f} {entry.primary_score.unit}"
        for entry in topline.entries
    )


def report_output_directory(path: Path) -> None:
    """Report the attempt location before execution, including failed attempts."""

    print(f"Output directory: {path}", file=sys.stderr, flush=True)


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    config = load_benchmark_config(parser.parse_args(arguments).config)
    configure_logging(config.log_level)
    try:
        topline = asyncio.run(run_benchmark(config, on_output_directory=report_output_directory))
    except BenchmarkRunFailure as error:
        print(error, file=sys.stderr)
        return 1
    print(format_topline(topline))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
