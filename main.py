"""Run one explicitly selected model against a fully specified benchmark suite."""

import argparse
import asyncio
import os
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from pydantic import ValidationError
from yaml import YAMLError

from mam_bench.config import DEFAULT_SUITE, load_benchmark_config
from mam_bench.diagnostics import configure_logging
from mam_bench.runner import BenchmarkResult, BenchmarkRunFailure, BenchmarkRunner, CaseResult
from mam_bench.telemetry import collect_telemetry


def format_case(index: int, result: CaseResult) -> str:
    return f"Case {index + 1}: {result.config.model_dump_json()} score={result.score:+.6f}"


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", required=True, type=Path, help="YAML selecting one language model"
    )
    parser.add_argument(
        "--suite", type=Path, default=DEFAULT_SUITE, help="complete replacement case list"
    )
    parser.add_argument("--output", type=Path, default=Path("results"))
    args = parser.parse_args(arguments)
    try:
        config = load_benchmark_config(args.model, args.suite, output_directory=args.output)
    except (ValidationError, ValueError, OSError, YAMLError) as error:
        # ValidationError hides input values; YAML parser errors can contain arbitrary input.
        print(
            str(error)
            if isinstance(error, ValidationError)
            else f"Configuration error ({type(error).__name__}).",
            file=sys.stderr,
        )
        return 2
    configure_logging(config.log_level)
    runner = BenchmarkRunner(config)

    async def run_benchmark() -> BenchmarkResult:
        # Collection and its trace context belong to the benchmark's async task.
        with ExitStack() as stack:

            def started(path: Path) -> None:
                stack.enter_context(
                    collect_telemetry(path, send_to_logfire=bool(os.environ.get("LOGFIRE_TOKEN")))
                )
                print(f"Output directory: {path}", file=sys.stderr, flush=True)

            return await runner.run(
                on_output_directory=started,
                on_case=lambda index, case: print(format_case(index, case), flush=True),
            )

    try:
        result = asyncio.run(run_benchmark())
    except BenchmarkRunFailure as error:
        if error.case_index is not None:
            print(
                f"Failed case: {config.cases[error.case_index].model_dump_json()}", file=sys.stderr
            )
        print(error, file=sys.stderr)
        return 1
    print(f"Model: {result.model.model}")
    print(f"Combined Benchmark Score: {result.total_score:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
