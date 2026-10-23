"""Run one explicitly selected model against a fully specified benchmark suite."""

import argparse
import asyncio
import sys
from collections.abc import Sequence
from contextlib import ExitStack
from pathlib import Path

from yaml import YAMLError

from mam_bench.config import DEFAULT_SUITE, load_benchmark_config
from mam_bench.diagnostics import configure_logging
from mam_bench.runner import (
    BenchmarkResult,
    BenchmarkRunFailure,
    BenchmarkRunner,
    CaseResult,
    load_incomplete_run,
)
from mam_bench.telemetry import collect_telemetry


def format_case(index: int, result: CaseResult) -> str:
    return (
        f"Case {index + 1}: {result.config.model_dump_json()} score={result.score:+.6f}"
        f" null={result.null_score:+.6f} adjusted={result.adjusted_score:+.6f}"
    )


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--model", type=Path, help="YAML selecting one language model")
    source.add_argument(
        "--resume",
        type=Path,
        metavar="RUN",
        help="run directory of an incomplete run; reruns only its unfinished cases",
    )
    parser.add_argument("--suite", type=Path, help="complete replacement case list")
    parser.add_argument("--output", type=Path, help="parent of new run directories")
    args = parser.parse_args(arguments)
    if args.resume is not None and (args.suite is not None or args.output is not None):
        parser.error("--resume reuses the run's saved suite and directory")
    try:
        config = (
            load_incomplete_run(args.resume)
            if args.resume is not None
            else load_benchmark_config(
                args.model,
                args.suite or DEFAULT_SUITE,
                output_directory=args.output or Path("results"),
            )
        )
    except (ValueError, OSError, YAMLError) as error:
        print(error, file=sys.stderr)
        return 2
    configure_logging(config.log_level)
    runner = BenchmarkRunner(config, resume=args.resume)

    async def run_benchmark() -> BenchmarkResult:
        # Collection and its trace context belong to the benchmark's async task.
        with ExitStack() as stack:

            def started(path: Path) -> None:
                stack.enter_context(collect_telemetry(path, send_to_logfire=True))
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
    except ValueError as error:
        # Only a refused resume raises before the run starts; the run is unchanged.
        print(error, file=sys.stderr)
        return 2
    print(f"Model: {result.model.model}")
    print(f"Combined Benchmark Score: {result.total_score:+.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
