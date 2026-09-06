import io
import tempfile
import unittest
from collections.abc import Callable
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

import main as benchmark_main

from mam_bench.benchmark import (
    BenchmarkRunFailure,
    BenchmarkTopline,
    PairFailure,
    PrimaryScore,
    ToplineEntry,
)
from mam_bench.config import BenchmarkConfig


class MainTests(unittest.TestCase):
    def test_reports_aggregate_pair_failure_without_a_traceback(self) -> None:
        failure = BenchmarkRunFailure(
            (
                PairFailure(
                    simulation_id="simulation-a",
                    simulation_version="v1",
                    model_id="model-a",
                    provider="openrouter",
                    model="vendor/model",
                    kind="provider",
                ),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.yaml"
            path.write_text(
                """simulations: [schelling-influence-pilot-v1]
models:
  - {id: model-a, runtime: openrouter, model: vendor/model, provider: provider-a}
output_directory: results
""",
                encoding="utf-8",
            )
            attempt = Path(directory) / "results" / "attempt"

            async def fail(
                config: BenchmarkConfig, *, on_output_directory: Callable[[Path], None]
            ) -> BenchmarkTopline:
                on_output_directory(attempt)
                raise failure

            error_output = io.StringIO()
            with (
                patch.object(benchmark_main, "run_benchmark", AsyncMock(side_effect=fail)),
                redirect_stderr(error_output),
            ):
                status = benchmark_main.main([str(path)])

        self.assertEqual(status, 1)
        self.assertEqual(
            error_output.getvalue(),
            f"Output directory: {attempt}\n"
            "benchmark failed pairs: simulation-a / model-a (provider)\n",
        )

    def test_loads_runs_and_prints_one_topline_without_yaml_extension(self) -> None:
        topline = BenchmarkTopline(
            entries=(
                ToplineEntry(
                    simulation_id="simulation-a",
                    simulation_version="v1",
                    model_id="model-a",
                    provider="openrouter",
                    model="vendor/model",
                    primary_score=PrimaryScore(name="score", value=0.125, unit="points"),
                ),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.config"
            path.write_text(
                """log_level: DEBUG
simulations: [schelling-influence-pilot-v1]
models:
  - {id: model-a, runtime: openrouter, model: vendor/model, provider: provider-a}
output_directory: results
""",
                encoding="utf-8",
            )
            attempt = Path(directory) / "results" / "attempt"

            async def succeed(
                config: BenchmarkConfig, *, on_output_directory: Callable[[Path], None]
            ) -> BenchmarkTopline:
                on_output_directory(attempt)
                return topline

            output = io.StringIO()
            error_output = io.StringIO()
            with (
                patch.object(benchmark_main, "run_benchmark", AsyncMock(side_effect=succeed)),
                patch.object(benchmark_main, "configure_logging") as configure_logging,
                redirect_stdout(output),
                redirect_stderr(error_output),
            ):
                status = benchmark_main.main([str(path)])
                configure_logging.assert_called_once_with("DEBUG")

        self.assertEqual(status, 0)
        self.assertEqual(error_output.getvalue(), f"Output directory: {attempt}\n")
        self.assertEqual(output.getvalue(), "simulation-a / model-a: 0.125000 points\n")


if __name__ == "__main__":
    unittest.main()
