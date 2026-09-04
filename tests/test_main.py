import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import AsyncMock, patch

import main as benchmark_main

from mam_bench.benchmark import BenchmarkTopline, PrimaryScore, ToplineEntry


class MainTests(unittest.TestCase):
    def test_rejects_non_yaml_configuration_path(self) -> None:
        with self.assertRaises(SystemExit):
            benchmark_main.main(["benchmark.json"])

    def test_loads_runs_and_prints_one_topline(self) -> None:
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
            path = Path(directory) / "benchmark.yaml"
            path.write_text(
                """simulations: [schelling-influence-pilot-v1]
models:
  - {id: model-a, runtime: openrouter, model: vendor/model, provider: provider-a}
output_directory: results
""",
                encoding="utf-8",
            )
            output = io.StringIO()
            with (
                patch.object(benchmark_main, "run_benchmark", AsyncMock(return_value=topline)),
                redirect_stdout(output),
            ):
                status = benchmark_main.main([str(path)])

        self.assertEqual(status, 0)
        self.assertEqual(output.getvalue(), "simulation-a / model-a: 0.125000 points\n")


if __name__ == "__main__":
    unittest.main()
