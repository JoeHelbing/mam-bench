import json
import tempfile
import unittest
from pathlib import Path

from pydantic_ai.models.test import TestModel

from mam_bench.benchmark import ModelRuntime, PrimaryScore, RuntimeInfo
from mam_bench.config import (
    BenchmarkConfig,
    ModelSelection,
    OpenRouterModel,
    load_benchmark_config,
)
from mam_bench.runner import run_benchmark


def fake_runtime(model_id: str) -> ModelRuntime:
    return ModelRuntime(
        info=RuntimeInfo(
            model_id=model_id,
            provider="openrouter",
            model=f"vendor/{model_id}",
            endpoint="https://example.test/v1",
        ),
        model=TestModel(),
    )


def unexpected_runtime(model: ModelSelection) -> ModelRuntime:
    raise AssertionError(f"unexpected runtime: {model.id}")


class FakeSimulation:
    simulation_id = "fake-simulation"
    simulation_version = "v1"

    async def run(self, runtime: ModelRuntime, output_directory: Path) -> PrimaryScore:
        output_directory.mkdir(parents=True)
        (output_directory / "artifact.txt").write_text(runtime.info.model_id, encoding="utf-8")
        return PrimaryScore(name="score", value=float(len(runtime.info.model_id)), unit="points")


class BenchmarkRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_runs_the_yaml_matrix_and_writes_topline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "results"
            config_path = root / "benchmark.yaml"
            config_path.write_text(
                """simulations: [fake-simulation]
models:
  - {id: model-a, runtime: openrouter, model: vendor/a, provider: provider-a}
  - {id: model-b, runtime: openrouter, model: vendor/b, provider: provider-b}
output_directory: results
""",
                encoding="utf-8",
            )
            config = load_benchmark_config(config_path)

            topline = await run_benchmark(
                config,
                simulations={"fake-simulation": FakeSimulation()},
                runtime_builder=lambda model: fake_runtime(model.id),
            )

            self.assertEqual(
                tuple(entry.model_id for entry in topline.entries),
                ("model-a", "model-b"),
            )
            self.assertEqual(
                (output / "runs" / "fake-simulation" / "model-a" / "artifact.txt").read_text(),
                "model-a",
            )
            payload = json.loads((output / "topline.json").read_text(encoding="utf-8"))
            self.assertEqual(payload, topline.model_dump(mode="json"))

    async def test_refuses_to_overwrite_an_existing_output_directory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("unchanged", encoding="utf-8")
            config = BenchmarkConfig(
                simulations=("fake-simulation",),
                models=(
                    OpenRouterModel(
                        id="model-a",
                        runtime="openrouter",
                        model="vendor/a",
                        provider="provider-a",
                    ),
                ),
                output_directory=output,
            )

            with self.assertRaises(FileExistsError):
                await run_benchmark(
                    config,
                    simulations={"fake-simulation": FakeSimulation()},
                    runtime_builder=unexpected_runtime,
                )

            self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")

    async def test_unknown_simulation_fails_before_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            config = BenchmarkConfig(
                simulations=("missing",),
                models=(
                    OpenRouterModel(
                        id="model-a",
                        runtime="openrouter",
                        model="vendor/a",
                        provider="provider-a",
                    ),
                ),
                output_directory=output,
            )

            with self.assertRaises(KeyError):
                await run_benchmark(config, simulations={})

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
