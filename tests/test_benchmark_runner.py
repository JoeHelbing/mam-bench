import json
import tempfile
import unittest
from contextlib import chdir
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.test import TestModel

from mam_bench.benchmark import (
    BenchmarkRunFailure,
    PairFailure,
    PairInfrastructureFailure,
    PrimaryScore,
)
from mam_bench.config import (
    BenchmarkConfig,
    ModelSelection,
    OpenRouterModel,
    load_benchmark_config,
)
from mam_bench.model import ModelRuntime, RuntimeInfo
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


class MixedSimulation(FakeSimulation):
    def __init__(self) -> None:
        self.attempts: list[str] = []

    async def run(self, runtime: ModelRuntime, output_directory: Path) -> PrimaryScore:
        self.attempts.append(runtime.info.model_id)
        if runtime.info.model_id in {"model-a", "model-c"}:
            failure = PairFailure(
                simulation_id=self.simulation_id,
                simulation_version=self.simulation_version,
                model_id=runtime.info.model_id,
                provider=runtime.info.provider,
                model=runtime.info.model,
                kind="provider",
            )
            output_directory.mkdir(parents=True)
            (output_directory / "failure.json").write_text(
                failure.model_dump_json(), encoding="utf-8"
            )
            (output_directory / "failed-events.jsonl").write_text("", encoding="utf-8")
            raise PairInfrastructureFailure(failure)
        return await super().run(runtime, output_directory)


class BenchmarkRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_finishes_mixed_matrix_before_raising_aggregate_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            models = tuple(
                OpenRouterModel(
                    id=model_id,
                    runtime="openrouter",
                    model=f"vendor/{model_id}",
                    provider="provider-a",
                )
                for model_id in ("model-a", "model-b", "model-c")
            )
            config = BenchmarkConfig(
                simulations=("fake-simulation",),
                models=models,
                output_directory=output,
            )
            simulation = MixedSimulation()

            with self.assertRaises(BenchmarkRunFailure) as raised:
                await run_benchmark(
                    config,
                    simulations={"fake-simulation": simulation},
                    runtime_builder=lambda model: fake_runtime(model.id),
                )

            self.assertEqual(simulation.attempts, ["model-a", "model-b", "model-c"])
            self.assertEqual(
                tuple(failure.model_id for failure in raised.exception.failures),
                ("model-a", "model-c"),
            )
            output = next(output.iterdir())
            topline = json.loads((output / "topline.json").read_text(encoding="utf-8"))
            self.assertEqual([entry["model_id"] for entry in topline["entries"]], ["model-b"])
            self.assertEqual(
                {path.name for path in (output / "runs" / "fake-simulation" / "model-a").iterdir()},
                {"failure.json", "failed-events.jsonl"},
            )
            self.assertEqual(
                (output / "runs" / "fake-simulation" / "model-b" / "artifact.txt").read_text(
                    encoding="utf-8"
                ),
                "model-b",
            )

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
            with chdir(root):
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
            output = next(output.iterdir())
            self.assertEqual(
                (output / "runs" / "fake-simulation" / "model-a" / "artifact.txt").read_text(),
                "model-a",
            )
            payload = json.loads((output / "topline.json").read_text(encoding="utf-8"))
            self.assertEqual(payload, topline.model_dump(mode="json"))

    async def test_reruns_full_matrix_without_changing_previous_attempts(self) -> None:
        for partial_failure in (False, True):
            with (
                self.subTest(partial_failure=partial_failure),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "results"
                output.mkdir()
                marker = output / "keep.txt"
                marker.write_text("unchanged", encoding="utf-8")
                config = BenchmarkConfig(
                    simulations=("first", "second"),
                    models=tuple(
                        OpenRouterModel(
                            id=name, runtime="openrouter", model=name, provider="provider-a"
                        )
                        for name in ("model-a", "model-b")
                    ),
                    output_directory=output,
                )
                first = MixedSimulation() if partial_failure else FakeSimulation()
                first.simulation_id = "first"
                second = FakeSimulation()
                second.simulation_id = "second"
                attempts: list[Path] = []

                def record_attempt(path: Path, recorded: list[Path] = attempts) -> None:
                    self.assertTrue(path.is_dir())
                    self.assertEqual(list(path.iterdir()), [])
                    recorded.append(path)

                def build(model: ModelSelection) -> ModelRuntime:
                    return fake_runtime(model.id)

                builder = Mock(side_effect=build)
                with (
                    patch.object(first, "run", wraps=first.run) as first_run,
                    patch.object(second, "run", wraps=second.run) as second_run,
                    patch("mam_bench.runner.datetime") as clock,
                ):
                    # Even attempts started within the same second must be distinct.
                    clock.now.return_value = datetime(2026, 1, 1, tzinfo=UTC)
                    snapshots: dict[Path, bytes] = {}
                    for _ in range(2):
                        invocation = run_benchmark(
                            config,
                            simulations={"first": first, "second": second},
                            runtime_builder=builder,
                            on_output_directory=record_attempt,
                        )
                        if partial_failure:
                            with self.assertRaises(BenchmarkRunFailure):
                                await invocation
                        else:
                            await invocation
                        for path, content in snapshots.items():
                            self.assertEqual(path.read_bytes(), content)
                        snapshots = {
                            path: path.read_bytes() for path in output.rglob("*") if path.is_file()
                        }
                    self.assertEqual(first_run.await_count, 4)
                    self.assertEqual(second_run.await_count, 4)
                    self.assertEqual(builder.call_count, 4)

                self.assertEqual(len(set(attempts)), 2)
                self.assertEqual(config.output_directory, output)
                self.assertEqual(marker.read_text(encoding="utf-8"), "unchanged")
                for attempt in attempts:
                    self.assertEqual(attempt.parent, output)
                    self.assertTrue(attempt.name.startswith("20260101T000000Z-"))
                    payload = json.loads((attempt / "topline.json").read_text())
                    self.assertEqual(len(payload["entries"]), 3 if partial_failure else 4)
                    self.assertEqual(len(list((attempt / "runs").glob("*/*"))), 4)

    async def test_registry_accepts_names_without_regex_restrictions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = BenchmarkConfig(
                simulations=("Custom Simulation",),
                models=(
                    OpenRouterModel(
                        id="Model A", runtime="openrouter", model="vendor/a", provider="provider-a"
                    ),
                ),
                output_directory=Path(directory) / "results",
            )
            topline = await run_benchmark(
                config,
                simulations={"Custom Simulation": FakeSimulation()},
                runtime_builder=lambda model: fake_runtime(model.id),
            )
            self.assertEqual(len(topline.entries), 1)

    async def test_runs_simulations_without_probing_models(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "results"
            config = BenchmarkConfig(
                simulations=("first", "second"),
                models=tuple(
                    OpenRouterModel(
                        id=name, runtime="openrouter", model=name, provider="provider-a"
                    )
                    for name in ("model-a", "model-b")
                ),
                output_directory=output,
            )
            first = Mock(spec=FakeSimulation)
            first.simulation_id = "first"
            first.simulation_version = "v1"
            first.run = AsyncMock(return_value=PrimaryScore(name="score", value=1, unit="points"))
            second = Mock(spec=FakeSimulation)
            second.simulation_id = "second"
            second.simulation_version = "v1"
            second.run = AsyncMock(return_value=PrimaryScore(name="score", value=1, unit="points"))

            def build(model: ModelSelection) -> ModelRuntime:
                return fake_runtime(model.id)

            builder = Mock(side_effect=build)
            with patch.object(TestModel, "request", new_callable=AsyncMock) as request:
                await run_benchmark(
                    config,
                    simulations={"first": first, "second": second},
                    runtime_builder=builder,
                )
            request.assert_not_awaited()
            self.assertEqual(builder.call_count, 2)
            self.assertEqual(first.run.await_count, 2)
            self.assertEqual(second.run.await_count, 2)

    async def test_simulation_model_call_errors_propagate(self) -> None:
        for error in (TimeoutError("private detail"), RuntimeError("private detail")):
            with (
                self.subTest(error=type(error).__name__),
                tempfile.TemporaryDirectory() as directory,
            ):
                output = Path(directory) / "results"
                config = BenchmarkConfig(
                    simulations=("fake-simulation",),
                    models=tuple(
                        OpenRouterModel(
                            id=name, runtime="openrouter", model=name, provider="provider-a"
                        )
                        for name in ("model-a", "model-b")
                    ),
                    output_directory=output,
                )
                simulation = FakeSimulation()

                async def call_model(runtime: ModelRuntime, path: Path) -> PrimaryScore:
                    await runtime.model.request([], None, ModelRequestParameters())
                    raise AssertionError("expected model call to fail")

                with (
                    patch.object(TestModel, "request", side_effect=error) as request,
                    patch.object(simulation, "run", side_effect=call_model) as run,
                    self.assertRaises(type(error)) as raised,
                ):
                    await run_benchmark(
                        config,
                        simulations={"fake-simulation": simulation},
                        runtime_builder=lambda model: fake_runtime(model.id),
                    )
                run.assert_awaited_once()
                request.assert_awaited_once()
                self.assertTrue(output.exists())
                self.assertIs(raised.exception, error)

    async def test_missing_credentials_fail_initialization_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
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
                output_directory=Path(directory) / "results",
            )
            builder = Mock(side_effect=KeyError("private environment name"))
            with self.assertRaisesRegex(ValueError, "Runtime initialization failed"):
                await run_benchmark(
                    config,
                    simulations={"fake-simulation": FakeSimulation()},
                    runtime_builder=builder,
                )
            self.assertFalse(config.output_directory.exists())

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

            with self.assertRaisesRegex(
                ValueError, "Unknown simulations: 'missing'. Available simulations: fake-simulation"
            ):
                await run_benchmark(
                    config,
                    simulations={"fake-simulation": FakeSimulation()},
                    runtime_builder=unexpected_runtime,
                )

            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
