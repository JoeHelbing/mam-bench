import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from pydantic_ai.models.test import TestModel

from mam_bench.benchmark import ModelRuntime, RuntimeInfo
from mam_bench.simulations.schelling.simulation import SchellingBenchmarkSimulation

RUNTIME = ModelRuntime(
    info=RuntimeInfo(
        model_id="model-a",
        provider="openrouter",
        model="vendor/model",
        endpoint="https://example.test/v1",
    ),
    model=TestModel(),
)


class SchellingBenchmarkSimulationTests(unittest.IsolatedAsyncioTestCase):
    def test_compact_reference_data_is_a_package_resource(self) -> None:
        dataset = files("mam_bench.data").joinpath("schelling-reference-v2")

        self.assertTrue(dataset.joinpath("landscape.jsonl").is_file())
        self.assertTrue(dataset.joinpath("evaluation-reference.json").is_file())
        self.assertTrue(dataset.joinpath("evaluation-reference.npz").is_file())

    async def test_runs_the_fixed_case_and_returns_its_directional_lift(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            simulation = SchellingBenchmarkSimulation()
            evaluation = AsyncMock(return_value=SimpleNamespace(final_directional_lift=0.25))
            reference = SimpleNamespace()
            with (
                patch(
                    "mam_bench.simulations.schelling.simulation.load_evaluation_reference",
                    return_value=reference,
                ) as load_reference,
                patch(
                    "mam_bench.simulations.schelling.simulation.run_model_evaluation",
                    evaluation,
                ),
            ):
                score = await simulation.run(RUNTIME, root / "output")

            load_reference.assert_called_once()
            dataset_root = load_reference.call_args.args[0]
            self.assertTrue(dataset_root.joinpath("evaluation-reference.json").is_file())
            config = evaluation.await_args_list[0].args[0]
            self.assertIs(evaluation.await_args_list[0].kwargs["reference"], reference)
            self.assertEqual(config.seed_id, 50)
            self.assertEqual(config.cell.board_size, 20)
            self.assertEqual(str(config.cell.tolerance), "3/4")
            self.assertEqual(str(config.cell.vacancy_fraction), "1/4")
            self.assertEqual(score.value, 0.25)


if __name__ == "__main__":
    unittest.main()
