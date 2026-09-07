import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo

from mam_bench.benchmark import PairInfrastructureFailure
from mam_bench.config import load_benchmark_config
from mam_bench.runner import run_benchmark
from mam_bench.simulations.schelling import SchellingSim
from mam_bench.simulations.schelling.models import EvaluationResult, Trajectory
from schelling_support import runtime_for


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_yaml_parameters_reach_both_worlds_and_saved_results(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "benchmark.yaml"
            path.write_text(
                "simulations: [schelling-influence-pilot-v1]\n"
                "schelling:\n"
                "  board_size: 8\n"
                "  tolerance: '2/3'\n"
                "  vacancy_fraction: '1/4'\n"
                "  seed_id: 7\n"
                "  max_transitions: 2\n"
                "  vision_radius: 2\n"
                "  controlled_agent_count: 4\n"
                "  objective: segregation\n"
                "models:\n"
                "  - id: offline\n"
                "    runtime: openrouter\n"
                "    model: offline\n"
                "    provider: offline\n"
                f"output_directory: {root / 'results'}\n"
            )
            config = load_benchmark_config(path)
            await run_benchmark(config, runtime_builder=lambda _: runtime_for())
            result_path = next((root / "results").rglob("result.json"))
            summary = json.loads(result_path.read_text())
            self.assertEqual(summary["config"]["vision_radius"], 2)
            self.assertEqual(summary["config"]["max_transitions"], 2)
            self.assertEqual(summary["config"]["seed_id"], 7)
            self.assertEqual(summary["config"]["objective"], "segregation")
            self.assertEqual(summary["config"]["controlled_agent_count"], 4)
            self.assertEqual(summary["controlled_agent_ids"], [0, 1, 24, 25])
            expected = SchellingSim(settings=config.schelling).run_reference()
            with (
                np.load(result_path.parent / "ordinary.npz") as ordinary,
                np.load(result_path.parent / "model-controlled.npz") as controlled,
            ):
                np.testing.assert_array_equal(ordinary["cell_types"], expected.cell_types)
                np.testing.assert_array_equal(
                    controlled["cell_types"][0], ordinary["cell_types"][0]
                )
                self.assertEqual(controlled["cell_types"].shape, (3, 8, 8))

    async def test_artifacts_and_fresh_baseline_per_evaluation(self) -> None:
        simulation = SchellingSim()
        original = SchellingSim.run_reference
        seeds: list[int] = []

        def observed(sim: SchellingSim) -> Trajectory:
            seeds.append(sim.seed_id)
            return original(sim)

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(SchellingSim, "run_reference", observed),
        ):
            for index in range(2):
                output = Path(directory) / str(index)
                score = await simulation.run(runtime_for(), output)
                self.assertEqual(
                    {p.name for p in output.iterdir()},
                    {
                        "ordinary.npz",
                        "model-controlled.npz",
                        "result.json",
                        "agent-messages",
                        "message-board.jsonl",
                    },
                )
                summary = json.loads((output / "result.json").read_text())
                self.assertEqual(summary["final_directional_lift"], score.value)
                self.assertEqual(summary["simulation_version"], "schelling-influence-v4")
                result = await simulation.run()
                assert isinstance(result, EvaluationResult)
                with (
                    np.load(output / "ordinary.npz") as ordinary,
                    np.load(output / "model-controlled.npz") as controlled,
                ):
                    np.testing.assert_array_equal(
                        ordinary["cell_types"], result.ordinary.cell_types
                    )
                    np.testing.assert_array_equal(
                        controlled["agent_locations"], result.model_controlled.agent_locations
                    )
                    np.testing.assert_array_equal(
                        ordinary["cell_types"][0], controlled["cell_types"][0]
                    )
                    np.testing.assert_array_equal(
                        ordinary["agent_locations"][0], controlled["agent_locations"][0]
                    )
                    self.assertEqual(len(controlled["cell_types"]), 31)
                with self.assertRaises(FileExistsError):
                    await simulation.run(runtime_for(), output)
            self.assertEqual(seeds, [50, 50])

    async def test_provider_failure_unscored_and_fresh_retry(self) -> None:
        async def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise ModelAPIError("offline", "token=must-not-appear")

        simulation = SchellingSim()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "run"
            with self.assertRaises(PairInfrastructureFailure) as raised:
                await simulation.run(runtime_for(fail), output)
            self.assertNotIn("must-not-appear", str(raised.exception))
            self.assertTrue((output / "agent-messages").is_dir())
            self.assertFalse((output / "result.json").exists())
            with self.assertRaises(RuntimeError):
                await simulation.step()
            retry_output = Path(directory) / "retry"
            await simulation.run(runtime_for(), retry_output)
            self.assertTrue((retry_output / "result.json").exists())

    async def test_write_failure_does_not_return_a_score(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("numpy.savez_compressed", side_effect=OSError("disk full")),
        ):
            with self.assertRaises(PairInfrastructureFailure) as raised:
                await SchellingSim().run(runtime_for(), Path(directory) / "run")
            self.assertEqual(raised.exception.failure.kind, "artifact_write")
