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
from mam_bench.simulations.schelling import SchellingSim
from mam_bench.simulations.schelling.models import EvaluationResult, Trajectory
from schelling_support import runtime_for


class EvaluationTests(unittest.IsolatedAsyncioTestCase):
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
                self.assertEqual(summary["simulation_version"], "schelling-influence-v3")
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
