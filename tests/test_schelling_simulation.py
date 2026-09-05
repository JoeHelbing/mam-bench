import json
import tempfile
import unittest
from importlib.resources import files
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest.mock import AsyncMock, patch

from pydantic_ai.models.test import TestModel

from mam_bench.benchmark import (
    AgentInfrastructureFailure,
    ModelRuntime,
    PairInfrastructureFailure,
    RuntimeInfo,
)
from mam_bench.evidence import EvidenceRecorder
from mam_bench.simulations.schelling.runtime import EvidencePublicationError
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
    async def test_provider_failure_publishes_only_redacted_partial_evidence(self) -> None:
        async def fail_evaluation(*args: object, **kwargs: object) -> object:
            del args
            evidence = cast(EvidenceRecorder, kwargs["evidence"])
            await evidence.record("request_started", api_key="must-not-appear")
            raise AgentInfrastructureFailure("provider", "raw provider secret")

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            with (
                patch(
                    "mam_bench.simulations.schelling.simulation.run_v2_model_evaluation",
                    side_effect=fail_evaluation,
                ),
                self.assertRaises(PairInfrastructureFailure) as raised,
            ):
                await SchellingBenchmarkSimulation().run(RUNTIME, output)

            self.assertEqual(raised.exception.failure.kind, "provider")
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"failure.json", "failed-events.jsonl"},
            )
            failure_text = (output / "failure.json").read_text(encoding="utf-8")
            events_text = (output / "failed-events.jsonl").read_text(encoding="utf-8")
            self.assertNotIn("raw provider secret", failure_text)
            self.assertNotIn("must-not-appear", events_text)
            self.assertIn("[REDACTED]", events_text)
            events = [json.loads(line) for line in events_text.splitlines()]
            self.assertEqual(events[-1]["kind"], "pair_failed")

    async def test_success_serialization_failure_becomes_an_unscored_pair(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            evaluation = AsyncMock(return_value=SimpleNamespace(final_directional_lift=0.25))
            with (
                patch(
                    "mam_bench.simulations.schelling.simulation.run_v2_model_evaluation",
                    evaluation,
                ),
                patch(
                    "mam_bench.simulations.schelling.simulation.publish_v2_success",
                    side_effect=EvidencePublicationError("serialization failed"),
                ),
                self.assertRaises(PairInfrastructureFailure) as raised,
            ):
                await SchellingBenchmarkSimulation().run(RUNTIME, output)

            self.assertEqual(raised.exception.failure.kind, "evidence_publication")
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {"failure.json", "failed-events.jsonl"},
            )

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
                    "mam_bench.simulations.schelling.simulation.run_v2_model_evaluation",
                    evaluation,
                ),
                patch("mam_bench.simulations.schelling.simulation.publish_v2_success") as publish,
            ):
                score = await simulation.run(RUNTIME, root / "output")

            load_reference.assert_called_once()
            dataset_root = load_reference.call_args.args[0]
            self.assertTrue(dataset_root.joinpath("evaluation-reference.json").is_file())
            config = evaluation.await_args_list[0].args[0]
            self.assertIs(evaluation.await_args_list[0].kwargs["reference"], reference)
            team = evaluation.await_args_list[0].args[1]
            self.assertIs(evaluation.await_args_list[0].kwargs["evidence"], team.evidence)
            publish.assert_called_once()
            self.assertEqual(config.seed_id, 50)
            self.assertEqual(config.cell.board_size, 20)
            self.assertEqual(str(config.cell.tolerance), "3/4")
            self.assertEqual(str(config.cell.vacancy_fraction), "1/4")
            self.assertEqual(score.value, 0.25)
            self.assertEqual(simulation.simulation_version, "schelling-influence-v2")

    async def test_refuses_to_overwrite_an_existing_pair_before_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "output"
            output.mkdir()
            (output / "keep.txt").write_text("keep", encoding="utf-8")

            with (
                patch(
                    "mam_bench.simulations.schelling.simulation.run_v2_model_evaluation",
                    new_callable=AsyncMock,
                ) as evaluation,
                self.assertRaises(FileExistsError),
            ):
                await SchellingBenchmarkSimulation().run(RUNTIME, output)

            evaluation.assert_not_awaited()
            self.assertEqual((output / "keep.txt").read_text(encoding="utf-8"), "keep")


if __name__ == "__main__":
    unittest.main()
