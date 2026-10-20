"""Acceptance checks for the explicit 0.1.0 suite contract."""

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from mam_bench.config import BenchmarkConfig
from mam_bench.model_setup import VLLMModelMetadata
from mam_bench.runner import BenchmarkRunner
from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim
from mam_bench.simulations.schelling.simulation import SchellingSim
from support import SandboxTestCase, civil, runtime, schelling, selection, stay, streaming_model


class ConfigurationAcceptanceTests(unittest.TestCase):
    def test_missing_case_fields_are_rejected_upfront(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            BenchmarkConfig.model_validate(
                {
                    "model": {
                        "runtime": "vllm",
                        "model": "scripted",
                        "base_url": "http://127.0.0.1:1234/v1",
                    },
                    "cases": [{"simulation": "schelling"}],
                    "output_directory": "unused",
                }
            )
        self.assertIn("cases.0.schelling.board_size", str(caught.exception))


class SuiteAcceptanceTests(SandboxTestCase):
    def setUp(self) -> None:
        super().setUp()
        discovery = patch(
            "mam_bench.model_setup._fetch_vllm_model",
            AsyncMock(return_value=VLLMModelMetadata(id="scripted", max_model_len=260_000)),
        )
        discovery.start()
        self.addCleanup(discovery.stop)

    async def test_failed_and_cancelled_cases_close_their_sandboxes(self) -> None:
        for kind in ("schelling", "civil-violence"):
            for cancel in (False, True):
                with (
                    self.subTest(simulation=kind, cancel=cancel),
                    tempfile.TemporaryDirectory() as directory,
                ):
                    entered = asyncio.Event()

                    async def model(
                        messages: list[ModelMessage],
                        info: AgentInfo,
                        *,
                        entered: asyncio.Event = entered,
                        cancel: bool = cancel,
                    ) -> ModelResponse:
                        entered.set()
                        if cancel:
                            await asyncio.Event().wait()
                        raise RuntimeError("simulated model failure")

                    simulation = (
                        SchellingSim(schelling())
                        if kind == "schelling"
                        else CivilViolenceSim(civil())
                    )
                    resources = runtime(Path(directory) / "case", FunctionModel(model))
                    task = asyncio.create_task(simulation.evaluate(resources))
                    await asyncio.wait_for(entered.wait(), 10)
                    if cancel:
                        task.cancel()
                    with self.assertRaises(asyncio.CancelledError if cancel else RuntimeError):
                        await task
                    self.assertTrue(self.sandboxes)
                    self.assertTrue(all(sandbox.closed for sandbox in self.sandboxes))

    async def test_scripted_mixed_suite_completes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = BenchmarkConfig(
                model=selection(),
                cases=(schelling(), civil()),
                output_directory=Path(directory),
            )
            result = await BenchmarkRunner(
                config, model_factory=lambda _: streaming_model(FunctionModel(stay))
            ).run()
            self.assertEqual(len(result.cases), 2)
            self.assertEqual(result.total_score, sum(case.score for case in result.cases))
            self.assertEqual(len(self.sandboxes), 20)
            self.assertTrue(all(sandbox.closed for sandbox in self.sandboxes))
