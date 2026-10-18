"""Acceptance checks for the explicit 0.1.0 suite contract."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from pydantic import ValidationError
from pydantic_ai.models.function import FunctionModel

from mam_bench.config import BenchmarkConfig
from mam_bench.model_setup import VLLMModelMetadata
from mam_bench.runner import BenchmarkRunner
from support import civil, schelling, selection, stay, streaming_model


class ConfigurationAcceptanceTests(unittest.TestCase):
    def test_missing_case_fields_are_rejected_upfront(self) -> None:
        with self.assertRaises(ValidationError) as caught:
            BenchmarkConfig.model_validate(
                {
                    "model": {
                        "runtime": "vllm",
                        "model": "scripted",
                        "base_url": "http://127.0.0.1:1234/v1",
                        "api_key_env": "TEST_KEY",
                    },
                    "cases": [{"simulation": "schelling"}],
                    "output_directory": "unused",
                }
            )
        self.assertIn("cases.0.schelling.board_size", str(caught.exception))


class SuiteAcceptanceTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        discovery = patch(
            "mam_bench.model_setup._fetch_vllm_model",
            AsyncMock(return_value=VLLMModelMetadata(id="scripted", max_model_len=260_000)),
        )
        discovery.start()
        self.addCleanup(discovery.stop)

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
