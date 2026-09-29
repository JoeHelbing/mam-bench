"""CLI validation, per-case display and withheld totals on failure."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import main
import yaml
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from mam_bench.config import BenchmarkConfig
from mam_bench.runner import BenchmarkRunner
from support import civil, schelling, selection, stay


class MainTests(unittest.TestCase):
    def files(self, root: Path) -> list[str]:
        (root / "model.yaml").write_text(yaml.safe_dump(selection().model_dump()))
        (root / "suite.yaml").write_text(
            yaml.safe_dump(
                {
                    "cases": [
                        schelling(max_steps=1).model_dump(),
                        civil().model_dump(),
                    ]
                }
            )
        )
        return [
            "--model",
            str(root / "model.yaml"),
            "--suite",
            str(root / "suite.yaml"),
            "--output",
            str(root / "results"),
        ]

    def test_cli_prints_effective_cases_and_complete_total(self) -> None:
        def runner(config: BenchmarkConfig) -> BenchmarkRunner:
            return BenchmarkRunner(config, model_factory=lambda _: FunctionModel(stay))

        with tempfile.TemporaryDirectory() as directory:
            args = self.files(Path(directory))
            output, errors = io.StringIO(), io.StringIO()
            with (
                patch(
                    "main.BenchmarkRunner",
                    side_effect=runner,
                ),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(errors),
            ):
                self.assertEqual(main.main(args), 0)
            text = output.getvalue()
            self.assertIn("Case 1:", text)
            self.assertIn("Case 2:", text)
            self.assertIn('"seed":7', text)
            self.assertIn('"objective":"increase"', text)
            self.assertIn("Combined Benchmark Score:", text)
            self.assertIn("Output directory:", errors.getvalue())

    def test_failed_case_is_visible_but_never_prints_total_or_sensitive_body(self) -> None:
        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(tool.name == "participate" for tool in info.output_tools):
                raise ModelAPIError("offline", "SECRET-PROVIDER")
            return await stay(messages, info)

        def runner(config: BenchmarkConfig) -> BenchmarkRunner:
            return BenchmarkRunner(config, model_factory=lambda _: FunctionModel(script))

        with tempfile.TemporaryDirectory() as directory:
            args = self.files(Path(directory))
            output, errors = io.StringIO(), io.StringIO()
            with (
                patch(
                    "main.BenchmarkRunner",
                    side_effect=runner,
                ),
                contextlib.redirect_stdout(output),
                contextlib.redirect_stderr(errors),
            ):
                self.assertEqual(main.main(args), 1)
            self.assertIn("Case 1:", output.getvalue())
            self.assertNotIn("Combined Benchmark Score:", output.getvalue())
            self.assertIn("Failed case:", errors.getvalue())
            self.assertNotIn("SECRET-PROVIDER", errors.getvalue())

    def test_last_case_validation_prevents_runner_construction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.files(root)
            (root / "suite.yaml").write_text(
                yaml.safe_dump(
                    {
                        "cases": [
                            schelling().model_dump(),
                            {"simulation": "civil-violence"},
                        ]
                    }
                )
            )
            errors = io.StringIO()
            with patch("main.BenchmarkRunner") as runner, contextlib.redirect_stderr(errors):
                self.assertEqual(main.main(args), 2)
                runner.assert_not_called()
            self.assertIn("cases.1.civil-violence.board_size", errors.getvalue())
            self.assertFalse((root / "results").exists())
