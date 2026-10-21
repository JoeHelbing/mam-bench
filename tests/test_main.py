"""CLI validation, per-case display and withheld totals on failure."""

import contextlib
import io
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import main
import yaml
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from mam_bench.config import BenchmarkConfig
from mam_bench.runner import BenchmarkRunner
from support import (
    SandboxTestCase,
    civil,
    schelling,
    selection,
    served_model,
    stay,
    streaming_model,
)


class MainTests(SandboxTestCase):
    def setUp(self) -> None:
        super().setUp()
        # Local credentials must never enable remote export in offline CLI tests.
        environment = patch.dict(os.environ, {"LOGFIRE_TOKEN": ""})
        environment.start()
        self.addCleanup(environment.stop)
        self.enterContext(served_model())

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
            return BenchmarkRunner(
                config, model_factory=lambda _: streaming_model(FunctionModel(stay))
            )

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
            run = next((Path(directory) / "results").iterdir())
            for signal in ("traces", "logs", "metrics"):
                self.assertTrue((run / "telemetry" / f"{signal}.jsonl").read_text())
            for case in (run / "cases").iterdir():
                self.assertFalse(
                    {"agent-messages", "agent-streams", "provider-streams", "message-board.jsonl"}
                    & {path.name for path in case.iterdir()}
                )

    def test_failed_case_keeps_error_details_and_never_prints_total(self) -> None:
        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(tool.name == "act" for tool in info.output_tools):
                raise ModelHTTPError(400, "offline", "SECRET-PROVIDER")
            return await stay(messages, info)

        def runner(config: BenchmarkConfig) -> BenchmarkRunner:
            return BenchmarkRunner(
                config, model_factory=lambda _: streaming_model(FunctionModel(script))
            )

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
            records = "\n".join(
                record.read_text() for record in (Path(directory) / "results").rglob("*.json*")
            )
            self.assertIn("SECRET-PROVIDER", records)

    def test_invalid_model_with_default_suite_does_not_claim_suite_is_deferred(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.yaml"
            model.write_text("runtime: invalid\nmodel: nonexistent\n")
            errors = io.StringIO()
            with patch("main.BenchmarkRunner") as runner, contextlib.redirect_stderr(errors):
                self.assertEqual(main.main(["--model", str(model)]), 2)
                runner.assert_not_called()
            self.assertIn("model", errors.getvalue())
            self.assertIn("input_value=", errors.getvalue())
            self.assertNotIn("deferred", errors.getvalue())

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
