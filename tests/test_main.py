"""CLI validation, per-case display and withheld totals on failure."""

import contextlib
import io
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import main
import yaml
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionDef, FunctionModel

from mam_bench.config import BenchmarkConfig
from mam_bench.runner import BenchmarkRunner
from support import (
    SandboxTestCase,
    civil,
    records,
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
        def runner(config: BenchmarkConfig, resume: Path | None = None) -> BenchmarkRunner:
            return BenchmarkRunner(
                config, model_factory=lambda _: streaming_model(FunctionModel(stay)), resume=resume
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

        def runner(config: BenchmarkConfig, resume: Path | None = None) -> BenchmarkRunner:
            return BenchmarkRunner(
                config,
                model_factory=lambda _: streaming_model(FunctionModel(script)),
                resume=resume,
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

    def cli(self, arguments: list[str], script: FunctionDef) -> tuple[int, str, str]:
        """Run the CLI with a scripted model; return its exit code, stdout and stderr."""

        def runner(config: BenchmarkConfig, resume: Path | None = None) -> BenchmarkRunner:
            return BenchmarkRunner(
                config,
                model_factory=lambda _: streaming_model(FunctionModel(script)),
                resume=resume,
            )

        output, errors = io.StringIO(), io.StringIO()
        with (
            patch("main.BenchmarkRunner", side_effect=runner),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(errors),
        ):
            code = main.main(arguments)
        return code, output.getvalue(), errors.getvalue()

    def run_directory(self, errors: str) -> Path:
        return Path(errors.split("Output directory: ", 1)[1].split("\n", 1)[0])

    def test_resume_reruns_unfinished_cases_to_the_uninterrupted_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            args = self.files(Path(directory))
            code, _, errors = self.cli(args, stay)
            self.assertEqual(code, 0)
            uninterrupted = json.loads((self.run_directory(errors) / "benchmark.json").read_text())

            with patch("mam_bench.runner.source_version", return_value=COMMIT_A):
                code, _, errors = self.cli(args, overloaded_in_case_2)
            self.assertEqual(code, 1)
            run = self.run_directory(errors)
            first_case = {p: p.read_bytes() for p in (run / "cases/001").rglob("*")}

            with (
                patch("mam_bench.runner.source_version", return_value=COMMIT_B),
                self.assertLogs("mam_bench", "WARNING") as logs,
            ):
                code, output, errors = self.cli(["--resume", str(run)], stay)
            self.assertEqual(code, 0, errors)
            self.assertEqual(json.loads((run / "benchmark.json").read_text()), uninterrupted)
            self.assertIn("Case 1:", output)
            self.assertIn("Case 2:", output)
            self.assertIn(f"Combined Benchmark Score: {uninterrupted['total_score']:+.6f}", output)
            # Case 1 is reused as it was; the failed attempt's leftovers are kept apart.
            self.assertTrue(all(path.read_bytes() == data for path, data in first_case.items()))
            self.assertFalse((run / "failure.json").exists())
            archive = run / "attempts/001"
            self.assertEqual(json.loads((archive / "failure.json").read_text())["kind"], "provider")
            self.assertTrue((archive / "cases/002/failure.json").exists())
            self.assertTrue((archive / "telemetry/traces.jsonl").read_text())
            self.assertTrue((run / "telemetry/traces.jsonl").read_text())
            # Provenance: which attempt produced each case, and from which code.
            self.assertEqual(
                [case["attempt"] for case in records(run / "completed-cases.jsonl")], [1, 2]
            )
            self.assertEqual(
                [
                    (attempt["attempt"], attempt["commit"], attempt["cases"])
                    for attempt in records(run / "attempts.jsonl")
                ],
                [(1, COMMIT_A["commit"], [1, 2]), (2, COMMIT_B["commit"], [2])],
            )
            self.assertIn("resume.code_changed", "\n".join(logs.output))

    def test_resume_refuses_final_or_changed_runs_and_leaves_them_unchanged(self) -> None:
        def fabricated_suite(run: Path) -> None:
            # A run saved before a defaulted field existed reads differently now.
            config = json.loads((run / "config.json").read_text())
            del config["log_level"]
            (run / "config.json").write_text(json.dumps(config))

        for name, script, change, message in (
            ("complete", stay, None, "is complete"),
            ("failed qualification", text_destination, None, "failed qualification"),
            ("suite read differently", overloaded_in_case_2, fabricated_suite, "differently"),
            ("serving changed", overloaded_in_case_2, None, "serving setup changed"),
        ):
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                _, _, errors = self.cli(self.files(Path(directory)), script)
                run = self.run_directory(errors)
                if change is not None:
                    change(run)
                before = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
                # The endpoint now serves a different context window.
                with (
                    served_model(128_000) if name == "serving changed" else contextlib.nullcontext()
                ):
                    code, _, errors = self.cli(["--resume", str(run)], stay)
                self.assertEqual(code, 2)
                self.assertIn(message, errors)
                after = {p: p.read_bytes() for p in run.rglob("*") if p.is_file()}
                self.assertEqual(after, before)


COMMIT_A = {"commit": "a" * 40, "dirty": False}
COMMIT_B = {"commit": "b" * 40, "dirty": False}


async def overloaded_in_case_2(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Play Schelling (case 1); fail every Civil Violence (case 2) request past retries."""
    if any(tool.name == "act" for tool in info.output_tools):
        raise ModelHTTPError(503, "scripted overload")
    return await stay(messages, info)


async def text_destination(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Every Schelling move is invalid, so every turn falls back."""
    return ModelResponse(parts=[ToolCallPart("move", {"destination": "[3, 4]"})])
