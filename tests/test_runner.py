"""Ordered execution, complete totals, cancellation, and retained failure artifacts."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse
from pydantic_ai.models.function import AgentInfo, FunctionModel

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import AgentSettings, BenchmarkConfig, CaseSettings
from mam_bench.runner import BenchmarkRunFailure, BenchmarkRunner
from support import civil, observation, records, schelling, selection, stay


class RunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_complete_suite_order_fresh_case_state_and_saved_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = BenchmarkConfig(
                model=selection(),
                cases=(schelling(objective="segregation"), civil(), schelling()),
                output_directory=Path(directory),
            )
            seen: list[str] = []
            runner = BenchmarkRunner(config, model_factory=lambda _: FunctionModel(stay))
            result = await runner.run(on_case=lambda _, case: seen.append(case.config.simulation))
            self.assertEqual(seen, ["schelling", "civil-violence", "schelling"])
            self.assertAlmostEqual(result.cases[0].score, -result.cases[2].score)
            self.assertAlmostEqual(result.total_score, result.cases[1].score)
            assert runner.output_directory is not None
            saved = json.loads((runner.output_directory / "benchmark.json").read_text())
            self.assertEqual(saved["total_score"], sum(case["score"] for case in saved["cases"]))
            before = {
                p.relative_to(runner.output_directory): p.read_bytes()
                for p in runner.output_directory.rglob("*")
                if p.is_file()
            }
            rerun = BenchmarkRunner(config, model_factory=lambda _: FunctionModel(stay))
            again = await rerun.run()
            self.assertNotEqual(runner.output_directory, rerun.output_directory)
            self.assertEqual(result, again)
            self.assertTrue(
                all(
                    (runner.output_directory / p).read_bytes() == data for p, data in before.items()
                )
            )

    async def test_each_scheduler_cancels_on_failure_keeps_steps_and_blocks_total(self) -> None:
        for failed_case in (schelling(), civil(threshold=10_000)):
            with self.subTest(simulation=failed_case.simulation):
                await self.check_scheduler_failure(failed_case)

    async def check_scheduler_failure(self, failed_case: CaseSettings) -> None:
        with tempfile.TemporaryDirectory() as directory:
            failed_case_started = False
            admitted: list[str] = []
            sibling_started, sibling_cancelled = asyncio.Event(), asyncio.Event()

            async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                if failed_case_started and observation(messages)["step"] == 2:
                    admitted.append(info.instructions or "")
                    if len(admitted) == 1:
                        await sibling_started.wait()
                        raise ModelAPIError("offline", "SECRET-PROVIDER-BODY")
                    sibling_started.set()
                    try:
                        await asyncio.Event().wait()
                    finally:
                        sibling_cancelled.set()
                return await stay(messages, info)

            def completed(index: int, result: object) -> None:
                nonlocal failed_case_started
                failed_case_started = True

            config = BenchmarkConfig(
                model=selection().model_copy(
                    update={"settings": AgentSettings(concurrency=2, timeout_seconds=4)}
                ),
                cases=(schelling(max_steps=1), failed_case, schelling()),
                output_directory=Path(directory),
            )
            runner = BenchmarkRunner(config, model_factory=lambda _: FunctionModel(script))
            with (
                self.assertLogs("mam_bench", level="ERROR") as logs,
                self.assertRaises(BenchmarkRunFailure) as failure,
            ):
                await runner.run(on_case=completed)
            self.assertEqual(failure.exception.case_index, 1)
            self.assertEqual(len(admitted), 2)
            self.assertTrue(sibling_cancelled.is_set())
            self.assertEqual(len(runner.results), 1)
            root = runner.output_directory
            assert root is not None
            self.assertTrue((root / "cases/001/result.json").exists())
            self.assertFalse((root / "cases/002/result.json").exists())
            self.assertFalse((root / "cases/003").exists())
            self.assertFalse((root / "benchmark.json").exists())
            self.assertEqual(len(records(root / "cases/002/controlled.jsonl")), 2)
            self.assertTrue((root / "cases/002/ordinary.jsonl").exists())
            self.assertTrue((root / "cases/002/ordinary-outcome.json").exists())
            self.assertEqual(
                len(
                    json.loads((root / "cases/002/config.json").read_text())["controlled_agent_ids"]
                ),
                failed_case.controlled_agent_count,
            )
            self.assertTrue((root / "cases/002/agent-messages").is_dir())
            self.assertEqual(
                json.loads((root / "failure.json").read_text())["status"], "incomplete"
            )
            self.assertNotIn("SECRET-PROVIDER-BODY", "\n".join(logs.output))
            for path in root.rglob("*.json*"):
                self.assertNotIn("SECRET-PROVIDER-BODY", path.read_text())

    async def test_storage_failure_retains_previous_steps_and_reports_failed_failure_write(
        self,
    ) -> None:
        original_append, original_write = ArtifactWriter.append, ArtifactWriter.write

        def append(writer: ArtifactWriter, name: str, record: object) -> None:
            if name == "controlled.jsonl" and cast(dict[str, object], record)["step"] == 2:
                raise OSError("SECRET-STORAGE")
            original_append(writer, name, record)

        def write(writer: ArtifactWriter, name: str, record: object) -> None:
            if name == "failure.json":
                raise OSError("SECRET-FAILURE-RECORD")
            original_write(writer, name, record)  # pyright: ignore[reportArgumentType]

        with tempfile.TemporaryDirectory() as directory:
            runner = BenchmarkRunner(
                BenchmarkConfig(
                    model=selection(), cases=(schelling(),), output_directory=Path(directory)
                ),
                model_factory=lambda _: FunctionModel(stay),
            )
            with (
                patch.object(ArtifactWriter, "append", append),
                patch.object(ArtifactWriter, "write", write),
                self.assertLogs("mam_bench", level="ERROR") as logs,
                self.assertRaises(BenchmarkRunFailure) as failed,
            ):
                await runner.run()
            self.assertEqual(failed.exception.kind, "artifact_write")
            self.assertIn("failure_record.failed", "\n".join(logs.output))
            self.assertNotIn("SECRET", "\n".join(logs.output))
            root = runner.output_directory
            assert root is not None
            self.assertEqual(len(records(root / "cases/001/controlled.jsonl")), 2)
            self.assertFalse((root / "benchmark.json").exists())
