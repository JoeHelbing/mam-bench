"""Ordered execution, complete totals, cancellation, and retained failure artifacts."""

import asyncio
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, patch

from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import AgentSettings, BenchmarkConfig, CaseSettings, ModelSelection
from mam_bench.diagnostics import ExecutionFailure
from mam_bench.qualification import FallbackLedger
from mam_bench.runner import (
    BenchmarkResult,
    BenchmarkRunFailure,
    BenchmarkRunner,
    load_incomplete_run,
)
from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim
from mam_bench.simulations.schelling.simulation import SchellingSim
from support import (
    SandboxTestCase,
    civil,
    observation,
    records,
    runtime,
    schelling,
    selection,
    served_model,
    stay,
    streaming_model,
)


class RunnerTests(SandboxTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.enterContext(served_model())

    async def test_complete_suite_order_fresh_case_state_and_saved_total(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            config = BenchmarkConfig(
                model=selection(),
                cases=(schelling(objective="segregation"), civil(), schelling()),
                output_directory=Path(directory),
            )
            seen: list[str] = []
            runner = BenchmarkRunner(
                config, model_factory=lambda _: streaming_model(FunctionModel(stay))
            )
            result = await runner.run(on_case=lambda _, case: seen.append(case.config.simulation))
            self.assertEqual(seen, ["schelling", "civil-violence", "schelling"])
            self.assertAlmostEqual(result.cases[0].score, -result.cases[2].score)
            self.assertAlmostEqual(result.total_score, result.cases[1].adjusted_score)
            assert runner.output_directory is not None
            saved = json.loads((runner.output_directory / "benchmark.json").read_text())
            self.assertEqual(
                saved["total_score"], sum(case["adjusted_score"] for case in saved["cases"])
            )
            self.assertTrue((runner.output_directory / "effective-model.json").exists())
            before = {
                p.relative_to(runner.output_directory): p.read_bytes()
                for p in runner.output_directory.rglob("*")
                if p.is_file()
            }
            rerun = BenchmarkRunner(
                config, model_factory=lambda _: streaming_model(FunctionModel(stay))
            )
            again = await rerun.run()
            self.assertNotEqual(runner.output_directory, rerun.output_directory)
            self.assertEqual(result, again)
            self.assertTrue(self.sandboxes)
            self.assertTrue(all(sandbox.closed for sandbox in self.sandboxes))
            self.assertTrue(
                all(
                    (runner.output_directory / p).read_bytes() == data for p, data in before.items()
                )
            )

    async def test_total_counts_only_change_beyond_the_null_policy(self) -> None:
        # A lone case has no matched goal pair to cancel its null-policy score.
        with tempfile.TemporaryDirectory() as directory:
            config = BenchmarkConfig(
                model=selection(), cases=(schelling(),), output_directory=Path(directory)
            )
            runner = BenchmarkRunner(
                config, model_factory=lambda _: streaming_model(FunctionModel(stay))
            )
            result = await runner.run()
            self.assertNotEqual(result.cases[0].score, 0)
            self.assertEqual(result.total_score, 0)

    async def run_until_disqualified(
        self, cases: tuple[CaseSettings, ...], model: FunctionModel
    ) -> tuple[BenchmarkRunFailure, dict[str, Any]]:
        with tempfile.TemporaryDirectory() as directory:
            config = BenchmarkConfig(
                model=selection(), cases=cases, output_directory=Path(directory)
            )
            runner = BenchmarkRunner(config, model_factory=lambda _: streaming_model(model))
            with self.assertRaises(BenchmarkRunFailure) as failure, self.assertLogs("mam_bench"):
                await runner.run()
            root = runner.output_directory
            assert root is not None
            self.assertFalse((root / "benchmark.json").exists())
            return failure.exception, json.loads((root / "failure.json").read_text())

    async def test_model_that_cannot_play_fails_without_a_score(self) -> None:
        async def text_destination(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(parts=[ToolCallPart("move", {"destination": "[3, 4]"})])

        # Each Schelling fixture case allows 4 controlled agents x max_steps turns; agents
        # already mid-turn when the rule fires still finish and are counted.
        for name, cases, least, below in (
            # 12 possible turns: two fallbacks exceed 10% of every possible turn, so the
            # run stops before taking them all.
            ("certain", (schelling(),), 2, 12),
            # 360 possible turns: half of the first 48 turns falling back stops the run
            # before the 37 fallbacks that would make failure certain.
            ("early", (schelling(max_steps=30),) * 3, 24, 37),
        ):
            with self.subTest(rule=name):
                error, record = await self.run_until_disqualified(
                    cases, FunctionModel(text_destination)
                )
                self.assertEqual(error.kind, "model_fallbacks")
                self.assertEqual(record["status"], "failed")
                self.assertEqual(record["kind"], "model_fallbacks")
                counts = record["fallbacks"]
                self.assertEqual(counts["fallbacks"], counts["turns"])
                self.assertGreaterEqual(counts["fallbacks"], least)
                self.assertLess(counts["fallbacks"], below)

    def test_runs_that_end_early_judge_fallbacks_against_turns_taken(self) -> None:
        # A revolution can end worlds early: 2 of 10 taken turns is 20%, although 2 of
        # 100 possible turns would pass.
        ledger = FallbackLedger(possible_turns=100)
        for reason in ("call_error_limit", "usage_limit", *[None] * 8):
            ledger.record(reason)
        with self.assertRaises(ExecutionFailure) as failure:
            ledger.check(final=True)
        self.assertEqual(failure.exception.kind, "model_fallbacks")
        ledger.record(None)  # 2 of 11 is still over 10%; the in-progress check waits.

    async def test_provider_errors_that_outlast_retries_leave_the_run_incomplete(self) -> None:
        async def overloaded(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise ModelHTTPError(503, "scripted")

        error, record = await self.run_until_disqualified((schelling(),), FunctionModel(overloaded))
        self.assertEqual(error.kind, "provider")
        self.assertEqual(record["status"], "incomplete")
        # The provider never turned a turn into a fallback; the world stopped instead.
        self.assertEqual(record["fallbacks"], {"turns": 0, "fallbacks": 0})

    async def test_fallbacks_up_to_ten_percent_of_turns_keep_the_score(self) -> None:
        # Three cases x 4 agents x 3 steps = 36 turns. Three fallbacks (8.3%) keep the
        # score; a fourth exceeds 3.6, i.e. 10%.
        self.assertEqual((await self.run_with_failed_turns(3)).total_score, 0)
        with self.assertRaises(BenchmarkRunFailure) as failure, self.assertLogs():
            await self.run_with_failed_turns(4)
        self.assertEqual(failure.exception.kind, "model_fallbacks")

    async def run_with_failed_turns(self, fallbacks: int) -> BenchmarkResult:
        """Stay on every turn except the first `fallbacks` turns, which fall back."""
        failing: set[str] = set()
        failed_turns = 0

        async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal failed_turns
            agent = info.instructions or ""
            # A request that carries a user prompt starts the agent's next turn.
            if any(isinstance(part, UserPromptPart) for part in messages[-1].parts):
                failing.discard(agent)
                if failed_turns < fallbacks:
                    failing.add(agent)
                    failed_turns += 1
            if agent in failing:
                return ModelResponse(parts=[ToolCallPart("move", {"destination": "x"})])
            return await stay(messages, info)

        with tempfile.TemporaryDirectory() as directory:
            config = BenchmarkConfig(
                model=selection(), cases=(schelling(),) * 3, output_directory=Path(directory)
            )
            runner = BenchmarkRunner(
                config, model_factory=lambda _: streaming_model(FunctionModel(script))
            )
            return await runner.run()

    async def test_resumed_run_counts_earlier_fallbacks_toward_the_limit(self) -> None:
        # Three cases x 4 agents x 3 steps = 36 turns, so a fourth fallback exceeds 10%.
        # Attempt 1 falls back on 3 of case 1's 12 turns, then the provider fails in case 2.
        # Resumed, one more fallback is only 1 of 24 new turns but the run's fourth.
        def model(fallbacks: int, overload_after: int | None = None) -> FunctionModel:
            failing: set[str] = set()
            turns = 0

            async def script(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
                nonlocal turns
                agent = info.instructions or ""
                if any(isinstance(part, UserPromptPart) for part in messages[-1].parts):
                    turns += 1
                    failing.discard(agent)
                    if turns <= fallbacks:
                        failing.add(agent)
                if overload_after is not None and turns > overload_after:
                    raise ModelHTTPError(503, "scripted")
                if agent in failing:
                    return ModelResponse(parts=[ToolCallPart("move", {"destination": "x"})])
                return await stay(messages, info)

            return streaming_model(FunctionModel(script))

        with tempfile.TemporaryDirectory() as directory:
            config = BenchmarkConfig(
                model=selection(), cases=(schelling(),) * 3, output_directory=Path(directory)
            )
            interrupted = BenchmarkRunner(config, model_factory=lambda _: model(3, 12))
            with self.assertRaises(BenchmarkRunFailure) as failure, self.assertLogs("mam_bench"):
                await interrupted.run()
            self.assertEqual(failure.exception.kind, "provider")
            root = interrupted.output_directory
            assert root is not None
            self.assertEqual(
                records(root / "completed-cases.jsonl")[0]["fallbacks"],
                {"turns": 12, "fallbacks": 3},
            )

            resumed = BenchmarkRunner(
                load_incomplete_run(root), model_factory=lambda _: model(1), resume=root
            )
            with self.assertRaises(BenchmarkRunFailure) as failure, self.assertLogs("mam_bench"):
                await resumed.run()
            self.assertEqual(failure.exception.kind, "model_fallbacks")
            record = json.loads((root / "failure.json").read_text())
            self.assertEqual(record["status"], "failed")
            self.assertEqual(record["fallbacks"]["fallbacks"], 4)
            self.assertGreater(record["fallbacks"]["turns"], 12)
            self.assertFalse((root / "benchmark.json").exists())
            with self.assertRaisesRegex(ValueError, "failed qualification"):
                load_incomplete_run(root)

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

    async def test_preflight_records_effective_model_before_first_case(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            original = selection()
            effective = original.model_copy(
                update={"settings": AgentSettings(max_completion_tokens=512)}
            )
            config = BenchmarkConfig(
                model=original,
                cases=(schelling(max_steps=1),),
                output_directory=Path(directory),
            )
            seen: list[ModelSelection] = []

            def model_factory(selected: ModelSelection) -> FunctionModel:
                seen.append(selected)
                return streaming_model(FunctionModel(stay))

            runner = BenchmarkRunner(config, model_factory=model_factory)
            resolve = AsyncMock(
                return_value=SimpleNamespace(
                    selection=effective,
                    context_window_tokens=16_384,
                    metadata={"source": "test-endpoint", "reason": "smaller output limit"},
                )
            )
            with patch("mam_bench.runner.resolve_model_setup", resolve):
                result = await runner.run()
            self.assertEqual(seen, [effective])
            self.assertEqual(result.model, effective)
            assert runner.output_directory is not None
            output = runner.output_directory
            original_record = json.loads((output / "config.json").read_text())
            self.assertIsNone(original_record["model"]["settings"]["max_completion_tokens"])
            record = json.loads((output / "effective-model.json").read_text())
            self.assertEqual(record["model"]["settings"]["max_completion_tokens"], 512)
            self.assertEqual(record["metadata"]["source"], "test-endpoint")
            self.assertTrue((output / "cases/001/result.json").exists())

    async def test_failed_preflight_starts_no_case_or_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = BenchmarkRunner(
                BenchmarkConfig(
                    model=selection(),
                    cases=(schelling(max_steps=1),),
                    output_directory=Path(directory),
                ),
                model_factory=lambda _: self.fail("preflight must precede model construction"),
            )
            with (
                patch(
                    "mam_bench.runner.resolve_model_setup",
                    AsyncMock(side_effect=ValueError("missing endpoint limit")),
                ),
                self.assertRaises(BenchmarkRunFailure) as failure,
            ):
                await runner.run()
            self.assertIsNone(failure.exception.case_index)
            self.assertIsInstance(failure.exception.__cause__, ValueError)
            assert runner.output_directory is not None
            self.assertFalse((runner.output_directory / "cases").exists())
            record = json.loads((runner.output_directory / "failure.json").read_text())
            self.assertEqual(record["kind"], "ValueError")
            self.assertEqual(record["error"], "missing endpoint limit")
            self.assertIn("ValueError: missing endpoint limit", record["traceback"])

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
                        raise ModelHTTPError(400, "offline", "SECRET-PROVIDER-BODY")
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
            runner = BenchmarkRunner(
                config, model_factory=lambda _: streaming_model(FunctionModel(script))
            )
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
            self.assertEqual(
                json.loads((root / "failure.json").read_text())["status"], "incomplete"
            )
            self.assertIn("SECRET-PROVIDER-BODY", "\n".join(logs.output))
            for path in (root / "failure.json", root / "cases/002/failure.json"):
                record = json.loads(path.read_text())
                self.assertIn("ModelHTTPError", record["traceback"])
                self.assertIn("SECRET-PROVIDER-BODY", record["traceback"])
            self.assertIsInstance(failure.exception.__cause__, Exception)

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
                model_factory=lambda _: streaming_model(FunctionModel(stay)),
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
            self.assertIn("SECRET-STORAGE", "\n".join(logs.output))
            self.assertIn("SECRET-FAILURE-RECORD", "\n".join(logs.output))
            root = runner.output_directory
            assert root is not None
            self.assertEqual(len(records(root / "cases/001/controlled.jsonl")), 2)
            self.assertFalse((root / "benchmark.json").exists())
