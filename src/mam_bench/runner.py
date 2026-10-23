"""Sequential benchmark orchestration; simulations own evaluation and scoring."""

import json
import logging
import subprocess
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, computed_field
from pydantic_ai.models import Model

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import BenchmarkConfig, ModelSelection
from mam_bench.diagnostics import ExecutionFailure, failure_trace
from mam_bench.model_setup import ResolvedModelSetup, resolve_model_setup
from mam_bench.qualification import FallbackLedger
from mam_bench.runtime import CaseRuntime, create_model
from mam_bench.simulations.civil_violence.results import EvaluationResult as CivilViolenceResult
from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim
from mam_bench.simulations.schelling.results import EvaluationResult as SchellingResult
from mam_bench.simulations.schelling.settings import SchellingSettings
from mam_bench.simulations.schelling.simulation import SchellingSim

logger = logging.getLogger(__name__)
type CaseResult = SchellingResult | CivilViolenceResult


class BenchmarkResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    version: Literal["0.2.0"] = "0.2.0"
    model: ModelSelection
    cases: tuple[CaseResult, ...]

    @computed_field
    @property
    def total_score(self) -> float:
        return sum(case.adjusted_score for case in self.cases)


class BenchmarkRunFailure(RuntimeError):
    """Classified run failure with its original exception retained as the cause."""

    def __init__(self, case_index: int | None, kind: str) -> None:
        self.case_index, self.kind = case_index, kind
        location = "setup/finalization" if case_index is None else f"case {case_index + 1}"
        outcome = "failed" if kind == "model_fallbacks" else "incomplete"
        super().__init__(f"Benchmark {outcome}: {location} failed ({kind}); no final total.")


def source_version() -> dict[str, object]:
    """The checked-out commit and whether it has uncommitted changes, when known."""

    def git(*arguments: str) -> str:
        command = ["git", "-C", str(Path(__file__).parent), *arguments]
        return subprocess.run(command, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}
    except OSError, subprocess.CalledProcessError:
        return {"commit": None, "dirty": None}


def _effective_model(setup: ResolvedModelSetup) -> dict[str, object]:
    selection = setup.selection
    return {
        "model": selection.model_dump(mode="json"),
        # Request decisions that a resumed attempt must reproduce exactly.
        "setup": {
            "context_window_tokens": setup.context_window_tokens,
            "completion_token_parameter": selection.completion_token_parameter,
            "required_tool_choice_supported": selection.required_tool_choice_supported,
        },
        "metadata": setup.metadata,
    }


def _records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def load_incomplete_run(directory: Path) -> BenchmarkConfig:
    """Return an incomplete run's saved configuration; final or changed runs are refused."""
    if (directory / "benchmark.json").exists():
        raise ValueError(f"{directory} is complete; its total is final")
    failure = directory / "failure.json"
    if (
        failure.exists()
        and json.loads(failure.read_text(encoding="utf-8")).get("status") == "failed"
    ):
        raise ValueError(f"{directory} failed qualification; the model cannot play this run")
    if not (directory / "attempts.jsonl").exists():
        # Written after preflight: the run stopped before any case or predates resume.
        raise ValueError(f"{directory} has no attempts.jsonl to resume; start a new run")
    text = (directory / "config.json").read_text(encoding="utf-8")
    config = BenchmarkConfig.model_validate_json(text)
    # New fields or defaults would silently change the cases still to run.
    if json.loads(config.model_dump_json()) != json.loads(text):
        raise ValueError(f"{directory}: the current code reads its suite or model differently")
    return config


class BenchmarkRunner:
    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        model_factory: Callable[[ModelSelection], Model] = create_model,
        resume: Path | None = None,
    ) -> None:
        self.config = config
        self.model_factory = model_factory
        self.resume = resume
        self.results: list[CaseResult] = []
        self.output_directory: Path | None = None

    async def _restore(
        self, directory: Path, fallbacks: FallbackLedger, version: dict[str, object]
    ) -> tuple[int, ResolvedModelSetup]:
        """Reload completed cases and archive the previous attempt's leftovers.

        Every check runs before the run directory changes, so a refused resume
        leaves the incomplete run as it was.
        """
        completed = _records(directory / "completed-cases.jsonl")
        if len(completed) > len(self.config.cases):
            raise ValueError(f"{directory} records more completed cases than its suite has")
        for index, record in enumerate(completed):
            settings = self.config.cases[index]
            kind = (
                SchellingResult if isinstance(settings, SchellingSettings) else CivilViolenceResult
            )
            result = kind.model_validate_json(json.dumps(record))
            if record["case"] != index + 1 or result.config != settings:
                raise ValueError(
                    f"{directory}: completed case {index + 1} does not match its suite"
                )
            self.results.append(result)
            fallbacks.turns += record["fallbacks"]["turns"]
            fallbacks.fallbacks += record["fallbacks"]["fallbacks"]
        setup = await resolve_model_setup(self.config.model)
        saved = json.loads((directory / "effective-model.json").read_text(encoding="utf-8"))
        current = _effective_model(setup)
        if any(saved.get(key) != current[key] for key in ("model", "setup")):
            raise ValueError(f"{directory}: the model's serving setup changed since the run began")
        attempts = _records(directory / "attempts.jsonl")
        earlier = [{key: attempt[key] for key in ("commit", "dirty")} for attempt in attempts]
        if version["dirty"] is not False or any(source != version for source in earlier):
            logger.warning("resume.code_changed current=%s earlier=%s", version, earlier)
        previous = attempts[-1]["attempt"]
        archive = directory / "attempts" / f"{previous:03d}"
        archive.mkdir(parents=True)
        unfinished = range(len(completed) + 1, len(self.config.cases) + 1)
        for name in ("failure.json", "telemetry", *(f"cases/{case:03d}" for case in unfinished)):
            if (directory / name).exists():
                (archive / name).parent.mkdir(exist_ok=True)
                (directory / name).rename(archive / name)
        return previous + 1, setup

    async def run(
        self,
        *,
        on_output_directory: Callable[[Path], None] | None = None,
        on_case: Callable[[int, CaseResult], None] | None = None,
    ) -> BenchmarkResult:
        """Run every case, or a resumed run's unfinished cases, and save the total.

        `on_case` receives every case in suite order, including restored ones.
        """
        if self.output_directory is not None:
            raise RuntimeError("create a new runner for a new benchmark")
        fallbacks = FallbackLedger(
            sum(case.max_steps * case.controlled_agent_count for case in self.config.cases)
        )
        version = source_version()
        attempt, setup = 1, None
        if self.resume is None:
            self.output_directory = (
                self.config.output_directory
                / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:12]}"
            )
        else:
            attempt, setup = await self._restore(self.resume, fallbacks, version)
            self.output_directory = self.resume
        writer: ArtifactWriter | None = None
        case_writer: ArtifactWriter | None = None
        case_index: int | None = None
        try:
            writer = ArtifactWriter(self.output_directory, existing=self.resume is not None)
            if on_output_directory is not None:
                on_output_directory(self.output_directory)
            if setup is None:
                writer.write("config.json", self.config)
                setup = await resolve_model_setup(self.config.model)
                writer.write("effective-model.json", _effective_model(setup))
            first = len(self.results)
            writer.append(
                "attempts.jsonl",
                {
                    "attempt": attempt,
                    "started_at": datetime.now(UTC).isoformat(),
                    **version,
                    "cases": list(range(first + 1, len(self.config.cases) + 1)),
                },
            )
            if on_case is not None:
                for index, result in enumerate(self.results):
                    on_case(index, result)
            model = self.model_factory(setup.selection)
            for case_index, settings in enumerate(self.config.cases[first:], start=first):
                case_writer = ArtifactWriter(
                    self.output_directory / "cases" / f"{case_index + 1:03d}"
                )
                runtime = CaseRuntime(
                    model,
                    setup.selection.settings,
                    case_writer,
                    setup.context_window_tokens,
                    fallbacks,
                )
                simulation = (
                    SchellingSim(settings)
                    if isinstance(settings, SchellingSettings)
                    else CivilViolenceSim(settings)
                )
                logger.info(
                    "case.start index=%d simulation=%s", case_index + 1, settings.simulation
                )
                turns, fallen_back = fallbacks.turns, fallbacks.fallbacks
                result = await simulation.evaluate(runtime)
                writer.append(
                    "completed-cases.jsonl",
                    {
                        "case": case_index + 1,
                        "attempt": attempt,
                        # This case's share of the run-wide qualification counts.
                        "fallbacks": {
                            "turns": fallbacks.turns - turns,
                            "fallbacks": fallbacks.fallbacks - fallen_back,
                        },
                        **result.model_dump(mode="json"),
                    },
                )
                self.results.append(result)
                logger.info("case.end index=%d score=%+.6f", case_index + 1, result.score)
                if on_case is not None:
                    on_case(case_index, result)
            case_index, case_writer = None, None
            fallbacks.check(final=True)
            result = BenchmarkResult(model=setup.selection, cases=tuple(self.results))
            writer.write("benchmark.json", result)
            return result
        except BaseException as error:
            kind = (
                error.kind
                if isinstance(error, ExecutionFailure)
                else ("artifact_write" if isinstance(error, OSError) else type(error).__name__)
            )
            trace = failure_trace(error)
            logger.error("benchmark.failed case=%s kind=%s trace=%s", case_index, kind, trace)
            failure: dict[str, object] = {
                # A model that cannot play fails; provider and other failures are rerun.
                "status": "failed" if kind == "model_fallbacks" else "incomplete",
                "attempt": attempt,
                "case": None if case_index is None else case_index + 1,
                "kind": kind,
                "completed_cases": len(self.results),
                "fallbacks": fallbacks.summary(),
                "error": str(error),
                "traceback": trace,
            }
            for target in (case_writer, writer):
                if target is not None:
                    try:
                        target.write("failure.json", failure)
                    except Exception as persistence_error:
                        logger.error(
                            "failure_record.failed trace=%s", failure_trace(persistence_error)
                        )
            if not isinstance(error, Exception):
                raise
            raise BenchmarkRunFailure(case_index, kind) from error
