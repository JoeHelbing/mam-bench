"""Sequential benchmark orchestration; simulations own evaluation and scoring."""

import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, computed_field
from pydantic_ai.models import Model

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import BenchmarkConfig, ModelSelection
from mam_bench.diagnostics import ExecutionFailure, failure_trace
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
    version: Literal["0.1.0"] = "0.1.0"
    model: ModelSelection
    cases: tuple[CaseResult, ...]

    @computed_field
    @property
    def total_score(self) -> float:
        return sum(case.score for case in self.cases)


class BenchmarkRunFailure(RuntimeError):
    """Safe failure information; completed case results remain on the runner."""

    def __init__(self, case_index: int | None, kind: str) -> None:
        self.case_index, self.kind = case_index, kind
        location = "setup/finalization" if case_index is None else f"case {case_index + 1}"
        super().__init__(f"Benchmark incomplete: {location} failed ({kind}); no final total.")


class BenchmarkRunner:
    def __init__(
        self,
        config: BenchmarkConfig,
        *,
        model_factory: Callable[[ModelSelection], Model] = create_model,
    ) -> None:
        self.config = config
        self.model_factory = model_factory
        self.results: list[CaseResult] = []
        self.output_directory: Path | None = None

    async def run(
        self,
        *,
        on_output_directory: Callable[[Path], None] | None = None,
        on_case: Callable[[int, CaseResult], None] | None = None,
    ) -> BenchmarkResult:
        if self.output_directory is not None:
            raise RuntimeError("create a new runner for a new benchmark")
        self.output_directory = (
            self.config.output_directory / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid4().hex[:12]}"
        )
        writer: ArtifactWriter | None = None
        case_writer: ArtifactWriter | None = None
        case_index: int | None = None
        try:
            writer = ArtifactWriter(self.output_directory)
            if on_output_directory is not None:
                on_output_directory(self.output_directory)
            writer.write("config.json", self.config)
            model = self.model_factory(self.config.model)
            for case_index, settings in enumerate(self.config.cases):
                case_writer = ArtifactWriter(
                    self.output_directory / "cases" / f"{case_index + 1:03d}"
                )
                runtime = CaseRuntime(model, self.config.model.settings, case_writer)
                simulation = (
                    SchellingSim(settings)
                    if isinstance(settings, SchellingSettings)
                    else CivilViolenceSim(settings)
                )
                logger.info(
                    "case.start index=%d simulation=%s", case_index + 1, settings.simulation
                )
                result = await simulation.evaluate(runtime)
                writer.append(
                    "completed-cases.jsonl",
                    {"case": case_index + 1, **result.model_dump(mode="json")},
                )
                self.results.append(result)
                logger.info("case.end index=%d score=%+.6f", case_index + 1, result.score)
                if on_case is not None:
                    on_case(case_index, result)
            case_index, case_writer = None, None
            result = BenchmarkResult(model=self.config.model, cases=tuple(self.results))
            writer.write("benchmark.json", result)
            return result
        except BaseException as error:
            kind = (
                error.kind
                if isinstance(error, ExecutionFailure)
                else ("artifact_write" if isinstance(error, OSError) else type(error).__name__)
            )
            logger.error(
                "benchmark.failed case=%s kind=%s trace=%s", case_index, kind, failure_trace(error)
            )
            failure: dict[str, object] = {
                "status": "incomplete",
                "case": None if case_index is None else case_index + 1,
                "kind": kind,
                "completed_cases": len(self.results),
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
            raise BenchmarkRunFailure(case_index, kind) from None
