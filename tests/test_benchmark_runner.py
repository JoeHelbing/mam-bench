import hashlib
import tempfile
import unittest
from pathlib import Path

from pydantic import BaseModel, ConfigDict, JsonValue

from mam_bench.benchmark import (
    EvidenceReceipt,
    PrimaryScore,
    SimulationDescriptor,
    SimulationResult,
)
from mam_bench.runner import (
    BenchmarkSelection,
    CompatibilityPreflightError,
    EvidenceValidationError,
    preflight_benchmark,
    run_benchmark,
)
from mam_bench.runtime import (
    ModelRequest,
    ModelResponse,
    ModelRuntime,
    ModelUsage,
    RequestLimits,
    RuntimeCapability,
    RuntimeDescriptor,
    RuntimeMessage,
    RuntimeRequirements,
    SamplingControl,
    SamplingSettings,
    StructuredOutput,
    TextOutput,
    TextPart,
)


class _FakeEvidence(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    value: float


class _FakeManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_sha256: str


class _RecordingRuntime:
    def __init__(self, descriptor: RuntimeDescriptor) -> None:
        self.descriptor = descriptor
        self.calls: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.calls.append(request)
        output: JsonValue
        if isinstance(request.output, TextOutput):
            output = f"response from {self.descriptor.model_id}"
        else:
            output = {"accepted": True}
        return ModelResponse(
            request_id=request.request_id,
            messages=(
                RuntimeMessage(
                    role="assistant",
                    parts=(TextPart(text="complete"),),
                ),
            ),
            output=output,
            usage=ModelUsage(requests=1, input_tokens=10, output_tokens=2),
            latency_seconds=0.01,
            provider_response_id=None,
        )


class _RecordingRuntimeFactory:
    def __init__(self, descriptor: RuntimeDescriptor) -> None:
        self.descriptor = descriptor
        self.instances: list[_RecordingRuntime] = []

    def create(self) -> ModelRuntime:
        runtime = _RecordingRuntime(self.descriptor)
        self.instances.append(runtime)
        return runtime

    @property
    def call_count(self) -> int:
        return sum(len(runtime.calls) for runtime in self.instances)


class _PreparedCounter:
    descriptor = SimulationDescriptor(
        simulation_id="counter-v1",
        simulation_version="counter-v1",
        title="Counter simulation",
        primary_score_name="counter_score",
    )
    runtime_requirements = RuntimeRequirements(
        capabilities=frozenset({RuntimeCapability.TEXT_OUTPUT}),
        sampling_controls=frozenset(),
        minimum_concurrent_requests=1,
    )

    def __init__(self, increment: int) -> None:
        self.increment = increment

    async def execute(
        self,
        *,
        runtime: ModelRuntime,
        output_directory: Path,
        retain_diagnostic_artifacts: bool,
    ) -> None:
        del retain_diagnostic_artifacts
        await runtime.complete(_model_request("counter", structured=False))
        _write_evidence(
            output_directory,
            _FakeEvidence(
                model_id=runtime.descriptor.model_id,
                value=float(self.increment),
            ),
        )


class _CounterSimulation:
    descriptor = _PreparedCounter.descriptor

    def prepare(self) -> _PreparedCounter:
        return _PreparedCounter(increment=3)

    def validate(
        self,
        *,
        output_directory: Path,
        runtime: RuntimeDescriptor,
    ) -> SimulationResult:
        evidence, receipt = _validate_evidence(output_directory)
        return SimulationResult(
            simulation=self.descriptor,
            runtime=runtime,
            primary_score=PrimaryScore(
                name="counter_score",
                value=evidence.value,
                objective="increase the counter",
                meaning="fixed counter increment",
                unit="count",
                semantics_version="counter-v1",
            ),
            evidence=receipt,
            diagnostic_artifacts_retained=False,
        )


class _PreparedDemanding:
    descriptor = SimulationDescriptor(
        simulation_id="demanding-v1",
        simulation_version="demanding-v1",
        title="Demanding simulation",
        primary_score_name="demanding_score",
    )
    runtime_requirements = RuntimeRequirements(
        capabilities=frozenset(
            {
                RuntimeCapability.STRICT_STRUCTURED_OUTPUT,
                RuntimeCapability.TEXT_OUTPUT,
            }
        ),
        sampling_controls=frozenset({SamplingControl.TEMPERATURE}),
        minimum_concurrent_requests=2,
    )

    async def execute(
        self,
        *,
        runtime: ModelRuntime,
        output_directory: Path,
        retain_diagnostic_artifacts: bool,
    ) -> None:
        del runtime, output_directory, retain_diagnostic_artifacts
        raise AssertionError("incompatible simulation must not execute")


class _DemandingSimulation:
    descriptor = _PreparedDemanding.descriptor

    def prepare(self) -> _PreparedDemanding:
        return _PreparedDemanding()

    def validate(
        self,
        *,
        output_directory: Path,
        runtime: RuntimeDescriptor,
    ) -> SimulationResult:
        del output_directory, runtime
        raise AssertionError("incompatible simulation must not validate")


class _PreparedPhrase:
    descriptor = SimulationDescriptor(
        simulation_id="phrase-v1",
        simulation_version="phrase-v1",
        title="Phrase simulation",
        primary_score_name="phrase_length",
    )
    runtime_requirements = RuntimeRequirements(
        capabilities=frozenset({RuntimeCapability.STRICT_STRUCTURED_OUTPUT}),
        sampling_controls=frozenset(),
        minimum_concurrent_requests=1,
    )

    def __init__(self, phrase: str) -> None:
        self.phrase = phrase

    async def execute(
        self,
        *,
        runtime: ModelRuntime,
        output_directory: Path,
        retain_diagnostic_artifacts: bool,
    ) -> None:
        del retain_diagnostic_artifacts
        await runtime.complete(_model_request("phrase", structured=True))
        _write_evidence(
            output_directory,
            _FakeEvidence(
                model_id=runtime.descriptor.model_id,
                value=float(len(self.phrase)),
            ),
        )


class _PhraseSimulation:
    descriptor = _PreparedPhrase.descriptor

    def prepare(self) -> _PreparedPhrase:
        return _PreparedPhrase(phrase="civil")

    def validate(
        self,
        *,
        output_directory: Path,
        runtime: RuntimeDescriptor,
    ) -> SimulationResult:
        evidence, receipt = _validate_evidence(output_directory)
        return SimulationResult(
            simulation=self.descriptor,
            runtime=runtime,
            primary_score=PrimaryScore(
                name="phrase_length",
                value=evidence.value,
                objective="produce the fixed phrase",
                meaning="number of characters in the fixed phrase",
                unit="characters",
                semantics_version="phrase-v1",
            ),
            evidence=receipt,
            diagnostic_artifacts_retained=False,
        )


class _MissingEvidenceSimulation(_CounterSimulation):
    descriptor = SimulationDescriptor(
        simulation_id="missing-evidence-v1",
        simulation_version="missing-evidence-v1",
        title="Missing evidence simulation",
        primary_score_name="counter_score",
    )

    def prepare(self) -> _PreparedCounter:
        prepared = _PreparedCounter(increment=1)
        prepared.descriptor = self.descriptor
        return prepared

    def validate(
        self,
        *,
        output_directory: Path,
        runtime: RuntimeDescriptor,
    ) -> SimulationResult:
        evidence, _ = _validate_evidence(output_directory)
        return SimulationResult(
            simulation=self.descriptor,
            runtime=runtime,
            primary_score=PrimaryScore(
                name="counter_score",
                value=evidence.value,
                objective="increase the counter",
                meaning="fixed counter increment",
                unit="count",
                semantics_version="missing-evidence-v1",
            ),
            evidence=EvidenceReceipt(
                relative_directory=".",
                manifest_path="absent-manifest.json",
                manifest_sha256="0" * 64,
            ),
            diagnostic_artifacts_retained=False,
        )


class _InvalidEvidenceSimulation(_MissingEvidenceSimulation):
    descriptor = SimulationDescriptor(
        simulation_id="invalid-evidence-v1",
        simulation_version="invalid-evidence-v1",
        title="Invalid evidence simulation",
        primary_score_name="counter_score",
    )

    def validate(
        self,
        *,
        output_directory: Path,
        runtime: RuntimeDescriptor,
    ) -> SimulationResult:
        evidence, receipt = _validate_evidence(output_directory)
        return SimulationResult(
            simulation=self.descriptor,
            runtime=runtime,
            primary_score=PrimaryScore(
                name="counter_score",
                value=evidence.value,
                objective="increase the counter",
                meaning="fixed counter increment",
                unit="count",
                semantics_version="invalid-evidence-v1",
            ),
            evidence=receipt.model_copy(update={"manifest_sha256": "0" * 64}),
            diagnostic_artifacts_retained=False,
        )


def _runtime_descriptor(
    model_id: str,
    *,
    capabilities: frozenset[RuntimeCapability],
    sampling_controls: frozenset[SamplingControl] = frozenset(),
    max_concurrent_requests: int = 1,
    required_environment: tuple[str, ...] = (),
) -> RuntimeDescriptor:
    return RuntimeDescriptor(
        model_id=model_id,
        runtime="fake",
        provider="test",
        model=f"test/{model_id}",
        adapter_version="fake-v1",
        capabilities=capabilities,
        sampling_controls=sampling_controls,
        max_concurrent_requests=max_concurrent_requests,
        required_environment=required_environment,
    )


def _model_request(request_id: str, *, structured: bool) -> ModelRequest:
    output = (
        StructuredOutput(
            name="accepted",
            description="Return whether the request was accepted",
            json_schema={
                "type": "object",
                "properties": {"accepted": {"type": "boolean"}},
                "required": ["accepted"],
                "additionalProperties": False,
            },
        )
        if structured
        else TextOutput()
    )
    return ModelRequest(
        request_id=request_id,
        messages=(
            RuntimeMessage(
                role="user",
                parts=(TextPart(text="run the fake simulation"),),
            ),
        ),
        output=output,
        sampling=SamplingSettings(
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            reasoning_effort="none",
            max_output_tokens=32,
            request_seed=1,
        ),
        limits=RequestLimits(
            request_limit=1,
            tool_call_limit=0,
            timeout_seconds=1.0,
        ),
    )


def _write_evidence(output_directory: Path, evidence: _FakeEvidence) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    evidence_path = output_directory / "evidence.json"
    evidence_path.write_text(f"{evidence.model_dump_json()}\n", encoding="utf-8")
    manifest = _FakeManifest(evidence_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest())
    (output_directory / "evidence-manifest.json").write_text(
        f"{manifest.model_dump_json()}\n",
        encoding="utf-8",
    )


def _validate_evidence(
    output_directory: Path,
) -> tuple[_FakeEvidence, EvidenceReceipt]:
    evidence_path = output_directory / "evidence.json"
    manifest_path = output_directory / "evidence-manifest.json"
    evidence = _FakeEvidence.model_validate_json(evidence_path.read_text(encoding="utf-8"))
    manifest = _FakeManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if hashlib.sha256(evidence_path.read_bytes()).hexdigest() != manifest.evidence_sha256:
        raise ValueError("fake evidence hash does not match")
    return (
        evidence,
        EvidenceReceipt(
            relative_directory=".",
            manifest_path=manifest_path.name,
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        ),
    )


class BenchmarkRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_two_unrelated_simulations_preserve_matrix_order_and_scores(
        self,
    ) -> None:
        simulations = {
            "counter-v1": _CounterSimulation(),
            "phrase-v1": _PhraseSimulation(),
        }
        capabilities = frozenset(
            {
                RuntimeCapability.TEXT_OUTPUT,
                RuntimeCapability.STRICT_STRUCTURED_OUTPUT,
            }
        )
        factories = {
            model_id: _RecordingRuntimeFactory(
                _runtime_descriptor(model_id, capabilities=capabilities)
            )
            for model_id in ("model-a", "model-b")
        }
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "benchmark"
            selection = BenchmarkSelection(
                simulations=("counter-v1", "phrase-v1"),
                runtimes=("model-a", "model-b"),
                output_directory=output_directory,
                retain_diagnostic_artifacts=False,
            )

            prepared = preflight_benchmark(
                selection,
                simulations=simulations,
                runtimes=factories,
            )
            topline = await run_benchmark(prepared)

            self.assertEqual(
                [(entry.simulation_id, entry.model_id) for entry in topline.entries],
                [
                    ("counter-v1", "model-a"),
                    ("counter-v1", "model-b"),
                    ("phrase-v1", "model-a"),
                    ("phrase-v1", "model-b"),
                ],
            )
            self.assertEqual(
                [entry.primary_score.value for entry in topline.entries],
                [3.0, 3.0, 5.0, 5.0],
            )
            self.assertTrue(all(entry.primary_score.higher_is_better for entry in topline.entries))
            self.assertEqual(
                set(topline.model_dump()),
                {"schema_version", "status", "entries"},
            )
            self.assertEqual(factories["model-a"].call_count, 2)
            self.assertEqual(factories["model-b"].call_count, 2)
            reloaded = type(topline).model_validate_json(
                (output_directory / "topline.json").read_text(encoding="utf-8")
            )
            self.assertEqual(reloaded, topline)

    async def test_missing_evidence_prevents_complete_topline(self) -> None:
        factory = _RecordingRuntimeFactory(
            _runtime_descriptor(
                "model-a",
                capabilities=frozenset({RuntimeCapability.TEXT_OUTPUT}),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "benchmark"
            prepared = preflight_benchmark(
                BenchmarkSelection(
                    simulations=("missing-evidence-v1",),
                    runtimes=("model-a",),
                    output_directory=output_directory,
                ),
                simulations={
                    "missing-evidence-v1": _MissingEvidenceSimulation(),
                },
                runtimes={"model-a": factory},
            )

            with self.assertRaisesRegex(
                EvidenceValidationError,
                "evidence manifest does not exist",
            ):
                await run_benchmark(prepared)

            self.assertEqual(factory.call_count, 1)
            self.assertFalse((output_directory / "topline.json").exists())

    async def test_invalid_evidence_prevents_complete_topline(self) -> None:
        factory = _RecordingRuntimeFactory(
            _runtime_descriptor(
                "model-a",
                capabilities=frozenset({RuntimeCapability.TEXT_OUTPUT}),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "benchmark"
            prepared = preflight_benchmark(
                BenchmarkSelection(
                    simulations=("invalid-evidence-v1",),
                    runtimes=("model-a",),
                    output_directory=output_directory,
                ),
                simulations={
                    "invalid-evidence-v1": _InvalidEvidenceSimulation(),
                },
                runtimes={"model-a": factory},
            )

            with self.assertRaisesRegex(
                EvidenceValidationError,
                "evidence manifest hash does not match",
            ):
                await run_benchmark(prepared)

            self.assertEqual(factory.call_count, 1)
            self.assertFalse((output_directory / "topline.json").exists())


class CompatibilityPreflightTests(unittest.TestCase):
    def test_reports_every_incompatibility_before_calls_or_writes(self) -> None:
        factory = _RecordingRuntimeFactory(_runtime_descriptor("model-z", capabilities=frozenset()))
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "benchmark"
            selection = BenchmarkSelection(
                simulations=("demanding-v1",),
                runtimes=("model-z",),
                output_directory=output_directory,
            )

            with self.assertRaises(CompatibilityPreflightError) as raised:
                preflight_benchmark(
                    selection,
                    simulations={"demanding-v1": _DemandingSimulation()},
                    runtimes={"model-z": factory},
                )

            self.assertEqual(
                [
                    (issue.simulation_id, issue.model_id, issue.code)
                    for issue in raised.exception.issues
                ],
                [
                    (
                        "demanding-v1",
                        "model-z",
                        "missing-capability:strict-structured-output",
                    ),
                    ("demanding-v1", "model-z", "missing-capability:text-output"),
                    (
                        "demanding-v1",
                        "model-z",
                        "missing-sampling-control:temperature",
                    ),
                    ("demanding-v1", "model-z", "insufficient-concurrency"),
                ],
            )
            self.assertEqual(factory.call_count, 0)
            self.assertEqual(factory.instances, [])
            self.assertFalse(output_directory.exists())

    def test_missing_environment_fails_without_reading_a_secret_or_constructing_runtime(
        self,
    ) -> None:
        factory = _RecordingRuntimeFactory(
            _runtime_descriptor(
                "model-z",
                capabilities=frozenset({RuntimeCapability.TEXT_OUTPUT}),
                required_environment=("MODEL_CREDENTIAL",),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "benchmark"

            with self.assertRaises(CompatibilityPreflightError) as raised:
                preflight_benchmark(
                    BenchmarkSelection(
                        simulations=("counter-v1",),
                        runtimes=("model-z",),
                        output_directory=output_directory,
                    ),
                    simulations={"counter-v1": _CounterSimulation()},
                    runtimes={"model-z": factory},
                    environment={},
                )

            self.assertEqual(
                [issue.code for issue in raised.exception.issues],
                ["missing-environment:MODEL_CREDENTIAL"],
            )
            self.assertEqual(factory.instances, [])
            self.assertFalse(output_directory.exists())


if __name__ == "__main__":
    unittest.main()
