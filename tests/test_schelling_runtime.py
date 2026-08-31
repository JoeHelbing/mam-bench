import json
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast

from pydantic import JsonValue
from pydantic_ai.messages import (
    ModelRequest as PydanticModelRequest,
)
from pydantic_ai.messages import (
    ModelResponse as PydanticModelResponse,
)
from pydantic_ai.messages import (
    ToolCallPart as PydanticToolCallPart,
)
from pydantic_ai.usage import RequestUsage

from mam_bench.benchmark import (
    SimulationDescriptor,
    SimulationResult,
)
from mam_bench.pydantic_runtime import (
    PydanticModelRuntime,
    PydanticRuntimeSettings,
)
from mam_bench.runner import (
    BenchmarkSelection,
    CompatibilityPreflightError,
    preflight_benchmark,
)
from mam_bench.runtime import (
    ModelRequest,
    ModelResponse,
    ModelRuntime,
    ModelUsage,
    RequestLimits,
    RuntimeCapability,
    RuntimeDescriptor,
    RuntimeInfrastructureError,
    RuntimeMessage,
    SamplingControl,
    SamplingSettings,
    StructuredOutput,
    TextOutput,
    TextPart,
    ToolCallPart,
    ToolResultPart,
)
from mam_bench.simulations.schelling.evaluation import (
    COORDINATION_PHASE_CODE,
    InfluenceEvaluationConfig,
    InfluenceInfrastructureError,
    SteeringObjective,
    model_sampling_seed,
    run_model_evaluation,
)
from mam_bench.simulations.schelling.interaction import (
    SCHELLING_RUNTIME_REQUIREMENTS,
    RuntimeInfluenceTeam,
)
from mam_bench.simulations.schelling.profile import landscape_cell


class _ScriptedRuntime:
    descriptor = RuntimeDescriptor(
        model_id="scripted-schelling",
        runtime="scripted",
        provider="test",
        model="scripted/schelling",
        adapter_version="scripted-v1",
        capabilities=frozenset(RuntimeCapability),
        sampling_controls=frozenset(SamplingControl),
        max_concurrent_requests=16,
    )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        parts: tuple[TextPart | ToolCallPart, ...]
        output: JsonValue
        if request.request_id.endswith(":state"):
            state_output: dict[str, JsonValue] = {
                "board_round_offset": 0,
                "coordination_round_offset": 0,
                "include_reference": False,
            }
            output = state_output
            parts = (
                ToolCallPart(
                    call_id=f"{request.request_id}:call",
                    name="inspect_state",
                    arguments=state_output,
                ),
            )
        elif isinstance(request.output, TextOutput):
            round_number = int(request.request_id.split(":")[1][1:])
            output = f"round {round_number}"
            parts = (TextPart(text=output),)
        else:
            move_output: dict[str, JsonValue] = {
                "stay": True,
                "row": None,
                "column": None,
            }
            output = move_output
            parts = (
                ToolCallPart(
                    call_id=f"{request.request_id}:call",
                    name="submit_move",
                    arguments=move_output,
                ),
            )
        return ModelResponse(
            request_id=request.request_id,
            messages=(RuntimeMessage(role="assistant", parts=parts),),
            output=output,
            usage=ModelUsage(requests=1, input_tokens=10, output_tokens=2),
            latency_seconds=0.0,
            provider_response_id=None,
        )


class _PartiallyFailingRuntime(_ScriptedRuntime):
    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.request_id == "schelling:r1:p0:a0:action":
            raise RuntimeInfrastructureError("endpoint unavailable")
        return await super().complete(request)


class _FakePydanticModel:
    def __init__(self) -> None:
        self.calls: list[tuple[list[object], object, object]] = []

    async def request(
        self,
        messages: list[object],
        model_settings: object,
        model_request_parameters: object,
    ) -> PydanticModelResponse:
        self.calls.append((messages, model_settings, model_request_parameters))
        return PydanticModelResponse(
            parts=(
                PydanticToolCallPart(
                    tool_name="submit_move",
                    args={"stay": True, "row": None, "column": None},
                    tool_call_id="call-1",
                ),
            ),
            usage=RequestUsage(input_tokens=7, output_tokens=3),
            model_name="test-model",
            provider_name="test-provider",
            provider_response_id="response-1",
        )


class _RuntimeFactory:
    def __init__(self, descriptor: RuntimeDescriptor) -> None:
        self.descriptor = descriptor

    def create(self) -> ModelRuntime:
        raise AssertionError("incompatible Model Runtime must not be created")


class _PreparedRequirements:
    descriptor = SimulationDescriptor(
        simulation_id="schelling-requirements-test",
        simulation_version="test-v1",
        title="Schelling requirements test",
        primary_score_name="directional_lift",
    )
    runtime_requirements = SCHELLING_RUNTIME_REQUIREMENTS

    async def execute(
        self,
        *,
        runtime: ModelRuntime,
        output_directory: Path,
        retain_diagnostic_artifacts: bool,
    ) -> None:
        del runtime, output_directory, retain_diagnostic_artifacts
        raise AssertionError("incompatible simulation must not execute")


class _RequirementsSimulation:
    descriptor = _PreparedRequirements.descriptor

    def prepare(self) -> _PreparedRequirements:
        return _PreparedRequirements()

    def validate(
        self,
        *,
        output_directory: Path,
        runtime: RuntimeDescriptor,
    ) -> SimulationResult:
        del output_directory, runtime
        raise AssertionError("incompatible simulation must not validate")


class PydanticModelRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_adapter_translates_strict_request_and_all_sampling_controls(
        self,
    ) -> None:
        model = _FakePydanticModel()
        runtime = PydanticModelRuntime(
            PydanticRuntimeSettings(),
            model=cast(Any, model),
        )
        request = ModelRequest(
            request_id="request-1",
            messages=(
                RuntimeMessage(
                    role="system",
                    parts=(TextPart(text="standing instructions"),),
                ),
                RuntimeMessage(
                    role="user",
                    parts=(TextPart(text="choose a move"),),
                ),
            ),
            output=StructuredOutput(
                name="submit_move",
                description="Submit one move",
                json_schema={
                    "type": "object",
                    "properties": {
                        "stay": {"type": "boolean"},
                        "row": {"type": ["integer", "null"]},
                        "column": {"type": ["integer", "null"]},
                    },
                    "required": ["stay", "row", "column"],
                    "additionalProperties": False,
                },
            ),
            sampling=SamplingSettings(
                temperature=1.0,
                top_p=0.95,
                top_k=20,
                reasoning_effort="medium",
                max_output_tokens=4096,
                request_seed=1234,
            ),
            limits=RequestLimits(
                request_limit=1,
                tool_call_limit=1,
                timeout_seconds=600.0,
            ),
        )

        response = await runtime.complete(request)

        self.assertEqual(response.request_id, "request-1")
        self.assertEqual(
            response.output,
            {"stay": True, "row": None, "column": None},
        )
        self.assertEqual(response.usage.input_tokens, 7)
        self.assertEqual(response.usage.output_tokens, 3)
        self.assertEqual(response.provider_response_id, "response-1")
        self.assertIsInstance(response.messages[0].parts[0], ToolCallPart)
        messages, settings, parameters = model.calls[0]
        self.assertTrue(all(isinstance(message, PydanticModelRequest) for message in messages))
        settings_dict = dict(cast(Any, settings))
        self.assertEqual(settings_dict["temperature"], 1.0)
        self.assertEqual(settings_dict["top_p"], 0.95)
        self.assertEqual(settings_dict["max_tokens"], 4096)
        self.assertEqual(settings_dict["seed"], 1234)
        self.assertEqual(settings_dict["timeout"], 600.0)
        self.assertEqual(settings_dict["openai_reasoning_effort"], "medium")
        self.assertEqual(
            settings_dict["extra_body"],
            {
                "top_k": 20,
                "provider": {
                    "only": ["phala"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
            },
        )
        typed_parameters = cast(Any, parameters)
        self.assertEqual(typed_parameters.output_mode, "tool")
        self.assertFalse(typed_parameters.allow_text_output)
        self.assertEqual(typed_parameters.output_tools[0].name, "submit_move")
        self.assertTrue(typed_parameters.output_tools[0].strict)

        local_model = _FakePydanticModel()
        local_runtime = PydanticModelRuntime(
            PydanticRuntimeSettings(
                provider="openai-compatible",
                base_url="http://127.0.0.1:8000/v1",
                credential_environment="LOCAL_MODEL_API_KEY",
            ),
            model=cast(Any, local_model),
        )
        await local_runtime.complete(request)
        local_settings = dict(cast(Any, local_model.calls[0][1]))
        self.assertFalse(local_settings["parallel_tool_calls"])
        self.assertEqual(local_settings["extra_body"], {"top_k": 20})


class RuntimeInfluenceTeamTests(unittest.IsolatedAsyncioTestCase):
    async def test_scripted_runtime_completes_exact_schelling_protocol(self) -> None:
        runtime = _ScriptedRuntime()
        team = RuntimeInfluenceTeam(runtime)
        config = InfluenceEvaluationConfig(
            cell=landscape_cell(17, 3),
            seed_id=22,
            objective=SteeringObjective.INTEGRATION,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "case"

            result = await run_model_evaluation(config, team, output_directory)

            self.assertEqual(result.rounds_completed, 20)
            self.assertEqual(len(runtime.requests), 20 * 2 * 16 * 2)
            self.assertEqual(
                len({request.request_id for request in runtime.requests}),
                len(runtime.requests),
            )
            first_state, first_action = runtime.requests[:2]
            expected_seed = model_sampling_seed(
                config,
                round_number=1,
                phase_code=COORDINATION_PHASE_CODE,
                actor_id=0,
            )
            self.assertIsInstance(first_state.output, StructuredOutput)
            state_output = cast(StructuredOutput, first_state.output)
            self.assertEqual(state_output.name, "inspect_state")
            self.assertEqual(first_state.sampling.request_seed, expected_seed)
            self.assertIsInstance(first_action.output, TextOutput)
            self.assertEqual(first_action.sampling.request_seed, expected_seed)
            self.assertEqual(first_state.limits.request_limit, 1)
            self.assertEqual(first_action.limits.request_limit, 1)
            self.assertTrue(
                any(
                    isinstance(part, ToolResultPart)
                    for message in first_action.messages
                    for part in message.parts
                )
            )
            history = json.loads(
                (output_directory / "actor-histories" / "actor-000.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertTrue(history)
            self.assertEqual(history[0]["role"], "user")

    async def test_infrastructure_failure_commits_no_partial_wave_history(
        self,
    ) -> None:
        team = RuntimeInfluenceTeam(_PartiallyFailingRuntime())
        config = InfluenceEvaluationConfig(
            cell=landscape_cell(17, 3),
            seed_id=22,
            objective=SteeringObjective.INTEGRATION,
        )
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "case"

            with self.assertRaises(InfluenceInfrastructureError):
                await run_model_evaluation(config, team, output_directory)

            histories = [
                json.loads(path.read_text(encoding="utf-8"))
                for path in sorted((output_directory / "actor-histories").glob("*.json"))
            ]
            self.assertEqual(histories, [[] for _ in range(16)])


class SchellingRuntimePreflightTests(unittest.TestCase):
    def test_schelling_requirements_fail_complete_preflight(self) -> None:
        descriptor = RuntimeDescriptor(
            model_id="limited",
            runtime="limited",
            provider="test",
            model="test/limited",
            adapter_version="limited-v1",
            capabilities=frozenset({RuntimeCapability.TEXT_OUTPUT}),
            sampling_controls=frozenset({SamplingControl.TEMPERATURE}),
            max_concurrent_requests=1,
        )
        with tempfile.TemporaryDirectory() as directory:
            selection = BenchmarkSelection(
                simulations=("schelling-requirements-test",),
                runtimes=("limited",),
                output_directory=Path(directory) / "output",
            )

            with self.assertRaises(CompatibilityPreflightError) as raised:
                preflight_benchmark(
                    selection,
                    simulations={
                        "schelling-requirements-test": _RequirementsSimulation(),
                    },
                    runtimes={"limited": _RuntimeFactory(descriptor)},
                )

            self.assertEqual(
                [issue.code for issue in raised.exception.issues],
                [
                    "missing-capability:auditable-messages",
                    "missing-capability:request-seed",
                    "missing-capability:strict-structured-output",
                    "missing-capability:tool-result-continuation",
                    "missing-sampling-control:max-output-tokens",
                    "missing-sampling-control:reasoning-effort",
                    "missing-sampling-control:top-k",
                    "missing-sampling-control:top-p",
                    "insufficient-concurrency",
                ],
            )


if __name__ == "__main__":
    unittest.main()
