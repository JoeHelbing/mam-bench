import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

from pydantic_ai import ModelMessage
from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.usage import RequestUsage

from mam_bench.benchmark import ModelRuntime, RuntimeInfo, create_runtime
from mam_bench.config import OpenAICompatibleModel, OpenRouterModel
from mam_bench.simulations.schelling.agent import RuntimeInfluenceTeam
from mam_bench.simulations.schelling.models import (
    InfluenceEvaluationConfig,
    SteeringObjective,
)
from mam_bench.simulations.schelling.profile import LandscapeCell, Rational
from mam_bench.simulations.schelling.runtime import run_model_evaluation
from mam_bench.simulations.schelling.utils.reference_data import build_evaluation_reference


class ScriptedModel:
    def __init__(
        self,
        *,
        malformed_first_move: bool = False,
        coemit_first_move: bool = False,
    ) -> None:
        self.call_count = 0
        self.malformed_first_move = malformed_first_move
        self.coemit_first_move = coemit_first_move
        self._malformed_move_sent = False
        self.coemitted_move_sent = False
        self.active_calls = 0
        self.maximum_active_calls = 0
        self.seeds: list[int | None] = []
        self.inspection_call_count = 0
        self.inspection_results: list[str] = []

    async def __call__(
        self,
        messages: list[ModelMessage],
        info: AgentInfo,
    ) -> ModelResponse:
        self.call_count += 1
        self.active_calls += 1
        self.maximum_active_calls = max(self.maximum_active_calls, self.active_calls)
        await asyncio.sleep(0)
        settings = cast(dict[str, Any], info.model_settings)
        self.seeds.append(settings.get("seed"))
        output_name = info.output_tools[0].name if info.output_tools else None
        latest_inspection = [
            cast(str, part.content)
            for part in messages[-1].parts
            if isinstance(messages[-1], ModelRequest)
            and isinstance(part, ToolReturnPart)
            and part.tool_name == "inspect_state"
        ]

        if not latest_inspection:
            self.assert_inspect_state_tool(info)
            self.inspection_call_count += 1
            inspect_call = ToolCallPart(
                tool_name="inspect_state",
                args={
                    "board_round_offset": 0,
                    "coordination_round_offset": 0,
                    "include_reference": False,
                },
                tool_call_id=f"inspect-{self.call_count}",
            )
            if output_name == "submit_move" and self.coemit_first_move:
                self.coemit_first_move = False
                self.coemitted_move_sent = True
                parts = [
                    inspect_call,
                    ToolCallPart(
                        tool_name="submit_move",
                        args={"stay": True, "row": None, "column": None},
                        tool_call_id=f"move-{self.call_count}",
                    ),
                ]
            else:
                parts = [inspect_call]
        elif output_name == "submit_move":
            self.inspection_results.append(latest_inspection[0])
            if self.malformed_first_move and not self._malformed_move_sent:
                self._malformed_move_sent = True
                parts = [TextPart("not a structured move")]
            else:
                parts = [
                    ToolCallPart(
                        tool_name="submit_move",
                        args={"stay": True, "row": None, "column": None},
                        tool_call_id=f"move-{self.call_count}",
                    )
                ]
        else:
            self.inspection_results.append(latest_inspection[0])
            parts = [TextPart("coordinate")]

        self.active_calls -= 1
        return ModelResponse(
            parts=parts,
            usage=RequestUsage(input_tokens=10, output_tokens=2),
            model_name="scripted",
            provider_name="test",
        )

    @staticmethod
    def assert_inspect_state_tool(info: AgentInfo) -> None:
        assert [tool.name for tool in info.function_tools] == ["inspect_state"]


def scripted_runtime(script: ScriptedModel) -> ModelRuntime:
    return ModelRuntime(
        info=RuntimeInfo(
            model_id="scripted",
            provider="openrouter",
            model="test/scripted",
            endpoint="https://example.test/v1",
            routing_provider="provider-a",
        ),
        model=FunctionModel(script),
    )


class RuntimeConstructionTests(unittest.TestCase):
    def test_builds_both_pydantic_ai_provider_paths(self) -> None:
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "test",
                "OPENROUTER_BASE_URL": "https://router.example.test/v1",
                "LOCAL_API_KEY": "test",
            },
        ):
            openrouter = create_runtime(
                OpenRouterModel(
                    id="remote",
                    runtime="openrouter",
                    model="vendor/model",
                    provider="provider-a",
                )
            )
            local = create_runtime(
                OpenAICompatibleModel(
                    id="local",
                    runtime="openai-compatible",
                    model="local/model",
                    base_url="http://127.0.0.1:8000/v1",
                    api_key_env="LOCAL_API_KEY",
                )
            )

        self.assertIsInstance(openrouter.model, OpenAIChatModel)
        self.assertEqual(openrouter.info.endpoint, "https://router.example.test/v1")
        self.assertEqual(openrouter.info.routing_provider, "provider-a")
        self.assertIsInstance(local.model, OpenAIChatModel)
        self.assertEqual(local.info.endpoint, "http://127.0.0.1:8000/v1")


class RuntimeInfluenceTeamTests(unittest.IsolatedAsyncioTestCase):
    async def test_agents_run_the_two_request_protocol_with_bounded_concurrency(self) -> None:
        script = ScriptedModel(malformed_first_move=True, coemit_first_move=True)
        with tempfile.TemporaryDirectory() as directory:
            config = InfluenceEvaluationConfig(
                cell=LandscapeCell(20, Rational(3, 4), Rational(1, 4)),
                seed_id=50,
                objective=SteeringObjective.INTEGRATION,
            )
            result = await run_model_evaluation(
                config,
                RuntimeInfluenceTeam(scripted_runtime(script)),
                Path(directory) / "case",
                reference=build_evaluation_reference(config.cell, config.seed_id),
            )

        self.assertEqual(result.rounds_completed, 20)
        self.assertEqual(result.actor_invalid_action_count, 2)
        self.assertTrue(script.coemitted_move_sent)
        self.assertEqual(script.call_count, 20 * 16 * 4 - 1)
        self.assertEqual(script.maximum_active_calls, 4)
        self.assertTrue(all(seed is not None for seed in script.seeds))
        self.assertEqual(script.inspection_call_count, 20 * 16 * 2)
        self.assertEqual(len(script.inspection_results), 20 * 16 * 2 - 1)
        self.assertTrue(
            all(result != "Final result processed." for result in script.inspection_results)
        )

    async def test_provider_errors_are_redacted(self) -> None:
        async def fail(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del messages, info
            raise OSError("Authorization: Bearer secret-token")

        runtime = ModelRuntime(
            info=RuntimeInfo(
                model_id="failing",
                provider="openrouter",
                model="test/failing",
                endpoint="https://example.test/v1",
                routing_provider="provider-a",
            ),
            model=FunctionModel(fail),
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            self.assertRaisesRegex(RuntimeError, "REDACTED") as raised,
        ):
            config = InfluenceEvaluationConfig(
                cell=LandscapeCell(20, Rational(3, 4), Rational(1, 4)),
                seed_id=50,
                objective=SteeringObjective.INTEGRATION,
            )
            await run_model_evaluation(
                config,
                RuntimeInfluenceTeam(runtime),
                Path(directory) / "case",
                reference=build_evaluation_reference(config.cell, config.seed_id),
            )

        self.assertNotIn("secret-token", str(raised.exception))
        self.assertIsNone(raised.exception.__cause__)


if __name__ == "__main__":
    unittest.main()
