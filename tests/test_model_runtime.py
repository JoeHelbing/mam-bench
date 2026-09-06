import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from typing import Literal, cast
from unittest.mock import AsyncMock, patch

from openai import omit
from openai.types.chat import ChatCompletion
from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelRequest,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.tools import ToolDefinition

from mam_bench.agent import AgentSessionRuntime
from mam_bench.config import AgentSettings, OpenAICompatibleModel, load_benchmark_config
from mam_bench.config import OpenRouterModel as OpenRouterSelection
from mam_bench.model import create_runtime
from mam_bench.simulations.schelling import SchellingSim


def completion(*, tool_call: bool = False) -> ChatCompletion:
    return ChatCompletion.model_validate(
        {
            "id": "offline-response",
            "object": "chat.completion",
            "created": 1,
            "model": "vendor/test-model",
            "provider": "selected-provider",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "tool_calls" if tool_call else "stop",
                    "message": {
                        "role": "assistant",
                        "content": None if tool_call else "Summary or response.",
                        "tool_calls": [
                            {
                                "id": "stay-1",
                                "type": "function",
                                "function": {"name": "stay", "arguments": "{}"},
                            }
                        ]
                        if tool_call
                        else None,
                    },
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
        }
    )


@patch.dict(
    os.environ,
    {
        "OPENROUTER_API_KEY": "unused-offline-test-value",
        "OPENROUTER_BASE_URL": "https://router.example.test/v1",
        "COMPAT_TEST_API_KEY": "unused-offline-test-value",
    },
)
class ModelRuntimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_configured_tool_choice_preserves_terminal_tools_and_retries_text(self) -> None:
        choices: tuple[Literal["required", "auto"], ...] = ("required", "auto")
        for choice in choices:
            settings = AgentSettings.model_validate({"tool_choice": choice})
            for selection in (
                OpenRouterSelection(
                    id="choice-test",
                    runtime="openrouter",
                    model="vendor/test-model",
                    provider="selected-provider",
                    settings=settings,
                ),
                OpenAICompatibleModel(
                    id="choice-test",
                    runtime="openai-compatible",
                    model="test-model",
                    base_url="https://compatible.example.test/v1",
                    api_key_env="COMPAT_TEST_API_KEY",
                    settings=settings,
                ),
            ):
                with self.subTest(choice=choice, provider=selection.runtime):
                    runtime = create_runtime(selection)
                    model = runtime.model
                    assert isinstance(model, OpenAIChatModel)
                    self.addAsyncCleanup(model.client.close)
                    sim = SchellingSim.for_model(runtime)
                    # A text response must trigger a retry, not complete an actor turn.
                    responses = [completion(), *[completion(tool_call=True) for _ in range(16)]]
                    request = AsyncMock(side_effect=responses)
                    with patch.object(model.client.chat.completions, "create", request):
                        result = await sim.step()
                    assert result is not None
                    self.assertEqual(request.await_count, 17)
                    self.assertEqual(result.controlled_moves, ())
                    for call in request.call_args_list:
                        self.assertEqual(call.kwargs["tool_choice"], choice)
                        names = {tool["function"]["name"] for tool in call.kwargs["tools"]}
                        self.assertTrue({"move", "stay", "post_message"} <= names)
                    self.assertEqual(
                        runtime.info.agent_settings.model_dump()["tool_choice"], choice
                    )

    async def test_custom_settings_reach_requests_and_metadata(self) -> None:
        settings = AgentSettings(
            concurrency=4,
            context_window_tokens=65536,
            max_completion_tokens=8192,
            summary_completion_tokens=8192,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.yaml"
            path.write_text(
                "simulations: [schelling-influence-pilot-v1]\n"
                "models:\n"
                "  - id: offline-model\n"
                "    runtime: openrouter\n"
                "    model: vendor/test-model\n"
                "    provider: selected-provider\n"
                "    settings:\n"
                "      context_window_tokens: 65536\n"
                "      max_completion_tokens: 8192\n"
                "      summary_completion_tokens: 8192\n"
                "output_directory: results\n",
                encoding="utf-8",
            )
            config = load_benchmark_config(path)
        runtime = create_runtime(config.models[0])
        model = runtime.model
        assert isinstance(model, OpenRouterModel)
        self.addAsyncCleanup(model.client.close)
        self.assertEqual(runtime.info.agent_settings, settings)
        request = AsyncMock(return_value=completion())
        with patch.object(model.client.chat.completions, "create", request):
            await model.request(
                [ModelRequest(parts=[UserPromptPart("Respond.")])],
                None,
                ModelRequestParameters(),
            )
        self.assertEqual(request.call_args.kwargs["max_tokens"], 8192)

        active = maximum = 0

        async def stay(**kwargs: object) -> ChatCompletion:
            nonlocal active, maximum
            active += 1
            maximum = max(active, maximum)
            try:
                await asyncio.sleep(0.01)
                return completion(tool_call=True)
            finally:
                active -= 1

        sim = SchellingSim.for_model(runtime)
        assert sim.sessions is not None
        self.assertEqual(sim.sessions.settings, settings)
        with patch.object(model.client.chat.completions, "create", AsyncMock(side_effect=stay)):
            await sim.step()
        self.assertEqual(maximum, 4)

    def router_model(self) -> OpenRouterModel:
        runtime = create_runtime(
            OpenRouterSelection(
                id="test-model",
                runtime="openrouter",
                model="vendor/test-model",
                provider="selected-provider",
            )
        )
        model = runtime.model
        assert isinstance(model, OpenRouterModel)
        self.addAsyncCleanup(model.client.close)
        self.assertEqual(runtime.info.endpoint, "https://router.example.test/v1")
        self.assertEqual(str(model.client.base_url), "https://router.example.test/v1/")
        self.assertEqual(model.client.default_headers["X-Title"], "MAM-Bench")
        return model

    async def test_venice_config_omits_sampling_seed_but_requires_tools(self) -> None:
        config = load_benchmark_config(
            Path(__file__).resolve().parents[1] / "examples/schelling-qwen9b-venice.yaml"
        )
        runtime = create_runtime(config.models[0])
        model = runtime.model
        assert isinstance(model, OpenRouterModel)
        self.addAsyncCleanup(model.client.close)
        sim = SchellingSim.for_model(runtime)
        request = AsyncMock(return_value=completion(tool_call=True))
        with patch.object(model.client.chat.completions, "create", request):
            await sim.step()
        self.assertEqual(request.await_count, 16)
        for call in request.call_args_list:
            self.assertIs(call.kwargs["seed"], omit)
            self.assertEqual(call.kwargs["tool_choice"], "required")
            self.assertEqual(call.kwargs["max_tokens"], 32768)
            self.assertEqual(call.kwargs["extra_body"]["provider"]["only"], ["venice"])
        self.assertFalse(runtime.info.agent_settings.use_sampling_seed)

    async def test_openrouter_sends_native_routing_reasoning_and_terminal_tool(self) -> None:
        model = self.router_model()
        request = AsyncMock(return_value=completion(tool_call=True))
        with patch.object(model.client.chat.completions, "create", request):
            response = await model.request(
                [ModelRequest(parts=[UserPromptPart("Choose an action.")])],
                {"seed": 42},
                ModelRequestParameters(
                    output_mode="tool",
                    output_tools=[
                        ToolDefinition(
                            name="stay",
                            parameters_json_schema={"type": "object", "properties": {}},
                            strict=True,
                        )
                    ],
                    allow_text_output=False,
                ),
            )

        request.assert_awaited_once()
        sent = cast(dict[str, object], request.call_args.kwargs)
        self.assertEqual(
            sent["extra_body"],
            {
                "top_k": 20,
                "provider": {
                    "only": ["selected-provider"],
                    "allow_fallbacks": False,
                    "require_parameters": True,
                },
                "reasoning": {"effort": "medium"},
            },
        )
        self.assertEqual(sent["max_tokens"], 32_768)
        self.assertIs(sent["max_completion_tokens"], omit)
        self.assertEqual(sent["temperature"], 1.0)
        self.assertEqual(sent["top_p"], 0.95)
        self.assertEqual(sent["timeout"], 3_600.0)
        self.assertEqual(sent["seed"], 42)
        self.assertIsNot(sent.get("parallel_tool_calls"), False)
        self.assertEqual(sent["tool_choice"], "required")
        tools = cast(list[dict[str, object]], sent["tools"])
        function = cast(dict[str, object], tools[0]["function"])
        self.assertEqual(function["name"], "stay")
        self.assertIs(function["strict"], True)
        self.assertIsInstance(response.parts[0], ToolCallPart)
        self.assertEqual(response.usage.total_tokens, 12)

    async def test_compatible_endpoint_keeps_openai_request_format(self) -> None:
        runtime = create_runtime(
            OpenAICompatibleModel(
                id="local-model",
                runtime="openai-compatible",
                model="test-model",
                base_url="https://compatible.example.test/v1",
                api_key_env="COMPAT_TEST_API_KEY",
                settings=AgentSettings(
                    temperature=0.5,
                    top_p=0.8,
                    top_k=10,
                    reasoning_effort="low",
                    max_completion_tokens=2048,
                    timeout_seconds=45,
                ),
            )
        )
        model = runtime.model
        assert isinstance(model, OpenAIChatModel)
        self.assertNotIsInstance(model, OpenRouterModel)
        self.addAsyncCleanup(model.client.close)
        self.assertEqual(str(model.client.base_url), "https://compatible.example.test/v1/")
        request = AsyncMock(return_value=completion())
        with patch.object(model.client.chat.completions, "create", request):
            response = await model.request(
                [
                    ModelRequest(
                        parts=[
                            SystemPromptPart("Simulation instructions."),
                            SystemPromptPart("Notebook guidance."),
                            UserPromptPart("Respond."),
                        ]
                    )
                ],
                None,
                ModelRequestParameters(),
            )

        sent = cast(dict[str, object], request.call_args.kwargs)
        messages = cast(list[dict[str, object]], sent["messages"])
        self.assertEqual([message["role"] for message in messages], ["system", "user"])
        self.assertIn("Simulation instructions.", str(messages[0]["content"]))
        self.assertIn("Notebook guidance.", str(messages[0]["content"]))
        self.assertEqual(sent["reasoning_effort"], "low")
        self.assertEqual(sent["extra_body"], {"top_k": 10})
        self.assertEqual(sent["max_completion_tokens"], 2048)
        self.assertEqual(sent["temperature"], 0.5)
        self.assertEqual(sent["top_p"], 0.8)
        self.assertEqual(sent["timeout"], 45)
        self.assertIs(sent["max_tokens"], omit)
        self.assertIsInstance(response.parts[0], TextPart)

    async def test_summary_inherits_provider_defaults_without_actor_seed(self) -> None:
        settings = AgentSettings(
            context_window_tokens=200,
            compaction_trigger_fraction=0.5,
            compaction_tail_tokens=20,
            summary_completion_tokens=16,
            memory_injection_tokens=100,
        )
        runtime = create_runtime(
            OpenRouterSelection(
                id="summary-test",
                runtime="openrouter",
                model="vendor/test-model",
                provider="selected-provider",
                settings=settings,
            )
        )
        model = runtime.model
        assert isinstance(model, OpenRouterModel)
        self.addAsyncCleanup(model.client.close)
        sessions = AgentSessionRuntime(
            Agent(model, output_type=str),
            settings=runtime.info.agent_settings,
        )
        request = AsyncMock(return_value=completion())
        with patch.object(model.client.chat.completions, "create", request):
            await sessions.run("actor-7", "A" * 600, deps=None, model_settings={"seed": 42})
            await sessions.run("actor-7", "B" * 600, deps=None, model_settings={"seed": 43})

        calls = [cast(dict[str, object], call.kwargs) for call in request.call_args_list]
        self.assertEqual([call["max_tokens"] for call in calls], [32_768, 16, 32_768])
        self.assertIs(calls[1]["tool_choice"], omit)
        self.assertEqual([call["seed"] for call in calls], [42, omit, 43])
        self.assertEqual(calls[0]["extra_body"], calls[1]["extra_body"])
        self.assertEqual(calls[1]["extra_body"], calls[2]["extra_body"])
        self.assertTrue(all(call["timeout"] == 3_600.0 for call in calls))


if __name__ == "__main__":
    unittest.main()
