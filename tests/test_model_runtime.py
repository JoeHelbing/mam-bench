import os
import unittest
from typing import cast
from unittest.mock import AsyncMock, patch

from openai import omit
from openai.types.chat import ChatCompletion
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, TextPart, ToolCallPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.tools import ToolDefinition

from mam_bench.agent import AgentSessionRuntime
from mam_bench.benchmark import AgentSettings, create_runtime
from mam_bench.config import OpenAICompatibleModel
from mam_bench.config import OpenRouterModel as OpenRouterSelection


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
        self.assertIs(sent["parallel_tool_calls"], False)
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
                [ModelRequest(parts=[UserPromptPart("Respond.")])],
                None,
                ModelRequestParameters(),
            )

        sent = cast(dict[str, object], request.call_args.kwargs)
        self.assertEqual(sent["reasoning_effort"], "medium")
        self.assertEqual(sent["extra_body"], {"top_k": 20})
        self.assertEqual(sent["max_completion_tokens"], 32_768)
        self.assertIs(sent["max_tokens"], omit)
        self.assertIsInstance(response.parts[0], TextPart)

    async def test_summary_inherits_provider_defaults_without_actor_seed(self) -> None:
        model = self.router_model()
        sessions = AgentSessionRuntime(
            Agent(model, output_type=str),
            settings=AgentSettings(
                context_window_tokens=200,
                compaction_trigger_fraction=0.5,
                compaction_tail_tokens=20,
                summary_completion_tokens=16,
                memory_injection_tokens=100,
            ),
        )
        request = AsyncMock(return_value=completion())
        with patch.object(model.client.chat.completions, "create", request):
            await sessions.run("actor-7", "A" * 600, deps=None, model_settings={"seed": 42})
            await sessions.run("actor-7", "B" * 600, deps=None, model_settings={"seed": 43})

        calls = [cast(dict[str, object], call.kwargs) for call in request.call_args_list]
        self.assertEqual([call["max_tokens"] for call in calls], [32_768, 16, 32_768])
        self.assertEqual([call["seed"] for call in calls], [42, omit, 43])
        self.assertEqual(calls[0]["extra_body"], calls[1]["extra_body"])
        self.assertEqual(calls[1]["extra_body"], calls[2]["extra_body"])
        self.assertTrue(all(call["timeout"] == 3_600.0 for call in calls))


if __name__ == "__main__":
    unittest.main()
