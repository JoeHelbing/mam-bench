import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Literal, cast
from unittest.mock import AsyncMock, patch

import httpx2 as httpx
from openai import AsyncOpenAI, omit
from openai.types.chat import ChatCompletion
from pydantic_ai import Agent
from pydantic_ai.exceptions import UnexpectedModelBehavior
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

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import AgentSettings, OpenAICompatibleModel
from mam_bench.config import OpenRouterModel as OpenRouterSelection
from mam_bench.runtime import CaseRuntime, create_model
from mam_bench.sessions import AgentSessions
from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim
from mam_bench.simulations.schelling.simulation import SchellingSim
from support import civil, schelling


def completion(
    *, tool_call: bool = False, tool_name: str = "stay", tool_args: str = "{}"
) -> ChatCompletion:
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
                                "function": {"name": tool_name, "arguments": tool_args},
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
class ModelProviderTests(unittest.IsolatedAsyncioTestCase):
    def router_model(self) -> OpenRouterModel:
        model = create_model(
            OpenRouterSelection(
                runtime="openrouter",
                model="vendor/test-model",
                provider="selected-provider",
            )
        )
        assert isinstance(model, OpenRouterModel)
        self.addAsyncCleanup(model.client.close)
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
        self.assertIsNot(sent.get("parallel_tool_calls"), False)
        self.assertEqual(sent["tool_choice"], "required")
        tools = cast(list[dict[str, object]], sent["tools"])
        function = cast(dict[str, object], tools[0]["function"])
        self.assertEqual(function["name"], "stay")
        self.assertIs(function["strict"], True)
        self.assertIsInstance(response.parts[0], ToolCallPart)
        self.assertEqual(response.usage.total_tokens, 12)

    async def test_completion_parameter_reaches_sdk_http_body(self) -> None:
        from mam_bench.model_setup import EndpointResponse, resolve_model_setup

        for parameter in ("max_tokens", "max_completion_tokens"):
            with self.subTest(parameter=parameter):
                payload = EndpointResponse.model_validate(
                    {
                        "data": {
                            "endpoints": [
                                {
                                    "tag": "azure",
                                    "provider_name": "Azure",
                                    "status": 0,
                                    "context_length": 1047576,
                                    "max_completion_tokens": 32768,
                                    "supported_parameters": ["tools", "tool_choice", parameter],
                                    "supports_tool_choice": {"required": True, "auto": True},
                                }
                            ]
                        }
                    }
                )
                with patch(
                    "mam_bench.model_setup._fetch_endpoints", AsyncMock(return_value=payload)
                ):
                    setup = await resolve_model_setup(
                        OpenRouterSelection(
                            runtime="openrouter", model="openai/gpt-4.1", provider="azure"
                        )
                    )
                requests: list[dict[str, object]] = []

                def respond(
                    request: httpx.Request, captured: list[dict[str, object]] = requests
                ) -> httpx.Response:
                    captured.append(cast(dict[str, object], json.loads(request.content)))
                    return httpx.Response(200, json=completion().model_dump(mode="json"))

                async with AsyncOpenAI(
                    api_key="unused-offline-test-value",
                    base_url="https://router.example.test/v1",
                    http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
                ) as client:
                    with patch("mam_bench.runtime.AsyncOpenAI", return_value=client):
                        model = create_model(setup.selection)
                    await model.request(
                        [ModelRequest(parts=[UserPromptPart("Choose an action.")])],
                        None,
                        ModelRequestParameters(
                            output_mode="tool",
                            allow_text_output=False,
                            output_tools=[
                                ToolDefinition(
                                    name="stay",
                                    parameters_json_schema={"type": "object", "properties": {}},
                                )
                            ],
                        ),
                    )
                self.assertEqual(len(requests), 1)
                self.assertEqual(requests[0][parameter], 32768)
                other = (
                    "max_tokens"
                    if parameter == "max_completion_tokens"
                    else "max_completion_tokens"
                )
                self.assertNotIn(other, requests[0])
                self.assertEqual(requests[0]["tool_choice"], "required")
                self.assertEqual(
                    requests[0]["provider"],
                    {"only": ["azure"], "allow_fallbacks": False, "require_parameters": True},
                )

    async def test_resolved_request_omits_defaults_and_reports_cache_presence(self) -> None:
        selected = OpenRouterSelection(
            runtime="openrouter", model="vendor/test-model", provider="selected-provider"
        ).with_setup(
            AgentSettings(),
            frozenset({"temperature", "top_p", "top_k", "reasoning"}),
            "unverified",
        )
        model = create_model(selected)
        assert isinstance(model, OpenRouterModel)
        self.addAsyncCleanup(model.client.close)
        for raw_details, expected in (
            (None, False),
            ({"cached_tokens": 0}, True),
            ({"cached_tokens": 4, "cache_write_tokens": 2}, True),
        ):
            raw = completion().model_dump()
            usage = cast(dict[str, object], raw["usage"])
            if raw_details is not None:
                usage["prompt_tokens_details"] = raw_details
            request = AsyncMock(return_value=ChatCompletion.model_validate(raw))
            with patch.object(model.client.chat.completions, "create", request):
                response = await model.request(
                    [ModelRequest(parts=[UserPromptPart("Respond.")])],
                    None,
                    ModelRequestParameters(),
                )
            sent = cast(dict[str, object], request.call_args.kwargs)
            self.assertIs(sent["temperature"], omit)
            self.assertIs(sent["top_p"], omit)
            self.assertEqual(
                sent["extra_body"],
                {
                    "provider": {
                        "only": ["selected-provider"],
                        "allow_fallbacks": False,
                        "require_parameters": True,
                    }
                },
            )
            self.assertEqual(
                (response.provider_details or {}).get("cache_usage_reported"), expected
            )

    async def test_explicit_cache_adds_native_breakpoints_to_claude_requests(self) -> None:
        selected = OpenRouterSelection(
            runtime="openrouter", model="anthropic/claude-sonnet-4", provider="anthropic"
        ).with_setup(AgentSettings(tool_choice="auto"), frozenset(), "explicit")
        model = create_model(selected)
        assert isinstance(model, OpenRouterModel)
        self.addAsyncCleanup(model.client.close)
        request = AsyncMock(return_value=completion())
        with patch.object(model.client.chat.completions, "create", request):
            await model.request(
                [
                    ModelRequest(
                        parts=[SystemPromptPart("Stable instructions."), UserPromptPart("Respond.")]
                    )
                ],
                None,
                ModelRequestParameters(),
            )
        sent = cast(dict[str, object], request.call_args.kwargs)
        messages = cast(list[dict[str, object]], sent["messages"])
        content = cast(list[dict[str, object]], messages[-1]["content"])
        self.assertEqual(content[-1]["cache_control"], {"type": "ephemeral", "ttl": "5m"})

    async def test_compatible_endpoint_keeps_openai_request_format(self) -> None:
        model = create_model(
            OpenAICompatibleModel(
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

    async def test_strict_parameterless_tools_reach_the_model_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            for tool_choice, strict in (
                ("required", False),
                ("required", True),
                ("auto", False),
                ("auto", True),
            ):
                with self.subTest(tool_choice=tool_choice, strict=strict):
                    settings = AgentSettings(
                        tool_choice=cast(Literal["required", "auto"], tool_choice),
                        strict_parameterless_tools=strict,
                    )
                    model = create_model(
                        OpenAICompatibleModel(
                            runtime="openai-compatible",
                            model="test-model",
                            base_url="https://compatible.example.test/v1",
                            api_key_env="COMPAT_TEST_API_KEY",
                            settings=settings,
                        )
                    )
                    assert isinstance(model, OpenAIChatModel)
                    self.addAsyncCleanup(model.client.close)
                    writer = ArtifactWriter(Path(directory) / f"{tool_choice}-{strict}")
                    sim = SchellingSim(
                        schelling(),
                        runtime=CaseRuntime(model=model, settings=settings, writer=writer),
                    )
                    request = AsyncMock(
                        return_value=completion(
                            tool_call=True,
                            tool_name="move",
                            tool_args='{"destination": null}',
                        )
                    )
                    with patch.object(model.client.chat.completions, "create", request):
                        await sim.step()

                    self.assertEqual(request.await_count, len(sim.controlled_agent_ids))
                    self.assertFalse((writer.directory / "turns.jsonl").exists())
                    for call in request.await_args_list:
                        sent = cast(dict[str, object], call.kwargs)
                        self.assertEqual(sent["tool_choice"], tool_choice)
                        tools = cast(list[dict[str, object]], sent["tools"])
                        functions = [cast(dict[str, object], tool["function"]) for tool in tools]
                        by_name = {cast(str, function["name"]): function for function in functions}
                        self.assertNotIn("stay", by_name)
                        self.assertEqual(
                            cast(dict[str, object], by_name["read_messages"]["parameters"])[
                                "properties"
                            ],
                            {},
                        )
                        self.assertEqual(
                            by_name["read_messages"].get("strict"), True if strict else None
                        )
                        for name in ("move", "post_message"):
                            self.assertIs(by_name[name].get("strict"), True)

    async def test_civil_citizen_defer_uses_per_model_strict_schema(self) -> None:
        for strict in (False, True):
            with self.subTest(strict=strict), tempfile.TemporaryDirectory() as directory:
                settings = AgentSettings(strict_parameterless_tools=strict)
                model = create_model(
                    OpenAICompatibleModel(
                        runtime="openai-compatible",
                        model="test-model",
                        base_url="https://compatible.example.test/v1",
                        api_key_env="COMPAT_TEST_API_KEY",
                        settings=settings,
                    )
                )
                assert isinstance(model, OpenAIChatModel)
                self.addAsyncCleanup(model.client.close)
                writer = ArtifactWriter(Path(directory) / "case")
                sim = CivilViolenceSim(
                    civil(), runtime=CaseRuntime(model=model, settings=settings, writer=writer)
                )
                for identity in sim.controlled_agent_ids:
                    sim.citizens[identity].location = None
                    sim.citizens[identity].jail_remaining = 1
                request = AsyncMock(return_value=completion(tool_call=True, tool_name="defer"))
                with patch.object(model.client.chat.completions, "create", request):
                    await sim.step()

                self.assertEqual(request.await_count, len(sim.controlled_agent_ids))
                for call in request.await_args_list:
                    sent = cast(dict[str, object], call.kwargs)
                    tools = cast(list[dict[str, object]], sent["tools"])
                    functions = [cast(dict[str, object], tool["function"]) for tool in tools]
                    by_name = {cast(str, function["name"]): function for function in functions}
                    self.assertEqual(sent["tool_choice"], "required")
                    for name in ("defer", "read_messages"):
                        self.assertEqual(
                            cast(dict[str, object], by_name[name]["parameters"])["properties"],
                            {},
                        )
                        self.assertEqual(by_name[name].get("strict"), True if strict else None)
                    self.assertIs(by_name["act"].get("strict"), True)
                    self.assertIs(by_name["post_message"].get("strict"), True)
                self.assertFalse((writer.directory / "turns.jsonl").exists())

    async def test_compatible_endpoint_accepts_sglang_weight_version_spans(self) -> None:
        model = create_model(
            OpenAICompatibleModel(
                runtime="openai-compatible",
                model="test-model",
                base_url="https://compatible.example.test/v1",
                api_key_env="COMPAT_TEST_API_KEY",
            )
        )
        assert isinstance(model, OpenAIChatModel)
        self.addAsyncCleanup(model.client.close)
        spans = [{"version": "default", "start": 0, "end": 2}]
        for tool_call in (False, True):
            with self.subTest(tool_call=tool_call):
                # The OpenAI SDK constructs responses without validating this metadata.
                raw = completion(tool_call=tool_call).model_copy(
                    update={"metadata": {"weight_version": "default", "weight_versions": spans}}
                )
                with patch.object(
                    model.client.chat.completions, "create", AsyncMock(return_value=raw)
                ):
                    response = await model.request(
                        [ModelRequest(parts=[UserPromptPart("Respond.")])],
                        None,
                        ModelRequestParameters(),
                    )
                self.assertEqual(
                    response.parts[0],
                    ToolCallPart("stay", "{}", "stay-1")
                    if tool_call
                    else TextPart("Summary or response."),
                )
                self.assertEqual(response.usage.total_tokens, 12)
                self.assertEqual(
                    raw.metadata, {"weight_version": "default", "weight_versions": spans}
                )

    async def test_compatible_endpoint_still_rejects_other_invalid_metadata(self) -> None:
        model = create_model(
            OpenAICompatibleModel(
                runtime="openai-compatible",
                model="test-model",
                base_url="https://compatible.example.test/v1",
                api_key_env="COMPAT_TEST_API_KEY",
            )
        )
        assert isinstance(model, OpenAIChatModel)
        self.addAsyncCleanup(model.client.close)
        raw = completion().model_copy(update={"metadata": {"unrelated": [1]}})
        with (
            patch.object(model.client.chat.completions, "create", AsyncMock(return_value=raw)),
            self.assertRaisesRegex(UnexpectedModelBehavior, "metadata.unrelated"),
        ):
            await model.request(
                [ModelRequest(parts=[UserPromptPart("Respond.")])],
                None,
                ModelRequestParameters(),
            )

    async def test_summary_inherits_provider_defaults_without_agent_seed(self) -> None:
        settings = AgentSettings(
            context_window_tokens=200,
            compaction_trigger_fraction=0.5,
            compaction_tail_tokens=20,
            summary_completion_tokens=16,
            memory_injection_tokens=100,
        )
        model = create_model(
            OpenRouterSelection(
                runtime="openrouter",
                model="vendor/test-model",
                provider="selected-provider",
                settings=settings,
            )
        )
        assert isinstance(model, OpenRouterModel)
        self.addAsyncCleanup(model.client.close)
        sessions = AgentSessions(
            Agent(model, output_type=str),
            settings=settings,
        )
        request = AsyncMock(return_value=completion())
        with patch.object(model.client.chat.completions, "create", request):
            await sessions.run("agent-7", "A" * 600, deps=None, model_settings={"seed": 42})
            await sessions.run("agent-7", "B" * 600, deps=None, model_settings={"seed": 43})

        calls = [cast(dict[str, object], call.kwargs) for call in request.call_args_list]
        self.assertEqual([call["max_tokens"] for call in calls], [32_768, 16, 32_768])
        self.assertIs(calls[1]["tool_choice"], omit)
        self.assertEqual([call["seed"] for call in calls], [42, omit, 43])
        self.assertEqual(calls[0]["extra_body"], calls[1]["extra_body"])
        self.assertEqual(calls[1]["extra_body"], calls[2]["extra_body"])
        self.assertTrue(all(call["timeout"] == 3_600.0 for call in calls))


if __name__ == "__main__":
    unittest.main()
