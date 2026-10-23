import json
import os
import tempfile
import unittest
from contextlib import chdir
from pathlib import Path
from typing import Literal, cast
from unittest.mock import AsyncMock, patch

import httpx2 as httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI, AsyncStream, omit
from openai.types.chat import ChatCompletion, ChatCompletionChunk
from pydantic import StrictInt
from pydantic_ai import Agent, ToolOutput
from pydantic_ai.exceptions import ModelHTTPError
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
from pydantic_ai.providers.anthropic import AnthropicProvider
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import ToolDefinition

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import AgentSettings, ModelSelection
from mam_bench.runtime import CaseRuntime, create_model
from mam_bench.sessions import AgentSessions, Completed
from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim
from mam_bench.simulations.schelling.simulation import SchellingSim
from support import SandboxTestCase, civil, schelling


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


def streamed_request(client: AsyncOpenAI, *, tool_name: str, tool_args: str) -> AsyncMock:
    async def create(**kwargs: object) -> AsyncStream[ChatCompletionChunk]:
        assert kwargs["stream"] is True
        answer = completion(tool_call=True, tool_name=tool_name, tool_args=tool_args)
        chunk = answer.model_dump(mode="json")
        chunk["object"] = "chat.completion.chunk"
        choice = chunk["choices"][0]
        choice["delta"] = choice.pop("message")
        for index, call in enumerate(choice["delta"]["tool_calls"]):
            call["index"] = index
        response = httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
        )
        return AsyncStream(cast_to=ChatCompletionChunk, response=response, client=client)

    return AsyncMock(side_effect=create)


@patch.dict(
    os.environ,
    {
        "OPENROUTER_API_KEY": "unused-offline-test-value",
        "OPENROUTER_BASE_URL": "https://router.example.test/v1",
        "VLLM_API_KEY": "unused-offline-test-value",
    },
)
class ModelProviderTests(SandboxTestCase):
    async def test_vllm_sessions_allow_a_missing_or_empty_api_key(self) -> None:
        async def stay() -> str:
            return "Stayed."

        for key in (None, ""):
            with (
                self.subTest(key=key),
                tempfile.TemporaryDirectory(dir=".scratch") as directory,
                patch.dict(os.environ),
            ):
                os.environ.pop("VLLM_API_KEY", None)
                if key is not None:
                    os.environ["VLLM_API_KEY"] = key
                with chdir(directory):
                    model = create_model(
                        ModelSelection(
                            runtime="vllm",
                            model="vendor/test-model",
                            base_url="https://compatible.example.test/v1",
                        )
                    )
                assert isinstance(model, OpenAIChatModel)
                self.addAsyncCleanup(model.client.close)
                agent: Agent[None, str] = Agent(model, output_type=ToolOutput(stay, name="stay"))
                async with AgentSessions(agent, context_window_tokens=262144) as sessions:
                    request = AsyncMock(return_value=completion(tool_call=True))
                    with patch.object(model.client.chat.completions, "create", request):
                        result = await sessions.run("probe", "Stay.", deps=None)
                self.assertEqual(result, Completed("Stayed."))

    async def test_openai_sessions_send_equivalent_fixed_integer_arrays(self) -> None:
        async def move(destination: tuple[StrictInt, StrictInt] | None) -> str:
            return str(destination)

        for model_name in ("openai/gpt-6-luna", "vendor/test-model"):
            with self.subTest(model=model_name):
                settings = AgentSettings(timeout_seconds=3)
                model = create_model(
                    ModelSelection(
                        runtime="openrouter", model=model_name, provider="openai", settings=settings
                    )
                )
                assert isinstance(model, OpenRouterModel)
                self.addAsyncCleanup(model.client.close)
                agent: Agent[None, str] = Agent(model, output_type=ToolOutput(move, name="move"))
                sessions = AgentSessions(agent, context_window_tokens=262144, settings=settings)
                request = AsyncMock(
                    return_value=completion(
                        tool_call=True, tool_name="move", tool_args='{"destination": [2, 3]}'
                    )
                )
                with patch.object(model.client.chat.completions, "create", request):
                    result = await sessions.run("probe", "Move.", deps=None)
                self.assertEqual(result, Completed("(2, 3)"))
                schema = next(
                    tool["function"]["parameters"]
                    for tool in request.call_args.kwargs["tools"]
                    if tool["function"]["name"] == "move"
                )
                alternatives = schema["properties"]["destination"]["anyOf"]
                self.assertIn({"type": "null"}, alternatives)
                array = next(branch for branch in alternatives if branch.get("type") == "array")
                self.assertEqual(
                    array,
                    {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        **(
                            {"items": {"type": "integer"}}
                            if model_name.startswith("openai/")
                            else {"prefixItems": [{"type": "integer"}, {"type": "integer"}]}
                        ),
                    },
                )

    async def test_router_discovery_and_terminal_requests_keep_defaults_and_routing(self) -> None:
        from mam_bench.model_setup import resolve_model_setup

        for parameter in (None, "max_tokens", "max_completion_tokens"):
            with self.subTest(completion_parameter=parameter):
                requests: list[httpx.Request] = []

                def respond(
                    request: httpx.Request,
                    *,
                    parameter: str | None = parameter,
                    requests: list[httpx.Request] = requests,
                ) -> httpx.Response:
                    requests.append(request)
                    self.assertEqual(request.url.host, "router.example.test")
                    if request.method == "GET" and request.url.path == "/v1/models":
                        return httpx.Response(200, json={"data": []})
                    if request.method == "GET":
                        self.assertEqual(request.url.path, "/v1/models/openai/gpt-4.1/endpoints")
                        return httpx.Response(
                            200,
                            json={
                                "data": {
                                    "endpoints": [
                                        {
                                            "tag": "azure",
                                            "provider_name": "Azure",
                                            "status": 0,
                                            "context_length": 1047576,
                                            "max_completion_tokens": 32768,
                                            "supported_parameters": [
                                                "tools",
                                                "tool_choice",
                                                parameter,
                                            ],
                                            "supports_tool_choice": {
                                                "required": True,
                                                "auto": True,
                                            },
                                        }
                                    ]
                                }
                            },
                        )
                    self.assertEqual(
                        (request.method, request.url.path), ("POST", "/v1/chat/completions")
                    )
                    self.assertEqual(request.headers["X-Title"], "MAM-Bench")
                    return httpx.Response(
                        200, json=completion(tool_call=True).model_dump(mode="json")
                    )

                def client(**kwargs: object) -> AsyncOpenAI:
                    return AsyncOpenAI(
                        base_url=cast(str, kwargs["base_url"]),
                        api_key=cast(str, kwargs["api_key"]),
                        default_headers=cast(dict[str, str], kwargs.get("default_headers", {})),
                        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
                    )

                with (
                    patch("mam_bench.model_setup.AsyncOpenAI", side_effect=client),
                    patch("mam_bench.runtime.AsyncOpenAI", side_effect=client),
                ):
                    selected = ModelSelection(
                        runtime="openrouter", model="openai/gpt-4.1", provider="azure"
                    )
                    if parameter is not None:
                        selected = (await resolve_model_setup(selected)).selection
                    model = create_model(selected)
                    assert isinstance(model, OpenRouterModel)
                    self.addAsyncCleanup(model.client.close)
                    request_settings: ModelSettings = {}
                    if parameter:
                        cap = selected.settings.max_completion_tokens
                        assert cap is not None
                        request_settings["max_tokens"] = cap
                    response = await model.request(
                        [ModelRequest(parts=[UserPromptPart("Choose an action.")])],
                        request_settings or None,
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
                self.assertEqual(len(requests), 3 if parameter else 1)
                sent = json.loads(requests[-1].content)
                self.assertEqual(
                    sent["provider"],
                    {"only": ["azure"], "allow_fallbacks": False, "require_parameters": True},
                )
                self.assertEqual(sent["tool_choice"], "required")
                self.assertIs(sent["tools"][0]["function"]["strict"], True)
                for field in ("temperature", "top_p", "seed"):
                    self.assertNotIn(field, sent)
                if parameter:
                    self.assertEqual(sent[parameter], 32768)
                for field in {"max_tokens", "max_completion_tokens"} - {parameter}:
                    self.assertNotIn(field, sent)
                self.assertIsInstance(response.parts[0], ToolCallPart)
                self.assertEqual(response.usage.total_tokens, 12)

    async def test_explicit_cache_adds_native_breakpoints_to_claude_requests(self) -> None:
        selected = ModelSelection(
            runtime="openrouter",
            model="anthropic/claude-sonnet-4",
            provider="anthropic",
            settings=AgentSettings(tool_choice="auto"),
        )
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

    async def test_vllm_uses_native_profile_and_request_format(self) -> None:
        model = create_model(
            ModelSelection(
                runtime="vllm",
                model="google/gemma-4-27b-it",
                base_url="https://compatible.example.test/v1",
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
                {"max_tokens": 2048},
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

    async def test_vllm_sessions_send_explicit_effort_for_an_unrecognized_model(self) -> None:
        for effort in ("none", "medium"):
            with self.subTest(effort=effort):
                settings = AgentSettings(reasoning_effort=effort)
                model = create_model(
                    ModelSelection(
                        runtime="vllm",
                        model="nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB",
                        base_url="https://compatible.example.test/v1",
                        settings=settings,
                    )
                )
                assert isinstance(model, OpenAIChatModel)
                self.addAsyncCleanup(model.client.close)
                agent: Agent[None, str] = Agent(model)
                sessions = AgentSessions(agent, context_window_tokens=262144, settings=settings)
                request = AsyncMock(return_value=completion())
                with patch.object(model.client.chat.completions, "create", request):
                    result = await sessions.run("probe", "Respond.", deps=None)
                self.assertEqual(result, Completed("Summary or response."))
                self.assertEqual(request.call_args.kwargs["reasoning_effort"], effort)

    async def test_qwen38_sessions_send_thinking_through_the_chat_template(self) -> None:
        for effort, expected in (
            ("none", {"enable_thinking": False}),
            ("medium", {"enable_thinking": True, "reasoning_effort": "medium"}),
        ):
            with self.subTest(effort=effort):
                settings = AgentSettings(
                    reasoning_effort=cast(Literal["none", "medium"], effort), top_k=10
                )
                model = create_model(
                    ModelSelection(
                        runtime="vllm",
                        model="nvidia/Qwen3.8-27B-NVFP4",
                        base_url="https://compatible.example.test/v1",
                        settings=settings,
                    )
                )
                assert isinstance(model, OpenAIChatModel)
                self.addAsyncCleanup(model.client.close)
                agent: Agent[None, str] = Agent(model)
                sessions = AgentSessions(agent, context_window_tokens=262144, settings=settings)
                request = AsyncMock(return_value=completion())
                with patch.object(model.client.chat.completions, "create", request):
                    result = await sessions.run("probe", "Respond.", deps=None)
                self.assertEqual(result, Completed("Summary or response."))
                self.assertEqual(
                    request.call_args.kwargs["extra_body"],
                    {"top_k": 10, "chat_template_kwargs": expected},
                )
                self.assertIs(request.call_args.kwargs["reasoning_effort"], omit)

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
                        ModelSelection(
                            runtime="vllm",
                            model="google/gemma-4-27b-it",
                            base_url="https://compatible.example.test/v1",
                            settings=settings,
                        )
                    )
                    assert isinstance(model, OpenAIChatModel)
                    self.addAsyncCleanup(model.client.close)
                    writer = ArtifactWriter(Path(directory) / f"{tool_choice}-{strict}")
                    sim = SchellingSim(
                        schelling(),
                        runtime=CaseRuntime(
                            context_window_tokens=260_000,
                            model=model,
                            settings=settings,
                            writer=writer,
                        ),
                    )
                    request = streamed_request(
                        model.client, tool_name="move", tool_args='{"destination": null}'
                    )
                    with patch.object(model.client.chat.completions, "create", request):
                        await sim.step()

                    self.assertEqual(request.await_count, len(sim.controlled_agent_ids))
                    self.assertFalse((writer.directory / "turns.jsonl").exists())
                    for call in request.await_args_list:
                        sent = cast(dict[str, object], call.kwargs)
                        self.assertIs(sent["seed"], omit)
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
                        self.assertIs(by_name["read_messages"].get("strict"), True)
                        for name in ("move", "post_message"):
                            self.assertIs(by_name[name].get("strict"), True)

    async def test_vllm_repetition_stop_retains_cause_without_provider_failure(self) -> None:
        model = create_model(
            ModelSelection.model_validate(
                {
                    "runtime": "vllm",
                    "model": "test-model",
                    "base_url": "https://compatible.example.test/v1",
                    "repetition_detection": {},
                }
            )
        )
        assert isinstance(model, OpenAIChatModel)
        self.addAsyncCleanup(model.client.close)
        answer = completion()
        # The SDK constructs responses without validating vendor finish reasons.
        answer.choices[0] = answer.choices[0].model_copy(
            update={"finish_reason": "repetition", "stop_reason": "repetition_detected"}
        )
        request = AsyncMock(return_value=answer)
        with patch.object(model.client.chat.completions, "create", request):
            response = await model.request(
                [ModelRequest(parts=[UserPromptPart("Choose an action.")])],
                None,
                ModelRequestParameters(),
            )
        sent = cast(dict[str, object], request.call_args.kwargs)
        self.assertEqual(
            sent["extra_body"],
            {
                "repetition_detection": {
                    "min_pattern_size": 16,
                    "max_pattern_size": 128,
                    "min_count": 8,
                },
            },
        )
        self.assertEqual(response.finish_reason, "length")
        assert response.provider_details is not None
        self.assertEqual(response.provider_details["finish_reason"], "repetition")
        self.assertEqual(response.provider_details["stop_reason"], "repetition_detected")

    async def test_native_profiles_keep_tool_forcing_exceptions(self) -> None:
        for selected in (
            ModelSelection(
                runtime="vllm",
                model="openai/gpt-oss-120b",
                base_url="https://vllm.example.test/v1",
            ),
            ModelSelection(
                runtime="openrouter", model="qwen/qwen-3-coder", provider="selected-provider"
            ),
            ModelSelection(
                runtime="openrouter",
                model="anthropic/claude-sonnet-4",
                provider="anthropic",
                settings=AgentSettings(reasoning_effort="medium"),
            ),
        ):
            with self.subTest(model=selected.model):
                model = create_model(selected)
                assert isinstance(model, OpenAIChatModel)
                self.addAsyncCleanup(model.client.close)
                request = AsyncMock(return_value=completion(tool_call=True))
                with patch.object(model.client.chat.completions, "create", request):
                    await model.request(
                        [ModelRequest(parts=[UserPromptPart("Choose an action.")])],
                        None,
                        ModelRequestParameters(
                            output_mode="tool",
                            output_tools=[
                                ToolDefinition(
                                    name="stay",
                                    parameters_json_schema={"type": "object", "properties": {}},
                                )
                            ],
                            allow_text_output=False,
                        ),
                    )
                self.assertEqual(request.call_args.kwargs["tool_choice"], "auto")

    async def test_unified_thinking_drives_native_openrouter_sampling_rules(self) -> None:
        for name, effort, sampling in (
            ("openai/gpt-5.4", "medium", False),
            ("openai/gpt-5.5", "none", True),
        ):
            with self.subTest(model=name):
                model = create_model(
                    ModelSelection(
                        runtime="openrouter",
                        model=name,
                        provider="selected-provider",
                        settings=AgentSettings(
                            reasoning_effort=cast(Literal["medium", "none"], effort),
                            temperature=1.0,
                            top_p=0.95,
                        ),
                    )
                )
                assert isinstance(model, OpenRouterModel)
                self.addAsyncCleanup(model.client.close)
                request = AsyncMock(return_value=completion())
                with patch.object(model.client.chat.completions, "create", request):
                    await model.request(
                        [ModelRequest(parts=[UserPromptPart("Respond.")])],
                        None,
                        ModelRequestParameters(),
                    )
                sent = request.call_args.kwargs
                self.assertEqual(sent["temperature"], 1.0 if sampling else omit)
                self.assertEqual(sent["top_p"], 0.95 if sampling else omit)

    async def test_native_runtimes_send_reasoning_effort_verbatim(self) -> None:
        # The provider, not MAM-Bench, decides whether an effort is valid.
        for runtime, model_name in (("openai", "gpt-4.1-mini"), ("anthropic", "claude-opus-4-6")):
            with self.subTest(runtime=runtime):
                bodies: list[dict[str, object]] = []

                def reject(
                    request: httpx.Request, bodies: list[dict[str, object]] = bodies
                ) -> httpx.Response:
                    bodies.append(json.loads(request.content))
                    return httpx.Response(400, json={"error": {"message": "effort rejected"}})

                http_client = httpx.AsyncClient(transport=httpx.MockTransport(reject))
                provider = (
                    patch(
                        "mam_bench.runtime.OpenAIProvider",
                        return_value=OpenAIProvider(
                            openai_client=AsyncOpenAI(api_key="test", http_client=http_client)
                        ),
                    )
                    if runtime == "openai"
                    else patch(
                        "mam_bench.runtime.AnthropicProvider",
                        return_value=AnthropicProvider(
                            anthropic_client=AsyncAnthropic(api_key="test", http_client=http_client)
                        ),
                    )
                )
                with provider:
                    model = create_model(
                        ModelSelection.model_validate(
                            {
                                "runtime": runtime,
                                "model": model_name,
                                "settings": {"reasoning_effort": "ultra"},
                            }
                        )
                    )
                with self.assertRaises(ModelHTTPError):
                    await model.request(
                        [ModelRequest(parts=[UserPromptPart("Respond.")])],
                        None,
                        ModelRequestParameters(),
                    )
                await http_client.aclose()
                sent = cast(dict[str, dict[str, object]], bodies[-1])
                key = "reasoning" if runtime == "openai" else "output_config"
                self.assertEqual(sent[key]["effort"], "ultra")

    async def test_wire_requests_are_logged_once_per_distinct_shape(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=completion().model_dump(mode="json"))

        def client(**kwargs: object) -> AsyncOpenAI:
            hooks = cast(httpx.AsyncClient, kwargs["http_client"]).event_hooks
            return AsyncOpenAI(
                base_url=cast(str, kwargs["base_url"]),
                api_key="test",
                http_client=httpx.AsyncClient(
                    transport=httpx.MockTransport(respond), event_hooks=hooks
                ),
            )

        with patch("mam_bench.runtime.AsyncOpenAI", side_effect=client):
            model = create_model(
                ModelSelection(
                    runtime="openrouter",
                    model="deepseek/deepseek-v4.1-flash",
                    provider="deepinfra/fp8",
                    settings=AgentSettings(reasoning_effort="max"),
                )
            )
        assert isinstance(model, OpenRouterModel)
        self.addAsyncCleanup(model.client.close)
        with self.assertLogs("mam_bench.runtime", "INFO") as logs:
            for prompt, cap in (("One.", 100), ("Two.", 100), ("Three.", 200)):
                await model.request(
                    [ModelRequest(parts=[UserPromptPart(prompt)])],
                    ModelSettings(max_tokens=cap),
                    ModelRequestParameters(),
                )
        wire = [line for line in logs.output if "request.wire" in line]
        self.assertEqual(len(wire), 2)
        self.assertIn('"effort": "max"', wire[0])
        self.assertNotIn("One.", wire[0])

    async def test_openrouter_sends_configured_reasoning_effort_verbatim(self) -> None:
        for effort in ("xhigh", "max", "ultra"):
            with self.subTest(effort=effort):
                model = create_model(
                    ModelSelection(
                        runtime="openrouter",
                        model="deepseek/deepseek-v4.1-flash",
                        provider="deepinfra/fp8",
                        settings=AgentSettings(reasoning_effort=effort),
                    )
                )
                assert isinstance(model, OpenRouterModel)
                self.addAsyncCleanup(model.client.close)
                request = AsyncMock(return_value=completion())
                with patch.object(model.client.chat.completions, "create", request):
                    await model.request(
                        [ModelRequest(parts=[UserPromptPart("Respond.")])],
                        None,
                        ModelRequestParameters(),
                    )
                body = request.call_args.kwargs["extra_body"]
                self.assertEqual(body["reasoning"]["effort"], effort)
                self.assertEqual(body["provider"]["only"], ["deepinfra/fp8"])

    async def test_civil_citizen_defer_uses_per_model_strict_schema(self) -> None:
        for strict in (False, True):
            with self.subTest(strict=strict), tempfile.TemporaryDirectory() as directory:
                settings = AgentSettings(strict_parameterless_tools=strict)
                model = create_model(
                    ModelSelection(
                        runtime="vllm",
                        model="google/gemma-4-27b-it",
                        base_url="https://compatible.example.test/v1",
                        settings=settings,
                    )
                )
                assert isinstance(model, OpenAIChatModel)
                self.addAsyncCleanup(model.client.close)
                writer = ArtifactWriter(Path(directory) / "case")
                sim = CivilViolenceSim(
                    civil(),
                    runtime=CaseRuntime(
                        context_window_tokens=260_000, model=model, settings=settings, writer=writer
                    ),
                )
                for identity in sim.controlled_agent_ids:
                    sim.citizens[identity].location = None
                    sim.citizens[identity].jail_remaining = 1
                request = streamed_request(model.client, tool_name="defer", tool_args="{}")
                with patch.object(model.client.chat.completions, "create", request):
                    await sim.step()

                self.assertEqual(request.await_count, len(sim.controlled_agent_ids))
                for call in request.await_args_list:
                    sent = cast(dict[str, object], call.kwargs)
                    self.assertIs(sent["seed"], omit)
                    tools = cast(list[dict[str, object]], sent["tools"])
                    functions = [cast(dict[str, object], tool["function"]) for tool in tools]
                    by_name = {cast(str, function["name"]): function for function in functions}
                    self.assertEqual(sent["tool_choice"], "required")
                    for name in ("defer", "read_messages"):
                        self.assertEqual(
                            cast(dict[str, object], by_name[name]["parameters"])["properties"],
                            {},
                        )
                        self.assertIs(by_name[name].get("strict"), True)
                    self.assertIs(by_name["act"].get("strict"), True)
                    self.assertIs(by_name["post_message"].get("strict"), True)
                self.assertFalse((writer.directory / "turns.jsonl").exists())

    async def test_summary_inherits_provider_defaults_without_agent_seed(self) -> None:
        settings = AgentSettings(
            max_completion_tokens=1000,
            compaction_trigger_fraction=0.5,
            compaction_tail_tokens=20,
        )
        model = create_model(
            ModelSelection(
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
            context_window_tokens=200,
            settings=settings,
        )
        request = AsyncMock(return_value=completion())
        with patch.object(model.client.chat.completions, "create", request):
            await sessions.run("agent-7", "A" * 600, deps=None)
            await sessions.run("agent-7", "B" * 600, deps=None)

        calls = [cast(dict[str, object], call.kwargs) for call in request.call_args_list]
        self.assertEqual([call["max_tokens"] for call in calls], [1000, omit, 1000])
        self.assertIs(calls[1]["tool_choice"], omit)
        self.assertTrue(all(call["seed"] is omit for call in calls))
        self.assertEqual(calls[0]["extra_body"], calls[1]["extra_body"])
        self.assertEqual(calls[1]["extra_body"], calls[2]["extra_body"])
        self.assertTrue(all(call["timeout"] == 3_600.0 for call in calls))


if __name__ == "__main__":
    unittest.main()
