"""Native provider configurations execute real simulation actions over offline HTTP."""

import json
import os
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx2 as httpx
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import AgentSettings, ModelSelection, load_benchmark_config
from mam_bench.model_setup import resolve_model_setup
from mam_bench.runtime import CaseRuntime, create_model
from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim
from mam_bench.simulations.schelling.simulation import SchellingSim
from support import SandboxTestCase, civil, schelling


def locations(state: dict[str, object]) -> dict[int, tuple[int, int] | None]:
    if "agent_locations" in state:
        return {
            i: divmod(place, 8) for i, place in enumerate(cast(list[int], state["agent_locations"]))
        }
    people = cast(list[dict[str, object]], state["citizens"]) + cast(
        list[dict[str, object]], state["police"]
    )
    result: dict[int, tuple[int, int] | None] = {}
    for person in people:
        coordinate = cast(list[int] | None, person["location"])
        result[cast(int, person["agent_id"])] = (
            (coordinate[0], coordinate[1]) if coordinate is not None else None
        )
    return result


def event_stream(events: list[dict[str, object]]) -> str:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


def openai_stream(name: str, arguments: dict[str, object]) -> str:

    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": name,
        "arguments": json.dumps(arguments),
        "status": "completed",
    }
    response = {
        "id": "resp_1",
        "created_at": 1,
        "model": "gpt-4.1-mini",
        "object": "response",
        "parallel_tool_calls": True,
        "tool_choice": "required",
        "tools": [],
        "status": "completed",
        "output": [item],
        "usage": {
            "input_tokens": 2048,
            "output_tokens": 15,
            "total_tokens": 2063,
            "input_tokens_details": {"cached_tokens": 1024},
            "output_tokens_details": {"reasoning_tokens": 0},
        },
    }
    return event_stream(
        [
            {
                "type": "response.created",
                "sequence_number": 0,
                "response": {**response, "output": [], "status": "in_progress", "usage": None},
            },
            {
                "type": "response.output_item.added",
                "sequence_number": 1,
                "output_index": 0,
                "item": {**item, "arguments": "", "status": "in_progress"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "sequence_number": 2,
                "output_index": 0,
                "item_id": "fc_1",
                "delta": json.dumps(arguments),
            },
            {
                "type": "response.output_item.done",
                "sequence_number": 3,
                "output_index": 0,
                "item": item,
            },
            {"type": "response.completed", "sequence_number": 4, "response": response},
        ]
    )


def anthropic_stream(name: str, arguments: dict[str, object]) -> str:

    return event_stream(
        [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-haiku-4-5",
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": 512,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 1024,
                        "cache_creation_input_tokens": 2048,
                    },
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": name,
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": json.dumps(arguments)},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 15},
            },
            {"type": "message_stop"},
        ]
    )


@patch.dict(
    os.environ, {"OPENAI_API_KEY": "offline-test-key", "ANTHROPIC_API_KEY": "offline-test-key"}
)
class NativeProviderTests(SandboxTestCase):
    async def test_native_yaml_drives_validated_actions_and_retains_cache_usage(self) -> None:
        for service, model_name, context in (
            ("openai", "gpt-4.1-mini", 1000000),
            ("anthropic", "claude-haiku-4-5", 200000),
        ):
            with self.subTest(provider=service), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model_path = root / "model.yaml"
                model_path.write_text(
                    f"runtime: {service}\nmodel: {model_name}\n", encoding="utf-8"
                )
                config = load_benchmark_config(model_path, output_directory=root)
                setup = await resolve_model_setup(config.model)
                self.assertEqual(setup.context_window_tokens, context)
                self.assertEqual(setup.selection.settings, config.model.settings)
                self.assertEqual(setup.metadata["source"], "pydantic_ai_model_profile")
                self.assertIsNone(setup.selection.settings.max_completion_tokens)
                self.assertIsNone(setup.selection.settings.summary_completion_tokens)
                for invalid_model, cap, error in (
                    ("unknown-model", None, "context metadata"),
                    (model_name, context + 1, "exceeds"),
                ):
                    with (
                        self.subTest(model=invalid_model),
                        self.assertRaisesRegex(ValueError, error),
                    ):
                        await resolve_model_setup(
                            ModelSelection.model_validate(
                                {
                                    "runtime": service,
                                    "model": invalid_model,
                                    "settings": {"max_completion_tokens": cap},
                                }
                            )
                        )

                for kind in ("schelling", "citizen", "police"):
                    with self.subTest(provider=service, simulation=kind):
                        payloads: list[dict[str, object]] = []
                        destination: list[int] = []
                        tool_name = "move" if kind == "schelling" else "act"

                        def respond(
                            request: httpx.Request,
                            *,
                            service: str = service,
                            kind: str = kind,
                            payloads: list[dict[str, object]] = payloads,
                            destination: list[int] = destination,
                            tool_name: str = tool_name,
                        ) -> httpx.Response:
                            self.assertEqual(
                                request.url.path,
                                "/v1/responses" if service == "openai" else "/v1/messages",
                            )
                            payloads.append(json.loads(request.content))
                            if not destination:
                                # Use the agent's view after the preceding phase settles.
                                messages = cast(
                                    list[dict[str, object]],
                                    payloads[-1]["input" if service == "openai" else "messages"],
                                )
                                for message in reversed(messages):
                                    content = message.get("content", [])
                                    blocks = (
                                        [{"text": content}]
                                        if isinstance(content, str)
                                        else cast(list[dict[str, object]], content)
                                    )
                                    texts = [block.get("text", "") for block in blocks]
                                    field = (
                                        "empty_locations"
                                        if kind == "schelling"
                                        else "legal_destinations"
                                    )
                                    for text in texts:
                                        if isinstance(text, str) and text.lstrip().startswith("{"):
                                            view = cast(dict[str, object], json.loads(text))
                                            if field in view:
                                                destination.extend(
                                                    cast(list[list[int]], view[field])[0]
                                                )
                                                break
                                    if destination:
                                        break
                                self.assertEqual(len(destination), 2)
                            requested: object = (
                                destination[:1]
                                if len(payloads) == 1
                                else destination
                                if len(payloads) == 2
                                else None
                            )
                            arguments: dict[str, object] = {"destination": requested}
                            if kind == "citizen":
                                arguments["active"] = False
                            elif kind == "police":
                                arguments["target_location"] = None
                            stream = openai_stream if service == "openai" else anthropic_stream
                            return httpx.Response(
                                200,
                                headers={"content-type": "text/event-stream"},
                                content=stream(tool_name, arguments),
                            )

                        http_client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
                        sdk = (
                            AsyncOpenAI(api_key="offline-test-key", http_client=http_client)
                            if service == "openai"
                            else AsyncAnthropic(api_key="offline-test-key", http_client=http_client)
                        )
                        async with sdk as client:
                            if isinstance(client, AsyncOpenAI):
                                provider = patch(
                                    "mam_bench.runtime.OpenAIProvider",
                                    return_value=OpenAIProvider(openai_client=client),
                                )
                            else:
                                provider = patch(
                                    "pydantic_ai.providers.anthropic.AsyncAnthropic",
                                    return_value=client,
                                )
                            with provider:
                                selected = config.model.model_copy(
                                    update={
                                        "settings": AgentSettings(
                                            max_completion_tokens=1024,
                                            timeout_seconds=5,
                                            concurrency=1,
                                        )
                                    }
                                )
                                model = create_model(selected)
                            resources = CaseRuntime(
                                model=model,
                                settings=selected.settings,
                                context_window_tokens=context,
                                writer=ArtifactWriter(root / kind),
                            )
                            sim = (
                                SchellingSim(schelling(max_steps=1), runtime=resources)
                                if kind == "schelling"
                                else CivilViolenceSim(
                                    civil(
                                        controlled_role="citizen"
                                        if kind == "citizen"
                                        else "police",
                                        citizen_density=0.5 if kind == "police" else 0.75,
                                        police_density=0.25 if kind == "police" else 0,
                                        max_steps=1,
                                    ),
                                    runtime=resources,
                                )
                            )
                            before = locations(sim.snapshot())
                            await sim.step()
                            after = locations(sim.snapshot())
                            self.assertEqual(
                                sum(before[i] != after[i] for i in sim.controlled_agent_ids), 1
                            )
                            self.assertIn(
                                tuple(destination), [after[i] for i in sim.controlled_agent_ids]
                            )
                            self.assertFalse((resources.writer.directory / "turns.jsonl").exists())
                            assert sim.sessions is not None
                            self.assertEqual(len(payloads), len(sim.controlled_agent_ids) + 1)
                            self.assertEqual(sim.sessions.usage.requests, len(payloads))
                            self.assertEqual(
                                sim.sessions.usage.cache_read_tokens, 1024 * len(payloads)
                            )
                            self.assertEqual(
                                sim.sessions.usage.cache_write_tokens,
                                (0 if service == "openai" else 2048) * len(payloads),
                            )
                            for payload in payloads:
                                self.assertIs(payload["stream"], True)
                                self.assertNotIn("temperature", payload)
                                self.assertNotIn("top_p", payload)
                                self.assertEqual(
                                    payload[
                                        "max_output_tokens" if service == "openai" else "max_tokens"
                                    ],
                                    1024,
                                )
                                tools = cast(list[dict[str, object]], payload["tools"])
                                action = next(tool for tool in tools if tool["name"] == tool_name)
                                schema = cast(
                                    dict[str, object],
                                    action["parameters" if service == "openai" else "input_schema"],
                                )
                                properties = cast(
                                    dict[str, dict[str, object]], schema["properties"]
                                )
                                alternatives = cast(
                                    list[dict[str, object]], properties["destination"]["anyOf"]
                                )
                                self.assertIn({"type": "null"}, alternatives)
                                array = next(
                                    branch
                                    for branch in alternatives
                                    if branch.get("type") == "array"
                                )
                                self.assertEqual((array["minItems"], array["maxItems"]), (2, 2))
                                if service == "openai":
                                    self.assertEqual(array["items"], {"type": "integer"})
                                    self.assertEqual(
                                        payload["prompt_cache_key"], "mam-bench:gpt-4.1-mini"
                                    )
                                    self.assertIs(payload["store"], False)
                                    self.assertEqual(payload["tool_choice"], "required")
                                else:
                                    self.assertEqual(
                                        array["prefixItems"],
                                        [{"type": "integer"}, {"type": "integer"}],
                                    )
                                    self.assertEqual(
                                        payload["cache_control"], {"type": "ephemeral", "ttl": "5m"}
                                    )
                                    self.assertEqual(payload["tool_choice"], {"type": "any"})
                                    self.assertEqual(
                                        tools[-1]["cache_control"],
                                        {"type": "ephemeral", "ttl": "5m"},
                                    )
                                    self.assertTrue(
                                        any(
                                            "cache_control" in block
                                            for block in cast(
                                                list[dict[str, object]], payload["system"]
                                            )
                                        )
                                    )
