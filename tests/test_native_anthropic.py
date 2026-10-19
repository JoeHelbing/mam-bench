"""Native provider setup, streaming tools, and cache accounting without network calls."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx2 as httpx
from anthropic import AsyncAnthropic
from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput
from pydantic_ai.models.anthropic import AnthropicModel as NativeAnthropicModel

from mam_bench.config import AgentSettings, ModelSelection, load_benchmark_config
from mam_bench.model_setup import resolve_model_setup
from mam_bench.runtime import create_model
from mam_bench.sessions import AgentSessions, Completed


class Answer(BaseModel):
    value: int


def event_stream(events: list[dict[str, object]]) -> str:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


def streamed_answer() -> str:

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
                    "name": "record_probe",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "input_json_delta", "partial_json": '{"value": 7}'},
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


@patch.dict(os.environ, {"ANTHROPIC_API_KEY": "offline-test-key"})
class NativeAnthropicTests(unittest.IsolatedAsyncioTestCase):
    async def test_yaml_setup_uses_native_context_and_preserves_defaults(self) -> None:
        config = load_benchmark_config(
            Path("models/anthropic.yaml"), output_directory=Path("results")
        )
        setup = await resolve_model_setup(config.model)
        self.assertEqual(setup.context_window_tokens, 200000)
        self.assertEqual(setup.selection.settings, config.model.settings)
        self.assertEqual(setup.metadata["source"], "pydantic_ai_model_profile")
        defaults = await resolve_model_setup(
            ModelSelection(runtime="anthropic", model=config.model.model)
        )
        self.assertIsNone(defaults.selection.settings.max_completion_tokens)
        self.assertIsNone(defaults.selection.settings.summary_completion_tokens)
        with self.assertRaisesRegex(ValueError, "context metadata"):
            await resolve_model_setup(ModelSelection(runtime="anthropic", model="unknown-model"))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            await resolve_model_setup(
                ModelSelection(
                    runtime="anthropic",
                    model=config.model.model,
                    settings=AgentSettings(max_completion_tokens=200001),
                )
            )

    async def test_streamed_tool_output_keeps_provider_cache_usage(self) -> None:
        payloads: list[dict[str, object]] = []

        def respond(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/messages")
            payloads.append(json.loads(request.content))
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=streamed_answer()
            )

        async with AsyncAnthropic(
            api_key="offline-test-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            with patch("pydantic_ai.providers.anthropic.AsyncAnthropic", return_value=client):
                selection = ModelSelection(
                    runtime="anthropic",
                    model="claude-haiku-4-5",
                    settings=AgentSettings(max_completion_tokens=1024),
                )
                model = create_model(selection)
            self.assertIsInstance(model, NativeAnthropicModel)
            with tempfile.TemporaryDirectory() as directory:
                sessions = AgentSessions(
                    Agent(
                        model,
                        instructions="Return the requested value.",
                        output_type=ToolOutput(Answer, name="record_probe"),
                    ),
                    settings=selection.settings,
                    context_window_tokens=200000,
                    artifact_directory=Path(directory),
                )
                result = await sessions.run("agent", "Return 7.", deps=None)
                self.assertEqual(result, Completed(Answer(value=7)))
                self.assertEqual(sessions.usage.requests, 1)
                self.assertEqual(sessions.usage.cache_read_tokens, 1024)
                self.assertTrue(sessions.history("agent"))
            payload = payloads[0]
            self.assertIs(payload["stream"], True)
            self.assertNotIn("temperature", payload)
            self.assertNotIn("top_p", payload)

            self.assertEqual(payload["cache_control"], {"type": "ephemeral", "ttl": "5m"})
            self.assertEqual(payload["tool_choice"], {"type": "any"})
            self.assertEqual(payload["max_tokens"], 1024)
            tools = cast(list[dict[str, object]], payload["tools"])
            self.assertEqual(tools[-1]["cache_control"], {"type": "ephemeral", "ttl": "5m"})
            system = cast(list[dict[str, object]], payload["system"])
            self.assertTrue(any("cache_control" in block for block in system))
            self.assertEqual(sessions.usage.cache_write_tokens, 2048)


if __name__ == "__main__":
    unittest.main()
