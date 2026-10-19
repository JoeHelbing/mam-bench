"""Native provider setup, streaming tools, and cache accounting without network calls."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx2 as httpx
from openai import AsyncOpenAI
from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from mam_bench.config import AgentSettings, OpenAIModel, load_benchmark_config
from mam_bench.model_setup import resolve_model_setup
from mam_bench.runtime import create_model
from mam_bench.sessions import AgentSessions, Completed


class Answer(BaseModel):
    value: int


def event_stream(events: list[dict[str, object]]) -> str:
    return "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events)


def streamed_answer() -> str:

    item = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": "call_1",
        "name": "record_probe",
        "arguments": '{"value": 7}',
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
                "delta": '{"value": 7}',
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


@patch.dict(os.environ, {"OPENAI_API_KEY": "offline-test-key"})
class NativeOpenAITests(unittest.IsolatedAsyncioTestCase):
    async def test_yaml_setup_uses_native_context_and_preserves_defaults(self) -> None:
        config = load_benchmark_config(Path("models/openai.yaml"), output_directory=Path("results"))
        self.assertIsInstance(config.model, OpenAIModel)
        setup = await resolve_model_setup(config.model)
        self.assertEqual(setup.context_window_tokens, 1000000)
        self.assertEqual(setup.selection.settings, config.model.settings)
        self.assertEqual(setup.metadata["source"], "pydantic_ai_model_profile")
        defaults = await resolve_model_setup(
            OpenAIModel(runtime="openai", model=config.model.model)
        )
        self.assertIsNone(defaults.selection.settings.max_completion_tokens)
        self.assertIsNone(defaults.selection.settings.summary_completion_tokens)
        with self.assertRaisesRegex(ValueError, "context metadata"):
            await resolve_model_setup(OpenAIModel(runtime="openai", model="unknown-model"))
        with self.assertRaisesRegex(ValueError, "exceeds"):
            await resolve_model_setup(
                OpenAIModel(
                    runtime="openai",
                    model=config.model.model,
                    settings=AgentSettings(max_completion_tokens=1000001),
                )
            )

    async def test_streamed_tool_output_keeps_provider_cache_usage(self) -> None:
        payloads: list[dict[str, object]] = []

        def respond(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/v1/responses")
            payloads.append(json.loads(request.content))
            return httpx.Response(
                200, headers={"content-type": "text/event-stream"}, content=streamed_answer()
            )

        async with AsyncOpenAI(
            api_key="offline-test-key",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        ) as client:
            with patch(
                "mam_bench.runtime.OpenAIProvider",
                return_value=OpenAIProvider(openai_client=client),
            ):
                selection = OpenAIModel(
                    runtime="openai",
                    model="gpt-4.1-mini",
                    settings=AgentSettings(max_completion_tokens=1024),
                )
                model = create_model(selection)
            self.assertIsInstance(model, OpenAIResponsesModel)
            with tempfile.TemporaryDirectory() as directory:
                sessions = AgentSessions(
                    Agent(
                        model,
                        instructions="Return the requested value.",
                        output_type=ToolOutput(Answer, name="record_probe"),
                    ),
                    settings=selection.settings,
                    context_window_tokens=1000000,
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

            self.assertEqual(payload["prompt_cache_key"], "mam-bench:gpt-4.1-mini")
            self.assertIs(payload["store"], False)
            self.assertEqual(payload["tool_choice"], "required")
            self.assertEqual(payload["max_output_tokens"], 1024)
            self.assertEqual(sessions.usage.cache_write_tokens, 0)


if __name__ == "__main__":
    unittest.main()
