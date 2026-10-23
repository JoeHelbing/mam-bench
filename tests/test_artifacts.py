"""Agent content is recorded by native instrumentation, including stopped calls."""

import json
import tempfile
from pathlib import Path

import httpx2 as httpx
from openai import AsyncOpenAI
from pydantic_ai import Agent, ToolOutput
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.messages import ModelResponse
from pydantic_ai.providers.vllm import VLLMProvider

from mam_bench.kludge import VLLMChatModel
from mam_bench.sessions import AgentSessions, Stopped, stream_responses
from mam_bench.telemetry import collect_telemetry
from support import SandboxTestCase


class ArtifactTests(SandboxTestCase):
    async def test_streamed_repetition_retains_partial_tool_and_stop_cause(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.assertIs(json.loads(request.content)["stream"], True)
            chunks: list[dict[str, object]] = [
                {
                    "id": "partial",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "scripted",
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "tool_calls": [
                                    {
                                        "index": 0,
                                        "id": "call-1",
                                        "type": "function",
                                        "function": {"name": "finish", "arguments": '{"total":'},
                                    }
                                ],
                            },
                            "finish_reason": None,
                            "token_ids": [101, 102],
                        }
                    ],
                },
                {
                    "id": "partial",
                    "object": "chat.completion.chunk",
                    "created": 1,
                    "model": "scripted",
                    "choices": [{"index": 0, "delta": {}, "finish_reason": "repetition"}],
                },
            ]
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text="".join("data: " + json.dumps(c) + "\n\n" for c in chunks)
                + "data: [DONE]\n\n",
            )

        async def finish(total: int) -> int:
            return total

        async with AsyncOpenAI(
            api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        ) as client:
            model = VLLMChatModel("scripted", provider=VLLMProvider(openai_client=client))
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                runtime = AgentSessions(
                    Agent(
                        model,
                        output_type=ToolOutput(finish),
                        capabilities=[ProcessEventStream(stream_responses)],
                    ),
                    context_window_tokens=260_000,
                )
                with collect_telemetry(root):
                    self.assertEqual(
                        await runtime.run("agent-a", "finish", deps=None),
                        Stopped("output_token_limit"),
                    )
                responses = [m for m in runtime.history("agent-a") if isinstance(m, ModelResponse)]
                self.assertEqual(
                    (responses[-1].provider_details or {})["finish_reason"], "repetition"
                )
                traces = (root / "telemetry/traces.jsonl").read_text()
                self.assertIn("call-1", traces)
                self.assertEqual(sorted(path.name for path in root.iterdir()), ["telemetry"])
