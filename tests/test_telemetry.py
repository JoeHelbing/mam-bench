import gzip
import json
import os
import tempfile
from contextlib import ExitStack, chdir, nullcontext
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import patch

import httpx2 as httpx
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.capabilities import ProcessEventStream
from pydantic_ai.exceptions import ModelHTTPError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.openrouter import OpenRouterModel
from pydantic_ai.providers.openrouter import OpenRouterProvider
from pydantic_ai.usage import RequestUsage

from mam_bench.config import AgentSettings, BenchmarkConfig
from mam_bench.diagnostics import ExecutionFailure
from mam_bench.runner import BenchmarkRunner
from mam_bench.sessions import (
    PROVIDER_RETRY_DELAYS,
    AgentSessions,
    Completed,
    stream_responses,
)
from mam_bench.telemetry import collect_telemetry
from support import (
    SandboxTestCase,
    civil,
    schelling,
    selection,
    served_model,
    stay,
    streaming_model,
)


class TelemetryTests(SandboxTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.enterContext(served_model())

    async def test_native_collection_keeps_full_history_and_exports_three_signals(self) -> None:
        calls = 0

        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(
                parts=[ToolCallPart("post_message", {"text": "PRIVATE POST"})]
                if calls == 1
                else [ThinkingPart("PRIVATE REASONING"), TextPart("PRIVATE ANSWER")],
                usage=RequestUsage(input_tokens=10, output_tokens=2),
            )

        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                os.environ,
                {
                    "OTEL_RESOURCE_ATTRIBUTES": "mam_bench.campaign=synthetic-campaign",
                    "MODAL_TOKEN_ID": "offline-id",
                    "MODAL_TOKEN_SECRET": "offline-secret",
                },
                clear=True,
            ),
        ):
            root = Path(directory)
            with self.assertLogs("mam_bench", level="DEBUG"), collect_telemetry(root):
                sessions = AgentSessions(
                    Agent(
                        streaming_model(FunctionModel(respond)),
                        capabilities=[ProcessEventStream(stream_responses)],
                    ),
                    context_window_tokens=260_000,
                )
                outcome = await sessions.run("agent-a", "PRIVATE PROMPT", deps=None)
            self.assertEqual(outcome, Completed("PRIVATE ANSWER"))
            self.assertEqual(sessions.usage.requests, 2)
            contents = ("PRIVATE PROMPT", "PRIVATE POST", "PRIVATE REASONING", "PRIVATE ANSWER")
            exported = {
                name: [
                    json.loads(line)
                    for line in (root / "telemetry" / f"{name}.jsonl").read_text().splitlines()
                ]
                for name in ("traces", "logs", "metrics")
            }
            self.assertTrue(
                all(exported.values()), {name: len(rows) for name, rows in exported.items()}
            )
            requests = [
                span
                for span in exported["traces"]
                if span["attributes"].get("gen_ai.operation.name") == "chat"
            ]
            self.assertEqual(len(requests), 2)
            self.assertEqual(
                {span["attributes"]["gen_ai.agent.call.id"] for span in requests},
                {message.run_id for message in sessions.history("agent-a")},
            )
            self.assertEqual(
                sum(span["attributes"]["gen_ai.usage.input_tokens"] for span in requests),
                sessions.usage.input_tokens,
            )
            for content in contents:
                self.assertIn(content, json.dumps(requests))
            self.assertTrue(all(record["trace_id"] for record in exported["logs"]))
            for records in exported.values():
                self.assertIn('"mam_bench.campaign": "synthetic-campaign"', json.dumps(records))
            self.assertEqual(sorted(path.name for path in root.iterdir()), ["telemetry"])

    async def test_nested_compaction_requests_are_collected_once(self) -> None:
        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[TextPart("synthetic summary or answer")],
                usage=RequestUsage(input_tokens=5, output_tokens=2),
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with collect_telemetry(root):
                sessions = AgentSessions(
                    Agent(streaming_model(FunctionModel(respond))),
                    context_window_tokens=200,
                    settings=AgentSettings(
                        compaction_trigger_fraction=0.5,
                        compaction_tail_tokens=20,
                        summary_completion_tokens=16,
                    ),
                )
                await sessions.run("agent", "A" * 600, deps=None)
                await sessions.run("agent", "B" * 600, deps=None)
            self.assertEqual(sessions.usage.requests, 3)
            spans = [
                json.loads(line)
                for line in (root / "telemetry/traces.jsonl").read_text().splitlines()
            ]
            requests = [
                span for span in spans if span["attributes"].get("gen_ai.operation.name") == "chat"
            ]
            self.assertEqual(len(requests), 3)
            self.assertEqual(
                sum(span["attributes"]["gen_ai.usage.input_tokens"] for span in requests), 15
            )

    async def test_provider_failure_keeps_partial_history_and_error_details(
        self,
    ) -> None:
        calls = 0

        async def respond(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            if calls == 1:
                return ModelResponse(
                    parts=[ToolCallPart("post_message", {"text": "retained post"})]
                )
            raise ModelHTTPError(400, "synthetic", "PRIVATE_PROVIDER_SENTINEL")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = AgentSessions(
                Agent(streaming_model(FunctionModel(respond))),
                context_window_tokens=260_000,
            )
            with collect_telemetry(root), self.assertRaises(ExecutionFailure) as failure:
                await sessions.run("agent-a", "retained prompt", deps=None)
            self.assertEqual(failure.exception.kind, "provider")
            for name in ("traces", "logs"):
                saved = (root / "telemetry" / f"{name}.jsonl").read_text()
                self.assertIn("PRIVATE_PROVIDER_SENTINEL", saved)
                self.assertIn("ModelHTTPError", saved)
            spans = [
                json.loads(line)
                for line in (root / "telemetry/traces.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any(span["status"]["status_code"] == "ERROR" for span in spans))
            self.assertIn("retained post", json.dumps(spans))
            self.assertIn("retained prompt", json.dumps(spans))

    async def test_streamed_provider_error_keeps_message_body_and_traceback(self) -> None:
        def respond(request: httpx.Request) -> httpx.Response:
            self.assertIs(json.loads(request.content)["stream"], True)
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text='data: {"error": {"message": "Provider returned error", "code": 502, '
                '"metadata": {"raw": "UPSTREAM_STREAM_FAILURE"}}}\n\n',
            )

        async with AsyncOpenAI(
            api_key="offline", http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        ) as client:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                model = OpenRouterModel(
                    "openai/gpt-6-luna", provider=OpenRouterProvider(openai_client=client)
                )
                sessions = AgentSessions(
                    Agent(model, capabilities=[ProcessEventStream(stream_responses)]),
                    context_window_tokens=260_000,
                )
                with collect_telemetry(root), self.assertRaises(ExecutionFailure) as failure:
                    await sessions.run("stream-error", "Respond.", deps=None)
                self.assertEqual(failure.exception.kind, "provider")
                self.assertEqual(sessions.usage.requests, len(PROVIDER_RETRY_DELAYS) + 1)
                logs = (root / "telemetry/logs.jsonl").read_text()
                self.assertIn("UPSTREAM_STREAM_FAILURE", logs)
                self.assertIn("502", logs)
                self.assertIn("Provider returned error", logs)
                traces = (root / "telemetry/traces.jsonl").read_text()
                self.assertIn("Provider returned error", traces)
                self.assertIn("openai.APIError", traces)
                self.assertIn("Traceback", traces)

    async def test_collection_preserves_paired_benchmark_results_and_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = BenchmarkConfig(
                model=selection(),
                cases=(schelling(max_steps=2), civil(max_steps=2)),
                output_directory=root,
            )
            plain = BenchmarkRunner(
                config, model_factory=lambda _: streaming_model(FunctionModel(stay))
            )
            expected = await plain.run()
            observed = BenchmarkRunner(
                config, model_factory=lambda _: streaming_model(FunctionModel(stay))
            )
            # The CLI enters and exits collection in the same task that owns the run.
            with ExitStack() as stack:

                def started(path: Path) -> None:
                    stack.enter_context(collect_telemetry(path))

                actual = await observed.run(on_output_directory=started)
            self.assertEqual(actual, expected)
            assert plain.output_directory is not None and observed.output_directory is not None
            for index in ("001", "002"):
                for name in ("ordinary.jsonl", "controlled.jsonl", "result.json"):
                    relative = Path("cases") / index / name
                    self.assertEqual(
                        (plain.output_directory / relative).read_bytes(),
                        (observed.output_directory / relative).read_bytes(),
                    )

    async def test_dotenv_export_and_environment_override_preserve_local_signals(self) -> None:
        received: list[tuple[str, bytes, str | None]] = []
        status = 200

        class Receiver(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if self.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                received.append((self.path, body, self.headers.get("Authorization")))
                self.send_response(status)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                pass

        with HTTPServer(("127.0.0.1", 0), Receiver) as server:
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                for status in (200, 400):
                    received.clear()
                    with (
                        tempfile.TemporaryDirectory() as directory,
                        patch.dict(
                            os.environ,
                            {
                                "LOGFIRE_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                                "MODAL_TOKEN_ID": "offline-id",
                                "MODAL_TOKEN_SECRET": "offline-secret",
                            },
                            clear=True,
                        ),
                    ):
                        root = Path(directory)
                        calls = 0
                        (root / ".env").write_text(
                            "LOGFIRE_TOKEN=synthetic-dotenv-token\n"
                            "LOGFIRE_BASE_URL=http://127.0.0.1:1\n"
                        )

                        async def respond(
                            messages: list[ModelMessage], info: AgentInfo
                        ) -> ModelResponse:
                            nonlocal calls
                            calls += 1
                            return ModelResponse(
                                parts=[ToolCallPart("post_message", {"text": "synthetic post"})]
                                if calls == 1
                                else [ThinkingPart("synthetic reasoning"), TextPart("done")],
                                usage=RequestUsage(input_tokens=10, output_tokens=2),
                            )

                        with (
                            (
                                self.assertLogs("opentelemetry", level="ERROR")
                                if status == 400
                                else nullcontext()
                            ),
                            self.assertLogs("mam_bench", level="DEBUG"),
                            chdir(root),
                            collect_telemetry(root, send_to_logfire=True),
                        ):
                            sessions = AgentSessions(
                                Agent(streaming_model(FunctionModel(respond))),
                                context_window_tokens=260_000,
                            )
                            outcome = await sessions.run("agent", "synthetic input", deps=None)
                        self.assertEqual(outcome, Completed("done"))
                        self.assertEqual(
                            {path for path, _, _ in received},
                            {"/v1/traces", "/v1/logs", "/v1/metrics"},
                        )
                        for signal in ("traces", "logs", "metrics"):
                            self.assertTrue((root / "telemetry" / f"{signal}.jsonl").read_text())
                        self.assertTrue(sessions.history("agent"))
                        traces = b"".join(
                            body for path, body, _ in received if path == "/v1/traces"
                        )
                        local = (root / "telemetry/traces.jsonl").read_text()
                        for content in (
                            "synthetic input",
                            "synthetic post",
                            "synthetic reasoning",
                            "done",
                        ):
                            self.assertIn(content.encode(), traces)
                            self.assertIn(content, local)
                        self.assertTrue(
                            all(token == "synthetic-dotenv-token" for _, _, token in received)
                        )
            finally:
                server.shutdown()
                worker.join()
