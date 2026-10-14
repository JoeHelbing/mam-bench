import json
import os
import tempfile
import unittest
from contextlib import ExitStack, chdir
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from threading import Thread
from unittest.mock import AsyncMock, patch

from pydantic_ai import Agent, ModelMessagesTypeAdapter
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ThinkingPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.usage import RequestUsage
from pydantic_ai_harness.step_persistence import FileStepStore

from mam_bench.config import AgentSettings, BenchmarkConfig
from mam_bench.diagnostics import ExecutionFailure
from mam_bench.model_setup import VLLMModelMetadata
from mam_bench.runner import BenchmarkRunner
from mam_bench.sessions import AgentSessions, Completed
from mam_bench.telemetry import collect_telemetry
from support import civil, schelling, selection, stay


class TelemetryTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        discovery = patch(
            "mam_bench.model_setup._fetch_vllm_model",
            AsyncMock(return_value=VLLMModelMetadata(id="scripted", max_model_len=260_000)),
        )
        discovery.start()
        self.addCleanup(discovery.stop)

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
                {"OTEL_RESOURCE_ATTRIBUTES": "mam_bench.campaign=synthetic-campaign"},
                clear=True,
            ),
        ):
            root = Path(directory)
            with self.assertLogs("mam_bench", level="DEBUG"), collect_telemetry(root):
                sessions = AgentSessions(
                    Agent(FunctionModel(respond)),
                    context_window_tokens=260_000,
                    artifact_directory=root,
                )
                outcome = await sessions.run("agent-a", "PRIVATE PROMPT", deps=None)
            self.assertEqual(outcome, Completed("PRIVATE ANSWER"))
            self.assertEqual(sessions.usage.requests, 2)
            store = FileStepStore(root / "agent-messages", media_store=None)
            runs = await store.list_runs(conversation_id="agent-a")
            snapshots = await store.list_snapshots(run_id=runs[0].run_id)
            history = ModelMessagesTypeAdapter.dump_json(snapshots[-1].messages)
            for content in (
                b"PRIVATE PROMPT",
                b"PRIVATE POST",
                b"PRIVATE REASONING",
                b"PRIVATE ANSWER",
            ):
                self.assertIn(content, history)
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
                {message.run_id for message in snapshots[-1].messages},
            )
            self.assertEqual(
                sum(span["attributes"]["gen_ai.usage.input_tokens"] for span in requests), 20
            )
            self.assertNotIn("PRIVATE", json.dumps(exported))
            self.assertTrue(all(record["trace_id"] for record in exported["logs"]))
            for records in exported.values():
                self.assertIn('"mam_bench.campaign": "synthetic-campaign"', json.dumps(records))

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
                    Agent(FunctionModel(respond)),
                    context_window_tokens=200,
                    settings=AgentSettings(
                        compaction_trigger_fraction=0.5,
                        compaction_tail_tokens=20,
                        summary_completion_tokens=16,
                        memory_injection_tokens=100,
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

    async def test_provider_failure_keeps_partial_history_without_exporting_error_body(
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
            raise ModelAPIError("synthetic", "PRIVATE_PROVIDER_SENTINEL")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sessions = AgentSessions(
                Agent(FunctionModel(respond)),
                context_window_tokens=260_000,
                artifact_directory=root,
            )
            with collect_telemetry(root), self.assertRaises(ExecutionFailure) as failure:
                await sessions.run("agent-a", "retained prompt", deps=None)
            self.assertEqual(failure.exception.kind, "provider")
            store = FileStepStore(root / "agent-messages", media_store=None)
            runs = await store.list_runs(conversation_id="agent-a")
            snapshots = await store.list_snapshots(run_id=runs[0].run_id)
            history = ModelMessagesTypeAdapter.dump_json(snapshots[-1].messages)
            self.assertIn(b"retained post", history)
            self.assertIn(b"retained prompt", history)
            saved = "\n".join(path.read_text() for path in root.rglob("*.json*"))
            self.assertNotIn("PRIVATE_PROVIDER_SENTINEL", saved)
            self.assertIn("ModelAPIError", saved)
            spans = [
                json.loads(line)
                for line in (root / "telemetry/traces.jsonl").read_text().splitlines()
            ]
            self.assertTrue(any(span["status"]["status_code"] == "ERROR" for span in spans))

    async def test_collection_preserves_paired_benchmark_results_and_trajectories(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = BenchmarkConfig(
                model=selection(),
                cases=(schelling(max_steps=2), civil(max_steps=2)),
                output_directory=root,
            )
            plain = BenchmarkRunner(config, model_factory=lambda _: FunctionModel(stay))
            expected = await plain.run()
            observed = BenchmarkRunner(config, model_factory=lambda _: FunctionModel(stay))
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
        received: list[str] = []

        class Reject(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                self.rfile.read(int(self.headers["Content-Length"]))
                received.append(self.path)
                self.send_response(400)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                pass

        with HTTPServer(("127.0.0.1", 0), Reject) as server:
            worker = Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                with (
                    tempfile.TemporaryDirectory() as directory,
                    patch.dict(
                        os.environ,
                        {
                            "LOGFIRE_BASE_URL": f"http://127.0.0.1:{server.server_port}",
                        },
                        clear=True,
                    ),
                ):
                    root = Path(directory)
                    (root / ".env").write_text(
                        "LOGFIRE_TOKEN=synthetic-dotenv-token\n"
                        "LOGFIRE_BASE_URL=http://127.0.0.1:1\n"
                    )

                    async def respond(
                        messages: list[ModelMessage], info: AgentInfo
                    ) -> ModelResponse:
                        return ModelResponse(
                            parts=[TextPart("done")],
                            usage=RequestUsage(input_tokens=10, output_tokens=2),
                        )

                    with (
                        self.assertLogs("opentelemetry", level="ERROR"),
                        self.assertLogs("mam_bench", level="DEBUG"),
                        chdir(root),
                        collect_telemetry(root, send_to_logfire=True),
                    ):
                        sessions = AgentSessions(
                            Agent(FunctionModel(respond)),
                            context_window_tokens=260_000,
                            artifact_directory=root,
                        )
                        outcome = await sessions.run("agent", "synthetic input", deps=None)
                    self.assertEqual(outcome, Completed("done"))
                    self.assertEqual(set(received), {"/v1/traces", "/v1/logs", "/v1/metrics"})
                    for signal in ("traces", "logs", "metrics"):
                        self.assertTrue((root / "telemetry" / f"{signal}.jsonl").read_text())
                    self.assertTrue(sessions.history("agent"))
            finally:
                server.shutdown()
                worker.join()
