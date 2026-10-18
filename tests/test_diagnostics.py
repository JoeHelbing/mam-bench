import asyncio
import os
import unittest
from unittest.mock import patch

import httpx2 as httpx
from openai import APITimeoutError
from pydantic_ai import Agent, ModelMessage
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from mam_bench.config import AgentSettings
from mam_bench.diagnostics import ExecutionFailure, configure_logging
from mam_bench.sessions import AgentSessions, Completed, Stopped


class DiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    def test_environment_log_level_overrides_config(self) -> None:
        with patch("mam_bench.diagnostics.logging.getLogger") as get_logger:
            with patch.dict(os.environ, {}, clear=True):
                configure_logging("DEBUG")
            get_logger.return_value.setLevel.assert_called_with("DEBUG")
            with patch.dict(os.environ, {"MAM_BENCH_LOG_LEVEL": "warning"}):
                configure_logging("DEBUG")
            get_logger.return_value.setLevel.assert_called_with("WARNING")

    async def test_turn_deadline_identifies_session_and_preserves_failure(self) -> None:
        async def slow(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            await asyncio.Event().wait()
            return ModelResponse(parts=[])

        runtime = AgentSessions(
            Agent(FunctionModel(slow)),
            context_window_tokens=260_000,
            settings=AgentSettings(timeout_seconds=0.01),
        )
        with (
            self.assertLogs("mam_bench", level="DEBUG") as captured,
            self.assertRaises(ExecutionFailure) as failure,
        ):
            await runtime.run("agent-7", "PRIVATE PROMPT", deps=None)
        self.assertEqual(failure.exception.kind, "timeout")
        log = "\n".join(captured.output)
        self.assertIn("scope=turn_deadline", log)
        self.assertIn("session=agent-7", log)
        self.assertIn("CancelledError", log)
        self.assertNotIn("PRIVATE PROMPT", log)

    async def test_http_timeout_is_distinct_and_keeps_original_message(self) -> None:
        async def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise APITimeoutError(request=httpx.Request("POST", "https://SECRET.invalid"))

        runtime = AgentSessions(Agent(FunctionModel(broken)), context_window_tokens=260_000)
        with self.assertLogs("mam_bench", level="DEBUG") as captured:
            self.assertEqual(
                await runtime.run("actor", "SECRET PROMPT", deps=None), Stopped("call_error_limit")
            )
        self.assertEqual(runtime.usage.requests, 3)
        log = "\n".join(captured.output)
        self.assertIn("request.failed session=actor failures=3", log)
        self.assertIn("APITimeoutError", log)
        self.assertIn("Request timed out.", log)

    async def test_turn_summary_logs_usage_without_content(self) -> None:
        count = 0

        async def scripted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal count
            count += 1
            if count == 1:
                return ModelResponse(parts=[ToolCallPart("post_message", {"text": "PRIVATE POST"})])
            return ModelResponse(parts=[TextPart("PRIVATE OUTPUT")])

        runtime = AgentSessions(Agent(FunctionModel(scripted)), context_window_tokens=260_000)
        with self.assertLogs("mam_bench", level="DEBUG") as captured:
            self.assertEqual(
                await runtime.run("a", "PRIVATE INPUT", deps=None), Completed("PRIVATE OUTPUT")
            )
        log = "\n".join(captured.output)
        self.assertIn("turn.start", log)
        self.assertIn("turn.end", log)
        self.assertIn("tools=1", log)
        self.assertIn("requests=2", log)
        self.assertNotIn("PRIVATE", log)
