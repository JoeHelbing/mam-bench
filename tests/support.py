"""Explicit development fixtures and scripted provider responses; no suite calibration."""

import json
import os
import tempfile
import unittest
from collections.abc import AsyncIterator, Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast
from unittest.mock import patch

import httpx2 as httpx
from openai import AsyncOpenAI
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import (
    AgentInfo,
    DeltaThinkingCalls,
    DeltaThinkingPart,
    DeltaToolCall,
    DeltaToolCalls,
    FunctionModel,
)
from pydantic_ai.workspaces import LocalWorkspaceBackend, WorkspaceRef

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import AgentSettings, ModelSelection
from mam_bench.runtime import CaseRuntime
from mam_bench.sandboxes import ModalSettings
from mam_bench.sessions import PROVIDER_RETRY_DELAYS
from mam_bench.simulations.civil_violence.settings import CivilViolenceSettings
from mam_bench.simulations.schelling.settings import SchellingSettings

# Fail closed: no test may reach Modal with the developer's real credentials, even one
# that forgets SandboxTestCase. Ignore `.env` (a cleared os.environ would fall back to
# it) and default the process to offline values, which real Modal rejects.
ModalSettings.model_config["env_file"] = None
os.environ.update(MODAL_TOKEN_ID="offline-id", MODAL_TOKEN_SECRET="offline-secret")


class OfflineSandbox(LocalWorkspaceBackend):
    """Exercise real file/command tools without contacting the Modal service."""

    def __init__(self, root: Path, options: dict[str, object]) -> None:
        super().__init__(root)
        self.options = options
        self.closed = False

    @property
    def ref(self) -> WorkspaceRef:
        return WorkspaceRef(provider="modal", id=str(self._working_dir))

    async def get_sandbox(self) -> None:
        pass


class SandboxTestCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        super().setUp()
        scratch = Path(".scratch/test-sandboxes")
        scratch.mkdir(parents=True, exist_ok=True)
        directory = tempfile.TemporaryDirectory(dir=scratch)
        self.addCleanup(directory.cleanup)
        self.sandboxes: list[OfflineSandbox] = []

        def backend(**options: object) -> OfflineSandbox:
            root = Path(directory.name).resolve() / str(len(self.sandboxes))
            root.mkdir()
            value = OfflineSandbox(root, options)
            self.sandboxes.append(value)
            return value

        async def destroy(_capability: object, ref: WorkspaceRef) -> None:
            next(value for value in self.sandboxes if value.ref == ref).closed = True

        for replacement in (
            patch("pydantic_ai_harness.modal_sandbox.ModalSandboxBackend", backend),
            patch("pydantic_ai_harness.modal_sandbox.ModalSandbox.destroy", destroy),
            # Keep the retry count but skip the real backoff waits.
            patch("mam_bench.sessions.PROVIDER_RETRY_DELAYS", (0.0,) * len(PROVIDER_RETRY_DELAYS)),
            patch.dict(
                os.environ, MODAL_TOKEN_ID="offline-id", MODAL_TOKEN_SECRET="offline-secret"
            ),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)


def schelling(**updates: object) -> SchellingSettings:
    return SchellingSettings.model_validate(
        {
            "simulation": "schelling",
            "board_size": 8,
            "tolerance": 0.75,
            "vacancy_fraction": 0.25,
            "seed": 7,
            "max_steps": 3,
            "vision_radius": 2,
            "controlled_agent_count": 4,
            "objective": "integration",
            **updates,
        }
    )


def civil(**updates: object) -> CivilViolenceSettings:
    return CivilViolenceSettings.model_validate(
        {
            "simulation": "civil-violence",
            "controlled_role": "citizen",
            "objective": "increase",
            "board_size": 8,
            "citizen_density": 0.75,
            "police_density": 0.0,
            "vision_radius": 1,
            "threshold": -100.0,
            "private_preference_mean": 0.0,
            "private_preference_std": 0.0,
            "max_jail_term": 2,
            "seed": 7,
            "max_steps": 30,
            "controlled_agent_count": 16,
            **updates,
        }
    )


def selection() -> ModelSelection:
    return ModelSelection(
        runtime="vllm",
        model="scripted",
        base_url="http://127.0.0.1:1/v1",
    )


@contextmanager
def served_model(context_window: int = 260_000) -> Generator[None]:
    """Serve real offline discovery requests for scripted benchmark runs."""

    def respond(request: httpx.Request) -> httpx.Response:
        assert (request.method, request.url.path) == ("GET", "/v1/models")
        return httpx.Response(
            200,
            json={"data": [{"id": "scripted", "max_model_len": context_window}]},
        )

    def client(**_: object) -> AsyncOpenAI:
        return AsyncOpenAI(
            api_key="offline-test-key",
            base_url="https://provider.example.test/v1",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
        )

    with (
        patch.dict(os.environ, {"UNUSED_OFFLINE_TEST_KEY": "offline-test-key"}),
        patch("mam_bench.model_setup.AsyncOpenAI", side_effect=client),
    ):
        yield


def observation(messages: list[ModelMessage]) -> dict[str, object]:
    for message in reversed(messages):
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    content = part.content
                    text = (
                        content
                        if isinstance(content, str)
                        else next(
                            (
                                item.content
                                for item in content
                                if isinstance(item, TextContent) and item.content.startswith("{")
                            ),
                            "",
                        )
                    )
                    if text.startswith("{"):
                        return cast(dict[str, object], json.loads(text))
    raise AssertionError("missing JSON observation")


def turn_calls(messages: list[ModelMessage]) -> list[str]:
    """Function-tool calls since the latest observation (not previous turns)."""
    start = max(
        index
        for index, message in enumerate(messages)
        if isinstance(message, ModelRequest)
        and any(
            isinstance(part, UserPromptPart)
            and isinstance(part.content, str)
            and part.content.startswith("{")
            for part in message.parts
        )
    )
    return [
        part.tool_name
        for message in messages[start + 1 :]
        if isinstance(message, ModelResponse)
        for part in message.parts
        if isinstance(part, ToolCallPart)
    ]


async def stay(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    names = {tool.name for tool in info.output_tools}
    if "act" in names:
        view = observation(messages)
        if cast(dict[str, object], view["self"])["jailed"]:
            part = ToolCallPart("defer", {})
        elif "eligible_target_locations_if_staying" in view:
            part = ToolCallPart("act", {"destination": None, "target_location": None})
        else:
            part = ToolCallPart("act", {"active": False, "destination": None})
    else:
        part = ToolCallPart("move", {"destination": None})
    return ModelResponse(parts=[part])


def streaming_model(model: FunctionModel) -> FunctionModel:
    """Expose scripted replies as native chunks for streaming simulation turns."""

    async def stream(
        messages: list[ModelMessage], info: AgentInfo
    ) -> AsyncIterator[str | DeltaToolCalls | DeltaThinkingCalls]:
        response = await model.request(messages, info.model_settings, info.model_request_parameters)
        for index, part in enumerate(response.parts):
            if isinstance(part, TextPart):
                yield part.content
            elif isinstance(part, ThinkingPart):
                yield {index: DeltaThinkingPart(content=part.content)}
            elif isinstance(part, ToolCallPart):
                yield {
                    index: DeltaToolCall(
                        part.tool_name, part.args_as_json_str(), tool_call_id=part.tool_call_id
                    )
                }

    model.stream_function = stream
    return model


def runtime(path: Path, model: FunctionModel | None = None, **settings: object) -> CaseRuntime:
    return CaseRuntime(
        streaming_model(model or FunctionModel(stay)),
        AgentSettings.model_validate({"concurrency": 2, "timeout_seconds": 5, **settings}),
        ArtifactWriter(path),
        context_window_tokens=260_000,
    )


def records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]
