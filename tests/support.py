"""Explicit development fixtures and scripted provider responses; no suite calibration."""

import json
from pathlib import Path
from typing import cast

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextContent,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from mam_bench.artifacts import ArtifactWriter
from mam_bench.config import AgentSettings, OpenAICompatibleModel
from mam_bench.runtime import CaseRuntime
from mam_bench.simulations.civil_violence.settings import CivilViolenceSettings
from mam_bench.simulations.schelling.settings import SchellingSettings


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


def selection() -> OpenAICompatibleModel:
    return OpenAICompatibleModel(
        runtime="openai-compatible",
        model="scripted",
        base_url="http://127.0.0.1:1/v1",
        api_key_env="UNUSED_OFFLINE_TEST_KEY",
    )


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
        part = ToolCallPart("stay", {})
    return ModelResponse(parts=[part])


def runtime(path: Path, model: FunctionModel | None = None, **settings: object) -> CaseRuntime:
    return CaseRuntime(
        model or FunctionModel(stay),
        AgentSettings.model_validate({"concurrency": 2, "timeout_seconds": 5, **settings}),
        ArtifactWriter(path),
    )


def records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]
