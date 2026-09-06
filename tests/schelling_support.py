"""Script only provider responses; all sessions and simulation tools remain real."""

import re
from collections.abc import Awaitable, Callable

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from mam_bench.config import AgentSettings
from mam_bench.model import ModelRuntime, RuntimeInfo


def identity(messages: list[ModelMessage]) -> tuple[int, int]:
    prompts = [
        str(part.content)
        for message in messages
        if isinstance(message, ModelRequest)
        for part in message.parts
        if isinstance(part, UserPromptPart)
    ]
    for prompt in reversed(prompts):
        match = re.search(r"Round (\d+) of 30 for ModelControlledAgent (\d+)", prompt)
        if match:
            return int(match[1]), int(match[2])
    raise AssertionError("missing actor observation")


async def staying(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart("stay", {})])


def runtime_for(
    script: Callable[[list[ModelMessage], AgentInfo], Awaitable[ModelResponse]] = staying,
    *,
    concurrency: int = 4,
) -> ModelRuntime:
    return ModelRuntime(
        info=RuntimeInfo(
            model_id="offline",
            provider="openai-compatible",
            model="offline",
            endpoint="http://127.0.0.1",
            agent_settings=AgentSettings(concurrency=concurrency),
        ),
        model=FunctionModel(script),
    )
