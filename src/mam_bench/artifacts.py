"""Native agent archives with the package's error-content protections."""

import json
from collections.abc import Awaitable
from dataclasses import replace
from pathlib import Path

from pydantic import BaseModel
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)

from mam_bench.diagnostics import ExecutionFailure


class ArtifactWriter:
    """Write supplied records; simulations decide their content and timing."""

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=False)
        self.directory = directory

    def append(self, name: str, record: object) -> None:
        with (self.directory / name).open("a", encoding="utf-8") as output:
            output.write(json.dumps(record, allow_nan=False) + "\n")

    def write(self, name: str, record: BaseModel | dict[str, object]) -> None:
        text = (
            record.model_dump_json(indent=2)
            if isinstance(record, BaseModel)
            else json.dumps(record, indent=2, allow_nan=False)
        )
        path = self.directory / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(text + "\n", encoding="utf-8")
        temporary.replace(path)


class AgentMessageArchive(FileStepStore):
    """Keep native snapshots intact without persisting provider exception bodies."""

    async def _persist(self, operation: Awaitable[None]) -> None:
        try:
            await operation
        except OSError as error:
            raise ExecutionFailure("artifact_write", "agent archive write failed") from error

    async def register_run(self, record: RunRecord) -> None:
        await self._persist(super().register_run(record))

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        await self._persist(super().save_snapshot(snapshot))

    async def append_event(self, event: StepEvent) -> None:
        if event.error is not None:
            event = replace(event, error="Exception details omitted; see diagnostic log for types.")
        await self._persist(super().append_event(event))

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        if record.status == "failed":
            record = replace(record, effect_summary="Exception details omitted.")
        await self._persist(super().record_tool_effect(record))
