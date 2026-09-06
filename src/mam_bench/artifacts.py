"""Native agent archives with the package's error-content protections."""

from collections.abc import Awaitable
from dataclasses import replace

from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    FileStepStore,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)

from mam_bench.benchmark import AgentInfrastructureFailure


class AgentMessageArchive(FileStepStore):
    """Keep native snapshots intact without persisting provider exception bodies."""

    async def _persist(self, operation: Awaitable[None]) -> None:
        try:
            await operation
        except OSError as error:
            raise AgentInfrastructureFailure(
                "artifact_write", "agent archive write failed"
            ) from error

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
