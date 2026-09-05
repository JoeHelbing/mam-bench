import asyncio
import unittest

import numpy as np
from pydantic_ai import ModelRetry

from mam_bench.benchmark import AgentSettings, RuntimeInfo
from mam_bench.communication import SharedCommunication
from mam_bench.simulations.schelling.agent import SchellingTurnCoordinator
from mam_bench.simulations.schelling.models import (
    ActorTurnContext,
    ActorTurnCoordinator,
    ActorTurnResult,
    AppendMemory,
    InfluenceEvaluationConfig,
    PublicDocumentRecord,
    Stay,
    SteeringObjective,
    SubmitMove,
)
from mam_bench.simulations.schelling.profile import LandscapeCell, Rational
from mam_bench.simulations.schelling.reference import (
    EMPTY_CELL,
    reference_rng,
    reserve_ordinary_destinations,
    unhappy_agent_ids,
)
from mam_bench.simulations.schelling.runtime import (
    actor_admission_order,
    influence_actor_ids,
    resolve_staged_round,
)
from mam_bench.simulations.schelling.utils.reference_data import build_evaluation_reference


class CountingNotebook:
    def __init__(self) -> None:
        self.write_count = 0

    async def write(
        self,
        content: str,
        *,
        file: str = "MEMORY.md",
        old_text: str | None = None,
    ) -> object:
        del content, file, old_text
        self.write_count += 1
        return object()

    async def delete(self, file: str) -> object:
        del file
        return object()


class ControlledRoundTeam:
    runtime_info = RuntimeInfo(
        model_id="controlled-round",
        provider="openrouter",
        model="test/controlled-round",
        endpoint="https://example.test/v1",
        agent_settings=AgentSettings(concurrency=2),
    )

    def __init__(
        self,
        admission_order: tuple[int, ...],
        *,
        first_destination: int,
        retry_destination: int,
        actor_origin: int,
        size: int,
    ) -> None:
        self._admission_index = {actor_id: index for index, actor_id in enumerate(admission_order)}
        self._first_destination = first_destination
        self._retry_destination = retry_destination
        self._actor_origin = actor_origin
        self._size = size
        self._release_first = asyncio.Event()
        self.first_started = asyncio.Event()
        self.third_started = asyncio.Event()
        self.other_turns_completed = asyncio.Event()
        self.active = 0
        self.maximum_active = 0
        self.completed_other_turns = 0
        self.contexts: dict[int, ActorTurnContext] = {}
        self.notebooks: dict[int, CountingNotebook] = {}

    def release_first(self) -> None:
        self._release_first.set()

    async def run_turn(
        self,
        context: ActorTurnContext,
        coordinator: ActorTurnCoordinator,
    ) -> ActorTurnResult:
        index = self._admission_index[context.actor_id]
        notebook = self.notebooks.setdefault(context.actor_id, CountingNotebook())
        self.contexts[context.actor_id] = context
        self.active += 1
        self.maximum_active = max(self.maximum_active, self.active)
        if index == 0:
            self.first_started.set()
        if index == 2:
            self.third_started.set()
        try:
            if index == 0:
                await self._release_first.wait()
            await coordinator.post_message(context.actor_id, f"actor-{context.actor_id}")
            rejections: tuple[str, ...] = ()
            if index == 1:
                action = self._move(self._first_destination)
            elif index == 2:
                rejected = self._move(
                    self._first_destination,
                    memory=AppendMemory(operation="append", content="must not persist"),
                )
                try:
                    await coordinator.commit_terminal(context.actor_id, rejected, notebook)
                except ModelRetry as error:
                    rejections = (str(error),)
                action = self._move(self._retry_destination)
            elif index == 3:
                rejected = self._move(self._actor_origin)
                try:
                    await coordinator.commit_terminal(context.actor_id, rejected, notebook)
                except ModelRetry as error:
                    rejections = (str(error),)
                action = Stay()
            else:
                action = Stay()
            committed = await coordinator.commit_terminal(
                context.actor_id,
                action,
                notebook,
            )
            return ActorTurnResult(
                actor_id=context.actor_id,
                action=committed,
                accepted_call_count=1,
                policy_rejections=rejections,
            )
        finally:
            self.active -= 1
            if index != 0:
                self.completed_other_turns += 1
                if self.completed_other_turns == 15:
                    self.other_turns_completed.set()

    def _move(
        self,
        destination: int,
        *,
        memory: AppendMemory | None = None,
    ) -> SubmitMove:
        row, column = divmod(destination, self._size)
        return SubmitMove(row=row, column=column, memory=memory)


class StagedRoundTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.cell = LandscapeCell(20, Rational(3, 4), Rational(1, 4))
        self.config = InfluenceEvaluationConfig(
            cell=self.cell,
            seed_id=50,
            objective=SteeringObjective.INTEGRATION,
        )
        self.reference = build_evaluation_reference(self.cell, self.config.seed_id)

    def test_actor_admission_has_its_own_semantic_rng_stream(self) -> None:
        first = actor_admission_order(self.config, round_number=1)
        repeated = actor_admission_order(self.config, round_number=1)
        second = actor_admission_order(self.config, round_number=2)

        self.assertEqual(first, repeated)
        self.assertEqual(set(first), set(influence_actor_ids(self.cell.agent_count)))
        self.assertNotEqual(first, second)

        order_rng = reference_rng(self.cell, self.config.seed_id, 1)
        tie_rng = reference_rng(self.cell, self.config.seed_id, 2)
        order_control = reference_rng(self.cell, self.config.seed_id, 1)
        tie_control = reference_rng(self.cell, self.config.seed_id, 2)
        actor_admission_order(self.config, round_number=3)
        np.testing.assert_array_equal(
            order_rng.integers(100, size=8),
            order_control.integers(100, size=8),
        )
        np.testing.assert_array_equal(
            tie_rng.integers(100, size=8),
            tie_control.integers(100, size=8),
        )

    async def test_resolves_one_completion_ordered_concurrent_staged_round(self) -> None:
        initial_cell_types = self.reference.initial_cell_types.copy()
        initial_locations = self.reference.initial_agent_locations.copy()
        admission = actor_admission_order(self.config, round_number=1)
        vacancies = np.flatnonzero(initial_cell_types.ravel() == EMPTY_CELL)
        second_actor = admission[1]
        second_location = int(initial_locations[second_actor])
        second_row, second_column = divmod(second_location, self.cell.board_size)
        nearby = {
            ((second_row + row_delta) % self.cell.board_size) * self.cell.board_size
            + ((second_column + column_delta) % self.cell.board_size)
            for row_delta in (-1, 0, 1)
            for column_delta in (-1, 0, 1)
        }
        far_vacancies = [int(item) for item in vacancies if int(item) not in nearby]
        first_destination, retry_destination = far_vacancies[:2]
        excluded_origin = int(initial_locations[admission[4]])
        team = ControlledRoundTeam(
            admission,
            first_destination=first_destination,
            retry_destination=retry_destination,
            actor_origin=excluded_origin,
            size=self.cell.board_size,
        )
        communication: SharedCommunication[PublicDocumentRecord] = SharedCommunication()
        coordinator = SchellingTurnCoordinator(communication)

        round_task = asyncio.create_task(
            resolve_staged_round(
                self.config,
                self.reference,
                team,
                coordinator,
                cell_types=initial_cell_types,
                agent_locations=initial_locations,
                agent_types=self.reference.agent_types,
                round_number=1,
                order_rng=reference_rng(self.cell, self.config.seed_id, 1),
                tie_rng=reference_rng(self.cell, self.config.seed_id, 2),
            )
        )
        await team.first_started.wait()
        await team.third_started.wait()
        await team.other_turns_completed.wait()

        np.testing.assert_array_equal(initial_cell_types, self.reference.initial_cell_types)
        np.testing.assert_array_equal(initial_locations, self.reference.initial_agent_locations)
        self.assertFalse(round_task.done())
        team.release_first()
        result = await round_task

        self.assertEqual(team.maximum_active, 2)
        self.assertEqual(result.actor_admission_order, admission)
        self.assertEqual(result.actor_turns[-1].item, admission[0])
        self.assertEqual(
            [completed.completion_sequence for completed in result.actor_turns],
            list(range(16)),
        )
        self.assertEqual(len(team.contexts), 16)
        self.assertNotIn(
            first_destination,
            {
                item.location.row * self.cell.board_size + item.location.column
                for item in team.contexts[second_actor].neighborhood
            },
        )

        collision_result = next(
            completed.result for completed in result.actor_turns if completed.item == admission[2]
        )
        origin_result = next(
            completed.result for completed in result.actor_turns if completed.item == admission[3]
        )
        self.assertEqual(len(collision_result.policy_rejections), 1)
        self.assertEqual(len(origin_result.policy_rejections), 1)
        self.assertEqual(team.notebooks[admission[2]].write_count, 0)
        self.assertEqual(
            set(result.actor_destinations.tolist()),
            {first_destination, retry_destination},
        )

        records = communication.records
        self.assertEqual(records[0].author_session_id, str(admission[1]))
        self.assertEqual(records[-1].author_session_id, str(admission[0]))
        authoritative = [
            record for record in records if record.author_session_id == "schelling-runtime"
        ]
        self.assertEqual(len(authoritative), 2)

        unhappy = unhappy_agent_ids(
            self.reference.initial_cell_types,
            self.reference.initial_agent_locations,
            self.cell.tolerance,
        )
        ordinary_ids = unhappy[
            ~np.isin(unhappy, np.asarray(influence_actor_ids(self.cell.agent_count)))
        ]
        expected_agents, expected_destinations = reserve_ordinary_destinations(
            self.reference.initial_cell_types,
            self.reference.initial_agent_locations,
            self.reference.agent_types,
            ordinary_ids,
            self.cell.tolerance,
            reference_rng(self.cell, self.config.seed_id, 1),
            reference_rng(self.cell, self.config.seed_id, 2),
            unavailable_destinations=np.asarray(
                [first_destination, retry_destination], dtype=np.uint16
            ),
        )
        np.testing.assert_array_equal(result.moving_ordinary_agent_ids, expected_agents)
        np.testing.assert_array_equal(result.ordinary_destinations, expected_destinations)
        all_destinations = np.concatenate((result.actor_destinations, result.ordinary_destinations))
        self.assertEqual(len(np.unique(all_destinations)), len(all_destinations))
        for completed in result.actor_turns:
            if isinstance(completed.result.action, SubmitMove):
                origin = int(initial_locations[completed.item])
                self.assertEqual(result.cell_types.ravel()[origin], EMPTY_CELL)


if __name__ == "__main__":
    unittest.main()
