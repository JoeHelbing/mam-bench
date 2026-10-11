"""Public scheduling and addressed-randomness contract."""

import asyncio
import gc
import unittest
from collections.abc import Iterable

from mam_bench.scheduling import TurnScheduler


class TurnSchedulerTests(unittest.IsolatedAsyncioTestCase):
    async def test_selected_cohort_finishes_before_ordinary_cohort(self) -> None:
        scheduler = TurnScheduler(seed=12, concurrency=3)
        accepted: list[int] = []

        async def act(identity: int) -> None:
            accepted.append(identity)

        await scheduler.run(step=0, phase="citizen", selected=(2, 1), ordinary=(4, 3), act=act)
        self.assertEqual(set(accepted[:2]), {1, 2})
        self.assertEqual(set(accepted[2:]), {3, 4})

    async def test_admission_ignores_caller_order_and_rejects_overlap(self) -> None:
        scheduler = TurnScheduler(seed=77, concurrency=3)

        async def order(selected: Iterable[int], ordinary: Iterable[int]) -> list[int]:
            accepted: list[int] = []

            async def act(identity: int) -> None:
                accepted.append(identity)

            await scheduler.run(
                step=3, phase="police", selected=selected, ordinary=ordinary, act=act
            )
            return accepted

        first = await order((7, 1, 4), (5, 8, 2))
        self.assertEqual(first, await order((1, 4, 7), (2, 5, 8)))
        self.assertEqual(first, await order({4, 7, 1}, {8, 2, 5}))
        with self.assertRaises(ValueError):
            await order((1,), (1,))

    async def test_ready_turns_finish_without_waiting_for_earlier_turns(self) -> None:
        scheduler = TurnScheduler(seed=71, concurrency=4)
        identities = [4, 3, 2, 1]
        expected = sorted(identities)
        scheduler.rng(0, "police", 0, "_order_selected").shuffle(expected)
        completed = {i: asyncio.Event() for i in identities}
        all_started = asyncio.Event()
        started: set[int] = set()
        accepted: list[int] = []
        accepted_later = asyncio.Event()

        async def act(identity: int) -> None:
            started.add(identity)
            if len(started) == len(identities):
                all_started.set()
            await completed[identity].wait()
            accepted.append(identity)
            if len(accepted) == len(identities) - 1:
                accepted_later.set()

        run = asyncio.create_task(
            scheduler.run(step=0, phase="police", selected=identities, ordinary=(), act=act)
        )
        try:
            await asyncio.wait_for(all_started.wait(), 1)
            for identity in reversed(expected[1:]):
                completed[identity].set()
            await asyncio.wait_for(accepted_later.wait(), 1)
            self.assertEqual(accepted, list(reversed(expected[1:])))
            self.assertFalse(run.done())
        finally:
            completed[expected[0]].set()
            await asyncio.wait_for(run, 1)
        self.assertEqual(accepted[-1], expected[0])

    async def test_retry_does_not_block_other_turns(self) -> None:
        scheduler = TurnScheduler(seed=1, concurrency=2)
        order = [1, 2]
        scheduler.rng(0, "police", 0, "_order_selected").shuffle(order)
        first_attempted = asyncio.Event()
        release_first = asyncio.Event()
        accepted: list[int] = []
        second_finished = asyncio.Event()

        async def act(identity: int) -> None:
            if identity == order[0]:
                first_attempted.set()  # A rejected model claim returns to its caller.
                await release_first.wait()
            accepted.append(identity)
            if identity == order[1]:
                second_finished.set()

        run = asyncio.create_task(
            scheduler.run(step=0, phase="police", selected=(1, 2), ordinary=(), act=act)
        )
        try:
            await asyncio.wait_for(first_attempted.wait(), 1)
            await asyncio.wait_for(second_finished.wait(), 1)
            self.assertEqual(accepted, [order[1]])
        finally:
            release_first.set()
            await asyncio.wait_for(run, 1)
        self.assertEqual(accepted, list(reversed(order)))

    async def test_fast_turn_frees_slot_while_slow_turn_is_still_running(self) -> None:
        scheduler = TurnScheduler(seed=14, concurrency=2)
        order = list(range(3))
        scheduler.rng(0, "citizen", 0, "_order_selected").shuffle(order)
        unblock_first = asyncio.Event()
        first_two_started = asyncio.Event()
        third_started = asyncio.Event()
        begun: list[int] = []

        async def act(identity: int) -> None:
            begun.append(identity)
            if len(begun) == 2:
                first_two_started.set()
            if identity == order[0]:
                await unblock_first.wait()
            elif identity == order[2]:
                third_started.set()

        run = asyncio.create_task(
            scheduler.run(step=0, phase="citizen", selected=range(3), ordinary=(), act=act)
        )
        try:
            await asyncio.wait_for(first_two_started.wait(), 1)
            await asyncio.wait_for(third_started.wait(), 1)
            self.assertFalse(run.done())
        finally:
            unblock_first.set()
            await asyncio.wait_for(run, 1)
        self.assertEqual(begun, order)

    async def test_rolling_admission_never_exceeds_cap(self) -> None:
        scheduler = TurnScheduler(seed=3, concurrency=2)
        expected = list(range(5))
        scheduler.rng(0, "citizen", 0, "_order_selected").shuffle(expected)
        begun: list[int] = []
        active = 0
        peak = 0
        unblock = asyncio.Event()
        first_two_started = asyncio.Event()

        async def act(identity: int) -> None:
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            begun.append(identity)
            if len(begun) == 2:
                first_two_started.set()
            await unblock.wait()
            active -= 1

        run = asyncio.create_task(
            scheduler.run(
                step=0, phase="citizen", selected=reversed(range(5)), ordinary=(), act=act
            )
        )
        await asyncio.wait_for(first_two_started.wait(), 1)
        self.assertEqual(begun, expected[:2])
        self.assertEqual(peak, 2)
        unblock.set()
        await asyncio.wait_for(run, 1)
        self.assertEqual(set(begun), set(range(5)))
        self.assertEqual(peak, 2)

    async def test_failure_cancels_and_joins_admitted_peers(self) -> None:
        scheduler = TurnScheduler(seed=2, concurrency=2)
        order = list(range(3))
        scheduler.rng(0, "citizen", 0, "_order_selected").shuffle(order)
        begun: list[int] = []
        cancelled: list[int] = []

        async def act(identity: int) -> None:
            begun.append(identity)
            if identity == order[1]:
                raise RuntimeError("original failure")
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(identity)
                raise

        with self.assertRaisesRegex(RuntimeError, "original failure"):
            await asyncio.wait_for(
                scheduler.run(step=0, phase="citizen", selected=range(3), ordinary=(), act=act), 1
            )
        self.assertEqual(begun, order[:2])
        self.assertEqual(cancelled, [order[0]])
        accepted: list[int] = []

        async def again(identity: int) -> None:
            accepted.append(identity)

        await scheduler.run(step=1, phase="citizen", selected=(7,), ordinary=(), act=again)
        self.assertEqual(accepted, [7])

    async def test_cancelling_run_joins_all_admitted_work(self) -> None:
        scheduler = TurnScheduler(seed=7, concurrency=2)
        both_started = asyncio.Event()
        begun: set[int] = set()
        cancelled: set[int] = set()

        async def act(identity: int) -> None:
            begun.add(identity)
            if len(begun) == 2:
                both_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.add(identity)
                raise

        run = asyncio.create_task(
            scheduler.run(step=0, phase="citizen", selected=(1, 2, 3), ordinary=(), act=act)
        )
        await asyncio.wait_for(both_started.wait(), 1)
        run.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(run, 1)
        self.assertEqual(cancelled, begun)
        self.assertEqual(len(begun), 2)

    async def test_simultaneous_failures_leave_no_unretrieved_exception(self) -> None:
        scheduler = TurnScheduler(seed=4, concurrency=2)
        ready = asyncio.Event()
        started = asyncio.Event()
        count = 0
        unhandled: list[dict[str, object]] = []
        loop = asyncio.get_running_loop()
        previous = loop.get_exception_handler()
        loop.set_exception_handler(lambda _loop, context: unhandled.append(context))

        async def act(identity: int) -> None:
            nonlocal count
            count += 1
            if count == 2:
                started.set()
            await ready.wait()
            raise RuntimeError(f"failed {identity}")

        try:
            run = asyncio.create_task(
                scheduler.run(step=0, phase="police", selected=(1, 2), ordinary=(), act=act)
            )
            await asyncio.wait_for(started.wait(), 1)
            ready.set()
            with self.assertRaisesRegex(RuntimeError, "failed"):
                await asyncio.wait_for(run, 1)
            del run
            gc.collect()
            await asyncio.sleep(0)
            self.assertEqual(unhandled, [])
        finally:
            loop.set_exception_handler(previous)

    def test_random_slots_are_replayable_and_independent(self) -> None:
        scheduler = TurnScheduler(seed=2**130 + 9)
        slot = (5, "citizen", 17, "movement")
        first = scheduler.rng(*slot)
        expected = scheduler.rng(*slot).integers(0, 2**63, size=6).tolist()
        first.integers(0, 2**63, size=2000)
        scheduler.rng(5, "citizen", 16, "movement").integers(0, 2**63, size=100)
        scheduler.rng(5, "citizen", 17, "release").integers(0, 2**63, size=100)
        self.assertEqual(scheduler.rng(*slot).integers(0, 2**63, size=6).tolist(), expected)
        for address in (
            (6, "citizen", 17, "movement"),
            (5, "police", 17, "movement"),
            (5, "citizen", 18, "movement"),
            (5, "citizen", 17, "release"),
            (5, "citizen", 17, "_order_selected"),
        ):
            self.assertNotEqual(
                scheduler.rng(*address).integers(0, 2**63, size=6).tolist(), expected
            )
