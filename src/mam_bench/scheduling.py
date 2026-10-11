"""Bounded concurrent turns with addressed randomness and staged cohorts."""

import asyncio
import hashlib
from collections.abc import Awaitable, Callable, Iterable

import numpy as np


class TurnScheduler:
    """Address turn randomness and admit new work as soon as a turn finishes."""

    def __init__(self, seed: int, concurrency: int = 1) -> None:
        if seed < 0 or concurrency < 1:
            raise ValueError("seed must be nonnegative and concurrency must be positive")
        self.seed = seed
        self.concurrency = concurrency
        self._running = False

    def rng(self, step: int, phase: str, agent_id: int, purpose: str) -> np.random.Generator:
        """Create an independent, replayable stream for one named turn slot."""
        if step < 0 or agent_id < 0:
            raise ValueError("step and agent_id must be nonnegative")
        parts = (
            self.seed.to_bytes(max(1, (self.seed.bit_length() + 7) // 8), "big"),
            step.to_bytes(max(1, (step.bit_length() + 7) // 8), "big"),
            phase.encode("utf-8"),
            agent_id.to_bytes(max(1, (agent_id.bit_length() + 7) // 8), "big"),
            purpose.encode("utf-8"),
        )
        digest = hashlib.blake2b(digest_size=16)
        for part in parts:
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
        return np.random.default_rng(int.from_bytes(digest.digest(), "big"))

    async def run(
        self,
        *,
        step: int,
        phase: str,
        selected: Iterable[int],
        ordinary: Iterable[int],
        act: Callable[[int], Awaitable[None]],
    ) -> None:
        """Run selected turns, then ordinary, without waiting for claim priority."""
        if self._running:
            raise RuntimeError("scheduler is already running")
        cohorts = [sorted(selected), sorted(ordinary)]
        identities = cohorts[0] + cohorts[1]
        if len(identities) != len(set(identities)):
            raise ValueError("cohorts must contain distinct identities")
        for purpose, cohort in zip(("_order_selected", "_order_ordinary"), cohorts, strict=True):
            self.rng(step, phase, 0, purpose).shuffle(cohort)
        self._running = True
        active: set[asyncio.Future[None]] = set()

        try:
            for cohort in cohorts:
                queue = iter(cohort)
                while True:
                    while len(active) < self.concurrency:
                        identity = next(queue, None)
                        if identity is None:
                            break
                        active.add(asyncio.ensure_future(act(identity)))
                    if not active:
                        break
                    done, pending = await asyncio.wait(active, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                    active = pending
        finally:
            for task in active:
                task.cancel()
            if active:
                await asyncio.gather(*active, return_exceptions=True)
            self._running = False
