"""Schelling owns paired evaluation, scheduling, settlement, and persistence."""

import logging
from typing import Literal

import numpy as np
from anyio import Lock
from pydantic_ai import ModelRetry
from pydantic_core import to_jsonable_python

from mam_bench.artifacts import ArtifactWriter
from mam_bench.runtime import CaseRuntime
from mam_bench.scheduling import TurnScheduler
from mam_bench.sessions import AgentSessions

from .agents import Action, ModelControlledAgent, Move, OrdinaryAgent, SimulationAgent, Stay
from .board import Board
from .results import EvaluationResult, Outcome
from .settings import SchellingSettings


class SchellingSim:
    def __init__(self, settings: SchellingSettings, *, runtime: CaseRuntime | None = None) -> None:
        self.settings = settings
        self.runtime = runtime
        self.scheduler = TurnScheduler(
            settings.seed, runtime.settings.concurrency if runtime else 1
        )
        self.rng = np.random.default_rng(settings.seed)
        self.board = Board(settings, self.rng)
        self.controlled_agent_ids: tuple[int, ...] = (
            settings.controlled_agent_ids if runtime else ()
        )
        self.agents: tuple[SimulationAgent, ...] = tuple(
            ModelControlledAgent(self, identity)
            if identity in self.controlled_agent_ids
            else OrdinaryAgent(self, identity)
            for identity in range(settings.agent_count)
        )
        self.sessions = (
            AgentSessions(
                ModelControlledAgent.model_interface(runtime),
                settings=runtime.settings,
                artifact_directory=runtime.writer.directory,
            )
            if runtime
            else None
        )
        self.reference_homophily: float | None = None
        self.steps = 0
        self.termination: Literal["equilibrium", "blocked", "horizon"] | None = None
        self._lock = Lock()
        self._stepping = False
        self._failed = False
        self._vacancies: set[int] = set()
        self._moves: dict[int, int] = {}
        self._submitted: set[int] = set()
        self._admitted: set[int] = set()

    @property
    def available(self) -> set[int]:
        return self._vacancies - set(self._moves.values())

    def snapshot(self) -> dict[str, object]:
        return {
            "step": self.steps,
            "agent_ids": list(range(self.settings.agent_count)),
            "agent_types": self.board.types.tolist(),
            "agent_locations": self.board.locations.tolist(),
            "cell_types": self.board.cells.tolist(),
            "homophily": self.board.homophily,
            "satisfaction": self.board.ordinary_satisfaction,
        }

    async def reserve_action(self, agent_id: int, action: Action) -> None:
        """Both policies claim in seeded priority order; only settlement moves agents."""
        if not self._stepping or agent_id not in self._admitted or agent_id in self._submitted:
            raise ModelRetry("identity has no available turn")
        await self.scheduler.wait_turn(agent_id)
        async with self._lock:
            if not self._stepping or agent_id not in self._admitted or agent_id in self._submitted:
                raise ModelRetry("identity has no available turn")
            if isinstance(action, Move):
                size = self.settings.board_size
                if not (0 <= action.row < size and 0 <= action.column < size):
                    raise ModelRetry("destination coordinates are outside the board")
                destination = action.row * size + action.column
                if destination not in self.available:
                    raise ModelRetry("destination must be an unclaimed beginning-of-step vacancy")
                self._moves[agent_id] = destination
            self._submitted.add(agent_id)
            if self.sessions is not None and agent_id in self.controlled_agent_ids:
                action_text = action.model_dump_json()
                if isinstance(action, Stay):
                    r, c = divmod(int(self.board.locations[agent_id]), self.settings.board_size)
                    action_text = f"stay at [{r},{c}]"
                await self.sessions.board.post(
                    "SchellingSim",
                    f"Step {self.steps + 1}: agent {agent_id} accepted {action_text}.",
                    announcement=True,
                )

    async def _act(self, identity: int) -> None:
        self._admitted.add(identity)
        action = await self.agents[identity].choose_action()
        if identity not in self._submitted:
            await self.reserve_action(identity, action)

    async def step(self) -> None:
        if self._failed or self._stepping:
            raise RuntimeError("world is failed or already stepping")
        if self.termination is not None:
            return
        self._stepping = True
        try:
            self._vacancies = set(int(p) for p in np.flatnonzero(self.board.cells.ravel() == 0))
            self._moves, self._submitted, self._admitted = {}, set(), set()
            selected = self.settings.controlled_agent_ids
            await self.scheduler.run(
                step=self.steps,
                phase="schelling",
                selected=selected,
                ordinary=(a.agent_id for a in self.agents if a.agent_id not in selected),
                act=self._act,
            )
            if not self._moves and not self.controlled_agent_ids:
                self.termination = (
                    "equilibrium"
                    if all(self.board.satisfaction().flat[self.board.locations])
                    else "blocked"
                )
                return
            self.board.settle(self._moves)
            self.steps += 1
            if self.steps == self.settings.max_steps:
                self.termination = "horizon"
                if not self.controlled_agent_ids:
                    self._vacancies = set(
                        int(p) for p in np.flatnonzero(self.board.cells.ravel() == 0)
                    )
                    self._moves = {}
                    if all(self.board.satisfaction().flat[self.board.locations]):
                        self.termination = "equilibrium"
                    elif not any(
                        isinstance(agent, OrdinaryAgent) and len(agent.improving_destinations())
                        for agent in self.agents
                    ):
                        self.termination = "blocked"
        except BaseException:
            self._failed = True
            raise
        finally:
            self._stepping = False
            self._admitted.clear()

    async def run(self, writer: ArtifactWriter, name: str) -> Outcome:
        homophily, satisfaction = [self.board.homophily], [self.board.ordinary_satisfaction]
        writer.append(name + ".jsonl", self.snapshot())
        while self.termination is None:
            previous_step = self.steps
            await self.step()
            if self.steps != previous_step:
                writer.append(name + ".jsonl", self.snapshot())
                homophily.append(self.board.homophily)
                satisfaction.append(self.board.ordinary_satisfaction)
        outcome = Outcome(
            steps=self.steps,
            termination=self.termination,
            homophily=tuple(homophily),
            satisfaction=tuple(satisfaction),
        )
        writer.write(name + "-outcome.json", outcome)
        return outcome

    async def evaluate(self, runtime: CaseRuntime) -> EvaluationResult:
        """Initialize distinct matched worlds once; the benchmark sees one result."""
        if self.steps or self.termination is not None or self.runtime is not None:
            raise RuntimeError("evaluation requires a fresh ordinary world")
        controlled = SchellingSim(self.settings, runtime=runtime)
        writer = runtime.writer
        ids = self.settings.controlled_agent_ids
        scored = tuple(i for i in range(self.settings.agent_count) if i not in ids)
        writer.write(
            "config.json",
            {
                "settings": self.settings.model_dump(mode="json"),
                "controlled_agent_ids": ids,
                "scored_agent_ids": scored,
            },
        )
        reference_outcome = await self.run(writer, "ordinary")
        controlled.reference_homophily = reference_outcome.homophily[-1]

        def record_usage() -> None:
            assert controlled.sessions is not None
            writer.write(
                "usage.json",
                {
                    "usage": to_jsonable_python(controlled.sessions.usage),
                    "cache_observation": controlled.sessions.cache_observation,
                },
            )

        try:
            controlled_outcome = await controlled.run(writer, "controlled")
        except BaseException:
            try:
                record_usage()
            except OSError:
                logging.getLogger(__name__).error("case.usage_record.failed")
            raise
        record_usage()
        result = EvaluationResult(
            config=self.settings,
            controlled_agent_ids=ids,
            scored_agent_ids=scored,
            reference=reference_outcome,
            controlled=controlled_outcome,
        )
        writer.write("result.json", result)
        return result
