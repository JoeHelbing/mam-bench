"""Binary Cascade phase rules, simulation-owned scheduling and paired evaluation."""

import logging
from dataclasses import asdict
from typing import Literal

import numpy as np
from anyio import Lock
from pydantic_ai import ModelRetry
from pydantic_core import to_jsonable_python

from mam_bench.artifacts import ArtifactWriter
from mam_bench.runtime import CaseRuntime
from mam_bench.scheduling import TurnScheduler
from mam_bench.sessions import AgentSessions

from .agents import (
    Action,
    CitizenAction,
    Defer,
    ModelCitizenAgent,
    ModelPoliceAgent,
    OrdinaryCitizenAgent,
    OrdinaryPoliceAgent,
    SimulationAgent,
)
from .results import Citizen, Coordinate, EvaluationResult, Outcome, Police
from .settings import CivilViolenceSettings


class CivilViolenceSim:
    def __init__(
        self, settings: CivilViolenceSettings, *, runtime: CaseRuntime | None = None
    ) -> None:
        self.settings, self.runtime = settings, runtime
        self.rng = np.random.default_rng(settings.seed)
        self.scheduler = TurnScheduler(
            settings.seed, concurrency=runtime.settings.concurrency if runtime else 1
        )
        positions = self.rng.permutation(settings.board_size**2)
        self.citizens = {
            i: Citizen(
                i,
                divmod(int(positions[i]), settings.board_size),
                float(
                    self.rng.normal(
                        settings.private_preference_mean, settings.private_preference_std
                    )
                ),
            )
            for i in range(settings.citizen_count)
        }
        self.police = {
            i: Police(i, divmod(int(positions[i]), settings.board_size))
            for i in range(settings.citizen_count, settings.citizen_count + settings.police_count)
        }
        pool = tuple(self.citizens if settings.controlled_role == "citizen" else self.police)
        # Both worlds select the same replacement identities, consuming identical initial draws.
        self.replacement_ids = tuple(
            sorted(
                int(i)
                for i in self.rng.choice(pool, settings.controlled_agent_count, replace=False)
            )
        )
        self.scored_agent_ids = tuple(i for i in self.citizens if i not in self.replacement_ids)
        self.controlled_agent_ids: tuple[int, ...] = self.replacement_ids if runtime else ()
        self.agents: dict[int, SimulationAgent] = {}
        for i in self.citizens:
            agent_type = (
                ModelCitizenAgent if i in self.controlled_agent_ids else OrdinaryCitizenAgent
            )
            self.agents[i] = agent_type(self, i)
        for i in self.police:
            agent_type = ModelPoliceAgent if i in self.controlled_agent_ids else OrdinaryPoliceAgent
            self.agents[i] = agent_type(self, i)
        model_agent = (
            ModelCitizenAgent if settings.controlled_role == "citizen" else ModelPoliceAgent
        )
        self.sessions = (
            AgentSessions(
                model_agent.model_interface(runtime),
                settings=runtime.settings,
                artifact_directory=runtime.writer.directory,
            )
            if runtime
            else None
        )
        self.steps = 0
        self.termination: Literal["revolution", "horizon"] | None = None
        self._lock = Lock()
        self._stepping = False
        self._failed = False
        self._admitted: set[int] = set()
        self._phase: str | None = None
        self._phase_locations: dict[int, Coordinate] = {}
        self._vacancies: set[Coordinate] = set()
        self._moves: dict[int, Coordinate] = {}
        self._activities: dict[int, bool] = {}
        self._targets: dict[int, int] = {}
        self._custody: tuple[int, ...] = ()
        self._submitted: set[int] = set()

    @property
    def participating_count(self) -> int:
        return sum(
            self.citizens[i].active or self.citizens[i].location is None
            for i in self.scored_agent_ids
        )

    @property
    def participation(self) -> float:
        return self.participating_count / len(self.scored_agent_ids)

    def snapshot(self) -> dict[str, object]:
        return {
            "step": self.steps,
            "citizens": [asdict(c) for c in self.citizens.values()],
            "police": [asdict(p) for p in self.police.values()],
            "participation": self.participation,
        }

    def neighborhood(self, location: Coordinate, radius: int) -> tuple[Coordinate, ...]:
        size = self.settings.board_size
        return tuple(
            sorted(
                {
                    ((location[0] + dr) % size, (location[1] + dc) % size)
                    for dr in range(-min(radius, size), min(radius, size) + 1)
                    for dc in range(-min(radius, size), min(radius, size) + 1)
                }
                - {location}
            )
        )

    def _begin_phase(self, phase: str) -> None:
        self._phase = phase
        self._phase_locations = {
            c.agent_id: c.location for c in self.citizens.values() if c.location is not None
        } | {p.agent_id: p.location for p in self.police.values()}
        occupied = set(self._phase_locations.values())
        self._vacancies = {
            (r, c)
            for r in range(self.settings.board_size)
            for c in range(self.settings.board_size)
            if (r, c) not in occupied
        }
        self._moves = {}
        self._activities = {}
        self._targets = {}
        self._submitted = set()

    def _begin_citizen_phase(self) -> tuple[int, ...]:
        self._begin_phase("citizen")
        for citizen in self.citizens.values():
            if citizen.location is not None:
                continue
            if citizen.jail_remaining > 0:
                citizen.jail_remaining -= 1
            elif self._vacancies:
                choices = sorted(self._vacancies)
                release = self.scheduler.rng(self.steps, "citizen", citizen.agent_id, "release")
                citizen.location = choices[int(release.integers(len(choices)))]
                citizen.active = False
                self._phase_locations[citizen.agent_id] = citizen.location
                self._vacancies.remove(citizen.location)
        self._custody = tuple(c.agent_id for c in self.citizens.values() if c.location is None)
        return tuple(self.citizens)

    def legal_destinations(self, agent_id: int) -> tuple[Coordinate, ...]:
        adjacent = self.neighborhood(self._phase_locations[agent_id], 1)
        claimed = set(self._moves.values())
        return tuple(p for p in adjacent if p in self._vacancies and p not in claimed)

    def _validate_move(self, agent_id: int, destination: Coordinate | None) -> None:
        if agent_id in self._submitted:
            raise ValueError("agent already submitted an action")
        if destination is None:
            return
        if agent_id in self.controlled_agent_ids:
            if destination not in self._vacancies or destination in self._moves.values():
                raise ValueError("destination must be an unclaimed phase-start vacancy")
        elif destination not in self.legal_destinations(agent_id):
            raise ValueError("destination must be an unclaimed adjacent phase-start vacancy")

    def _reserve_citizen_action(
        self, agent_id: int, active: bool, destination: Coordinate | None = None
    ) -> None:
        if self._phase != "citizen" or agent_id not in self.citizens or agent_id in self._custody:
            raise ValueError("citizen is not eligible in this phase")
        self._validate_move(agent_id, destination)
        self._submitted.add(agent_id)
        self._activities[agent_id] = active
        if destination is not None:
            self._moves[agent_id] = destination

    def _settle_citizens(self) -> None:
        for agent_id, active in self._activities.items():
            self.citizens[agent_id].active = active
        for agent_id, destination in self._moves.items():
            self.citizens[agent_id].location = destination

    def _begin_police_phase(self) -> tuple[int, ...]:
        self._begin_phase("police")
        return tuple(self.police)

    def eligible_targets(
        self, agent_id: int, location: Coordinate | None = None
    ) -> tuple[int, ...]:
        adjacent = self.neighborhood(
            location if location is not None else self._phase_locations[agent_id], 1
        )
        claimed = set(self._targets.values())
        return tuple(
            c.agent_id
            for c in self.citizens.values()
            if c.active and c.location in adjacent and c.agent_id not in claimed
        )

    def _reserve_police_action(
        self,
        agent_id: int,
        target_location: Coordinate | None = None,
        destination: Coordinate | None = None,
    ) -> None:
        if self._phase != "police" or agent_id not in self.police:
            raise ValueError("police officer is not eligible in this phase")
        self._validate_move(agent_id, destination)
        location = destination if destination is not None else self._phase_locations[agent_id]
        target_id = next(
            (
                target
                for target in self.eligible_targets(agent_id, location=location)
                if self.citizens[target].location == target_location
            ),
            None,
        )
        if target_location is not None and target_id is None:
            raise ValueError("target square must hold an unclaimed adjacent active citizen")
        self._submitted.add(agent_id)
        if target_id is not None:
            self._targets[agent_id] = target_id
        if destination is not None:
            self._moves[agent_id] = destination

    def _settle_police(self) -> None:
        for target_id in self._targets.values():
            citizen = self.citizens[target_id]
            citizen.location = None
            jail = self.scheduler.rng(self.steps, "police", target_id, "jail")
            citizen.jail_remaining = int(jail.integers(self.settings.max_jail_term + 1))
        for agent_id, destination in self._moves.items():
            self.police[agent_id].location = destination
        self.steps += 1
        self._phase = None

    async def reserve_action(self, agent_id: int, action: Action) -> None:
        """Validate a whole action before claiming any resource, for either policy."""
        if agent_id not in self._admitted or agent_id in self._submitted:
            raise ModelRetry("identity has no available turn in this phase")
        await self.scheduler.wait_turn(agent_id)
        async with self._lock:
            if agent_id not in self._admitted or agent_id in self._submitted:
                raise ModelRetry("identity has no available turn in this phase")
            try:
                if isinstance(action, Defer):
                    if self._phase != "citizen" or agent_id not in self._custody:
                        raise ValueError("only a jailed citizen may defer")
                elif isinstance(action, CitizenAction):
                    self._reserve_citizen_action(agent_id, action.active, action.destination)
                else:
                    self._reserve_police_action(
                        agent_id, action.target_location, action.destination
                    )
            except ValueError as error:
                raise ModelRetry(str(error)) from None
            self._submitted.add(agent_id)
            if self.sessions is not None and agent_id in self.controlled_agent_ids:
                await self.sessions.board.post(
                    "CivilViolenceSim",
                    f"Step {self.steps + 1}: agent {agent_id} accepted "
                    f"{type(action).__name__} {asdict(action)}.",
                    announcement=True,
                )

    async def _act(self, agent_id: int) -> None:
        self._admitted.add(agent_id)
        action = await self.agents[agent_id].choose_action()
        if agent_id not in self._submitted:
            await self.reserve_action(agent_id, action)

    async def _phase_turns(self, eligible: tuple[int, ...], phase: str) -> None:
        selected = set(self.replacement_ids)
        await self.scheduler.run(
            step=self.steps,
            phase=phase,
            selected=(i for i in eligible if i in selected),
            ordinary=(i for i in eligible if i not in selected),
            act=self._act,
        )

    async def step(self) -> None:
        if self._failed or self._stepping:
            raise RuntimeError("world is failed or already stepping")
        if self.termination is not None:
            return
        self._stepping = True
        try:
            await self._phase_turns(self._begin_citizen_phase(), "citizen")
            self._settle_citizens()
            await self._phase_turns(self._begin_police_phase(), "police")
            self._settle_police()
            # Check the completed step, including the horizon, for revolution first.
            if self.participating_count * 100 >= len(self.scored_agent_ids) * 95:
                self.termination = "revolution"
            elif self.steps == self.settings.max_steps:
                self.termination = "horizon"
        except BaseException:
            self._failed = True
            raise
        finally:
            self._admitted.clear()
            self._stepping = False

    async def run(self, writer: ArtifactWriter, name: str) -> Outcome:
        participation = [self.participation]
        writer.append(name + ".jsonl", self.snapshot())
        while self.termination is None:
            await self.step()
            writer.append(name + ".jsonl", self.snapshot())
            participation.append(self.participation)
        outcome = Outcome(
            steps=self.steps, termination=self.termination, participation=tuple(participation)
        )
        writer.write(name + "-outcome.json", outcome)
        return outcome

    async def evaluate(self, runtime: CaseRuntime) -> EvaluationResult:
        if self.steps or self.termination is not None or self.runtime is not None:
            raise RuntimeError("evaluation requires a fresh ordinary world")
        controlled = CivilViolenceSim(self.settings, runtime=runtime)
        runtime.writer.write(
            "config.json",
            {
                "settings": self.settings.model_dump(mode="json"),
                "controlled_agent_ids": self.replacement_ids,
                "scored_agent_ids": self.scored_agent_ids,
            },
        )
        reference_outcome = await self.run(runtime.writer, "ordinary")

        def record_usage() -> None:
            assert controlled.sessions is not None
            runtime.writer.write(
                "usage.json",
                {
                    "usage": to_jsonable_python(controlled.sessions.usage),
                    "cache_observation": controlled.sessions.cache_observation,
                },
            )

        try:
            controlled_outcome = await controlled.run(runtime.writer, "controlled")
        except BaseException:
            try:
                record_usage()
            except OSError:
                logging.getLogger(__name__).error("case.usage_record.failed")
            raise
        record_usage()
        result = EvaluationResult(
            config=self.settings,
            controlled_agent_ids=self.replacement_ids,
            scored_agent_ids=self.scored_agent_ids,
            reference=reference_outcome,
            controlled=controlled_outcome,
        )
        runtime.writer.write("result.json", result)
        return result
