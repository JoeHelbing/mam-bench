"""Binary Cascade dynamics, phase reservations, and retained ordinary trajectories."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, replace
from hashlib import blake2b
from math import exp, isfinite
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from anyio import Lock

from .models import (
    Action,
    Citizen,
    CitizenAction,
    CitizenSnapshot,
    Coordinate,
    Police,
    PoliceAction,
    PoliceSnapshot,
    Role,
    Snapshot,
    Trajectory,
)
from .settings import CivilViolenceSettings

if TYPE_CHECKING:
    from mam_bench.agent import AgentSessionRuntime
    from mam_bench.benchmark import PrimaryScore
    from mam_bench.model import ModelRuntime

    from .occupants import ModelControlledAgent


class CivilViolenceSim:
    """Own one world; ordinary and tactical policies share its phase transitions."""

    def __init__(
        self,
        settings: CivilViolenceSettings | None = None,
        role: Role = "citizen",
        *,
        citizens: tuple[Citizen, ...] | None = None,
        police: tuple[Police, ...] | None = None,
    ) -> None:
        self.settings = settings or CivilViolenceSettings()
        self.role: Role = role
        suffix = "citizens" if role == "citizen" else "police"
        self.simulation_id = f"civil-violence-{suffix}-v1"
        self.sessions: AgentSessionRuntime[ModelControlledAgent, Action] | None = None
        self.ordinary_trajectory: Trajectory | None = None
        self.controlled_agent_ids: tuple[int, ...] = ()
        self._reservation_lock = Lock()
        self._admitted: set[int] = set()
        self._stepping = False
        self._failed = False
        self.cycle = 0
        size = self.settings.board_size
        positions = self._draw(0, "initial-positions").permutation(size**2)
        if citizens is None:
            citizens = tuple(
                Citizen(
                    i,
                    divmod(int(positions[i]), size),
                    float(
                        self._draw(i, "private-preference").normal(
                            self.settings.private_preference_mean,
                            self.settings.private_preference_std,
                        )
                    ),
                )
                for i in range(self.settings.citizen_count)
            )
        if police is None:
            police = tuple(
                Police(i, divmod(int(positions[i]), size))
                for i in range(
                    self.settings.citizen_count,
                    self.settings.citizen_count + self.settings.police_count,
                )
            )
        if not citizens:
            raise ValueError("at least one scored citizen is required")
        ids = [c.agent_id for c in citizens] + [p.agent_id for p in police]
        locations = [c.location for c in citizens if c.location is not None] + [
            p.location for p in police
        ]
        if len(set(ids)) != len(ids) or any(i < 0 for i in ids):
            raise ValueError("agent identities must be distinct nonnegative integers")
        if len(ids) > size**2 or len(set(locations)) != len(locations):
            raise ValueError("initial population exceeds single-occupancy capacity")
        if any(not (0 <= r < size and 0 <= c < size) for r, c in locations):
            raise ValueError("initial locations must be inside the board")
        if any(c.jail_remaining < 0 or not isfinite(c.private_preference) for c in citizens):
            raise ValueError("citizens need finite preferences and nonnegative jail terms")
        if any(c.location is not None and c.jail_remaining for c in citizens):
            raise ValueError("imprisoned citizens must be off-grid")
        self.citizens = {c.agent_id: replace(c) for c in citizens}
        self.police = {p.agent_id: replace(p) for p in police}
        self._phase: str | None = None
        self._phase_locations: dict[int, Coordinate] = {}
        self._vacancies: set[Coordinate] = set()
        self._moves: dict[int, Coordinate] = {}
        self._activities: dict[int, bool] = {}
        self._targets: dict[int, int] = {}
        self._custody: tuple[int, ...] = ()
        self._submitted: set[int] = set()
        self.snapshots: list[Snapshot] = [self.snapshot()]

    @property
    def finished(self) -> bool:
        return self.cycle >= self.settings.max_transitions

    def _draw(self, agent_id: int, purpose: str) -> np.random.Generator:
        """Stable streams: policy substitution never consumes a neighbor's randomness."""
        tag = int.from_bytes(blake2b(purpose.encode("ascii"), digest_size=8).digest(), "little")
        return np.random.default_rng(
            np.random.SeedSequence([self.settings.seed_id, self.cycle, agent_id, tag])
        )

    def snapshot(self) -> Snapshot:
        return Snapshot(
            self.cycle,
            tuple(
                CitizenSnapshot(
                    c.agent_id, c.location, c.private_preference, c.active, c.jail_remaining
                )
                for c in self.citizens.values()
            ),
            tuple(PoliceSnapshot(p.agent_id, p.location) for p in self.police.values()),
        )

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
        self._custody = tuple(c.agent_id for c in self.citizens.values() if c.location is None)
        return tuple(c.agent_id for c in self.citizens.values() if c.location is not None)

    def legal_destinations(self, agent_id: int) -> tuple[Coordinate, ...]:
        adjacent = self.neighborhood(self._phase_locations[agent_id], 1)
        claimed = set(self._moves.values())
        return tuple(p for p in adjacent if p in self._vacancies and p not in claimed)

    def _ordinary_citizen_action(
        self, agent_id: int, activation_draw: float | None = None
    ) -> CitizenAction:
        citizen = self.citizens[agent_id]
        nearby = set(
            self.neighborhood(self._phase_locations[agent_id], self.settings.citizen_vision)
        )
        active = 1 + sum(c.active and c.location in nearby for c in self.citizens.values())
        inactive = 1 + sum(not c.active and c.location in nearby for c in self.citizens.values())
        police = sum(p.location in nearby for p in self.police.values())
        opinion = -citizen.private_preference + active**2 / inactive
        x = opinion - self.settings.threshold
        sigmoid = 1 / (1 + exp(-x)) if x >= 0 else exp(x) / (1 + exp(x))
        probability = sigmoid - (1 - exp(-2.3 * police / active))
        draw = (
            float(self._draw(agent_id, "activation").random())
            if activation_draw is None
            else activation_draw
        )
        destinations = self.legal_destinations(agent_id)
        destination = None
        if destinations:
            destination = destinations[
                int(self._draw(agent_id, "citizen-move").integers(len(destinations)))
            ]
        return CitizenAction(probability > draw, destination)

    def _validate_move(self, agent_id: int, destination: Coordinate | None) -> None:
        if agent_id in self._submitted:
            raise ValueError("agent already submitted an action")
        if destination is not None and destination not in self.legal_destinations(agent_id):
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

        # Release after movement settlement so neither placement can consume
        # the other's reserved cell. Released citizens keep cached activity.
        occupied = {c.location for c in self.citizens.values() if c.location is not None}
        occupied.update(p.location for p in self.police.values())
        empty = {
            (r, c) for r in range(self.settings.board_size) for c in range(self.settings.board_size)
        } - occupied
        for agent_id in self._custody:
            citizen = self.citizens[agent_id]
            if citizen.jail_remaining > 0:
                citizen.jail_remaining -= 1
            elif empty:
                choices = sorted(empty)
                citizen.location = choices[
                    int(self._draw(agent_id, "release").integers(len(choices)))
                ]
                # Cascade releases into a random vacancy, then makes the usual
                # one-cell random move; neither choice consumes a reserved cell.
                hops = [p for p in self.neighborhood(citizen.location, 1) if p in empty]
                if hops:
                    citizen.location = hops[
                        int(self._draw(agent_id, "release-move").integers(len(hops)))
                    ]
                empty.remove(citizen.location)

    def _begin_police_phase(self) -> tuple[int, ...]:
        self._begin_phase("police")
        return tuple(self.police)

    def eligible_targets(self, agent_id: int) -> tuple[int, ...]:
        adjacent = self.neighborhood(self._phase_locations[agent_id], 1)
        claimed = set(self._targets.values())
        return tuple(
            c.agent_id
            for c in self.citizens.values()
            if c.active and c.location in adjacent and c.agent_id not in claimed
        )

    def _ordinary_police_action(self, agent_id: int) -> PoliceAction:
        destinations = self.legal_destinations(agent_id)
        destination = None
        if destinations:
            destination = destinations[
                int(self._draw(agent_id, "police-move").integers(len(destinations)))
            ]
        targets = self.eligible_targets(agent_id)
        target = None
        if targets:
            target = targets[int(self._draw(agent_id, "arrest-target").integers(len(targets)))]
        return PoliceAction(target, destination)

    def _reserve_police_action(
        self, agent_id: int, target_id: int | None = None, destination: Coordinate | None = None
    ) -> None:
        if self._phase != "police" or agent_id not in self.police:
            raise ValueError("police officer is not eligible in this phase")
        self._validate_move(agent_id, destination)
        if target_id is not None and target_id not in self.eligible_targets(agent_id):
            raise ValueError("target must be an unclaimed adjacent active citizen")
        self._submitted.add(agent_id)
        if target_id is not None:
            self._targets[agent_id] = target_id
        if destination is not None:
            self._moves[agent_id] = destination

    def _settle_police(self) -> Snapshot:
        for target_id in self._targets.values():
            citizen = self.citizens[target_id]
            citizen.location = None
            citizen.jail_remaining = int(
                self._draw(target_id, "jail-term").integers(self.settings.max_jail_term + 1)
            )
        for agent_id, destination in self._moves.items():
            self.police[agent_id].location = destination
        self.cycle += 1
        self._phase = None
        state = self.snapshot()
        self.snapshots.append(state)
        return state

    def step(self, activation_draws: Mapping[int, float] | None = None) -> Snapshot:
        if self._failed:
            raise RuntimeError("failed trials cannot be resumed")
        if self.sessions is not None:
            raise RuntimeError("controlled trials use astep")
        if self.finished:
            return self.snapshot()
        for agent_id in self._begin_citizen_phase():
            draw = None if activation_draws is None else activation_draws.get(agent_id)
            action = self._ordinary_citizen_action(agent_id, draw)
            self._reserve_citizen_action(agent_id, action.active, action.destination)
        self._settle_citizens()
        for agent_id in self._begin_police_phase():
            action = self._ordinary_police_action(agent_id)
            self._reserve_police_action(agent_id, action.target_id, action.destination)
        return self._settle_police()

    def run_reference(self) -> Trajectory:
        while not self.finished:
            self.step()
        return Trajectory(self.settings, self.role, tuple(self.snapshots))

    simulation_version = "cascade-binary-v1"

    def _start_model(self, runtime: ModelRuntime, directory: Path | None = None) -> None:
        from mam_bench.agent import AgentSessionRuntime

        from .occupants import ModelControlledAgent

        initial = self.snapshots[0]
        citizens = tuple(
            Citizen(c.agent_id, c.location, c.private_preference, c.active, c.jail_remaining)
            for c in initial.citizens
        )
        police = tuple(Police(p.agent_id, p.location) for p in initial.police)
        pool = tuple(self.citizens) if self.role == "citizen" else tuple(self.police)
        count = self.settings.controlled_agent_count
        if count > len(pool) or (self.role == "citizen" and count >= len(self.citizens)):
            raise ValueError(
                "controlled count exceeds role population or leaves no scored citizens"
            )
        # A new world of the same class provides the comparator for every evaluation.
        reference = CivilViolenceSim(self.settings, self.role, citizens=citizens, police=police)
        ordinary = reference.run_reference()
        self.__init__(self.settings, self.role, citizens=citizens, police=police)
        self.controlled_agent_ids = tuple(
            sorted(
                int(i)
                for i in self._draw(0, "controlled-identities").choice(pool, count, replace=False)
            )
        )
        self.ordinary_trajectory = ordinary
        self.sessions = AgentSessionRuntime(
            ModelControlledAgent.model_definition(runtime, self.role),
            settings=runtime.info.agent_settings,
            artifact_directory=directory,
        )

    async def reserve_action(self, agent_id: int, action: Action) -> None:
        """Validate and claim the whole tactical action before any physical mutation."""
        from pydantic_ai import ModelRetry

        async with self._reservation_lock:
            if agent_id not in self._admitted or agent_id in self._submitted:
                raise ModelRetry("identity has no available turn in this phase")
            try:
                if isinstance(action, CitizenAction):
                    self._reserve_citizen_action(agent_id, action.active, action.destination)
                else:
                    self._reserve_police_action(agent_id, action.target_id, action.destination)
            except ValueError as error:
                raise ModelRetry(str(error)) from None
            self._submitted.add(agent_id)
            assert self.sessions is not None
            await self.sessions.board.post(
                "CivilViolenceSim",
                f"Cycle {self.cycle + 1}: {self.role} {agent_id} reserved "
                + json.dumps(asdict(action)),
                announcement=True,
            )

    async def _model_turns(self, eligible: tuple[int, ...]) -> None:
        from mam_bench.agent import run_rolling

        from .occupants import ModelControlledAgent

        self._admitted = set(eligible).intersection(self.controlled_agent_ids)
        self._submitted = set()
        if not self._admitted:
            return
        sessions = self.sessions
        assert sessions is not None
        order = tuple(
            int(i) for i in self._draw(0, "model-admission").permutation(sorted(self._admitted))
        )

        async def act(agent_id: int) -> None:
            await ModelControlledAgent(self, agent_id).take_turn(sessions)
            if agent_id not in self._submitted:
                # Native exhausted-turn fallback: no new tactical state or resource claim.
                if self.role == "citizen":
                    self._reserve_citizen_action(agent_id, self.citizens[agent_id].active)
                else:
                    self._reserve_police_action(agent_id)

        await run_rolling(order, act, concurrency=sessions.settings.concurrency)

    async def astep(self) -> Snapshot:
        """Execute one complete controlled cycle through the ordinary phase authority."""
        if self._failed:
            raise RuntimeError("failed trials cannot be resumed")
        if self._stepping:
            raise RuntimeError("a cycle is already in progress")
        if self.finished:
            return self.snapshot()
        if self.sessions is None:
            return self.step()
        self._stepping = True
        try:
            citizens = self._begin_citizen_phase()
            if self.role == "citizen":
                await self._model_turns(citizens)
            for agent_id in citizens:
                if agent_id not in self.controlled_agent_ids:
                    action = self._ordinary_citizen_action(agent_id)
                    self._reserve_citizen_action(agent_id, action.active, action.destination)
            self._settle_citizens()
            police = self._begin_police_phase()
            if self.role == "police":
                await self._model_turns(police)
            for agent_id in police:
                if agent_id not in self.controlled_agent_ids:
                    police_action = self._ordinary_police_action(agent_id)
                    self._reserve_police_action(
                        agent_id, police_action.target_id, police_action.destination
                    )
            return self._settle_police()
        except BaseException:
            self._failed = True
            raise
        finally:
            self._admitted = set()
            self._stepping = False

    async def run(self, runtime: ModelRuntime, output_directory: Path) -> PrimaryScore:
        """Run a fresh matched reference and tactical trial, then publish its score."""
        from mam_bench.benchmark import (
            AgentInfrastructureFailure,
            PairFailure,
            PairInfrastructureFailure,
            PrimaryScore,
        )

        if self._stepping:
            raise RuntimeError("a cycle is already in progress")
        if output_directory.exists():
            raise FileExistsError(f"result directory already exists: {output_directory}")
        try:
            self._start_model(runtime, output_directory)
            while not self.finished:
                await self.astep()
            excluded = self.controlled_agent_ids if self.role == "citizen" else ()
            scored = tuple(i for i in self.citizens if i not in excluded)
            ordinary = self.ordinary_trajectory
            assert ordinary is not None
            controlled = Trajectory(self.settings, self.role, tuple(self.snapshots))
            reference_mean = ordinary.mean_activity(excluded)
            controlled_mean = controlled.mean_activity(excluded)
            lift = controlled_mean - reference_mean
            if self.role == "police":
                lift = -lift
            ordinary.write(output_directory / "ordinary.npz")
            controlled.write(output_directory / "model-controlled.npz")
            summary = {
                "simulation_id": self.simulation_id,
                "simulation_version": self.simulation_version,
                "source_revision": "bd9598d0c813e1b72ee0d62abb7610d452b26313",
                "role": self.role,
                "config": self.settings.model_dump(mode="json"),
                "runtime": runtime.info.model_dump(mode="json"),
                "controlled_agent_ids": self.controlled_agent_ids,
                "scored_agent_ids": scored,
                "ordinary": {"mean_activity": reference_mean},
                "model_controlled": {"mean_activity": controlled_mean},
                "activity_lift": lift,
                "cycles_completed": self.cycle,
                "rng": "PCG64 keyed by seed, cycle, identity, blake2b purpose",
            }
            temporary = output_directory / "result.json.tmp"
            temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
            temporary.replace(output_directory / "result.json")
        except (AgentInfrastructureFailure, OSError) as error:
            self._failed = True
            kind = error.kind if isinstance(error, AgentInfrastructureFailure) else "artifact_write"
            raise PairInfrastructureFailure(
                PairFailure(
                    simulation_id=self.simulation_id,
                    simulation_version=self.simulation_version,
                    model_id=runtime.info.model_id,
                    provider=runtime.info.provider,
                    model=runtime.info.model,
                    kind=kind,
                )
            ) from None
        return PrimaryScore(name="activity_lift", value=lift, unit="ordinary active fraction")
