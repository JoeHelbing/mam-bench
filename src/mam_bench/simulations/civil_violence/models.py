"""Detached scientific records and role-specific tactical actions."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from .settings import CivilViolenceSettings

type Coordinate = tuple[int, int]
type Role = Literal["citizen", "police"]


@dataclass
class Citizen:
    agent_id: int
    location: Coordinate | None
    private_preference: float
    active: bool = False
    jail_remaining: int = 0


@dataclass
class Police:
    agent_id: int
    location: Coordinate


@dataclass(frozen=True)
class CitizenAction:
    active: bool
    destination: Coordinate | None = None


@dataclass(frozen=True)
class PoliceAction:
    target_id: int | None = None
    destination: Coordinate | None = None


type Action = CitizenAction | PoliceAction


@dataclass(frozen=True)
class CitizenSnapshot:
    agent_id: int
    location: Coordinate | None
    private_preference: float
    active: bool
    jail_remaining: int


@dataclass(frozen=True)
class PoliceSnapshot:
    agent_id: int
    location: Coordinate


@dataclass(frozen=True)
class Snapshot:
    cycle: int
    citizens: tuple[CitizenSnapshot, ...]
    police: tuple[PoliceSnapshot, ...]

    def activity(self, excluded_ids: tuple[int, ...] = ()) -> float:
        scored = [c for c in self.citizens if c.agent_id not in excluded_ids]
        if not scored:
            raise ValueError("at least one scored citizen is required")
        return sum(c.active and c.location is not None for c in scored) / len(scored)


@dataclass(frozen=True)
class Trajectory:
    settings: CivilViolenceSettings
    role: Role
    snapshots: tuple[Snapshot, ...]

    @property
    def trajectory_length(self) -> int:
        return len(self.snapshots)

    def mean_activity(self, excluded_ids: tuple[int, ...] = ()) -> float:
        if len(self.snapshots) < 2:
            raise ValueError("mean activity requires at least one completed cycle")
        return sum(s.activity(excluded_ids) for s in self.snapshots[1:]) / (len(self.snapshots) - 1)

    def write(self, path: Path) -> None:
        """Retain every identity, trait, custody state and location for score replay."""
        states = self.snapshots
        initial = states[0]
        citizen_ids = tuple(c.agent_id for c in initial.citizens)
        police_ids = tuple(p.agent_id for p in initial.police)
        np.savez_compressed(
            path,
            settings_json=np.array(self.settings.model_dump_json()),
            role=np.array(self.role),
            cycles=np.array([s.cycle for s in states]),
            agent_ids=np.array(citizen_ids + police_ids),
            roles=np.array([0] * len(citizen_ids) + [1] * len(police_ids)),
            private_preferences=np.array(
                [c.private_preference for c in initial.citizens] + [0.0] * len(police_ids)
            ),
            agent_locations=np.array(
                [
                    [c.location if c.location is not None else (-1, -1) for c in s.citizens]
                    + [p.location for p in s.police]
                    for s in states
                ]
            ),
            active=np.array(
                [[c.active for c in s.citizens] + [False] * len(police_ids) for s in states]
            ),
            jailed=np.array(
                [
                    [c.location is None for c in s.citizens] + [False] * len(police_ids)
                    for s in states
                ]
            ),
            jail_remaining=np.array(
                [[c.jail_remaining for c in s.citizens] + [0] * len(police_ids) for s in states]
            ),
        )
