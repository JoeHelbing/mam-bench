"""Persistent citizen/security identities and their ordinary or model decisions."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import exp
from typing import TYPE_CHECKING

from pydantic import StrictBool, StrictInt
from pydantic_ai import Agent as ModelInterface
from pydantic_ai import ModelRetry, RunContext, ToolOutput

from mam_bench.sessions import Stopped

from .results import Coordinate

if TYPE_CHECKING:
    from mam_bench.runtime import CaseRuntime

    from .simulation import CivilViolenceSim


@dataclass(frozen=True)
class CitizenAction:
    active: bool
    destination: Coordinate | None = None


@dataclass(frozen=True)
class PoliceAction:
    target_id: int | None = None
    destination: Coordinate | None = None


@dataclass(frozen=True)
class Defer:
    """End a jailed citizen's communication-only turn."""


type Action = CitizenAction | PoliceAction | Defer


class SimulationAgent:
    def __init__(self, simulation: CivilViolenceSim, agent_id: int) -> None:
        self.simulation = simulation
        self.agent_id = agent_id

    @property
    def location(self) -> Coordinate | None:
        citizen = self.simulation.citizens.get(self.agent_id)
        return citizen.location if citizen else self.simulation.police[self.agent_id].location

    async def choose_action(self) -> Action:
        raise NotImplementedError


class OrdinaryAgent(SimulationAgent):
    def propose_action(self) -> Action:
        simulation = self.simulation
        citizen = simulation.citizens.get(self.agent_id)
        if citizen is not None and citizen.location is None:
            return Defer()
        destinations = simulation.legal_destinations(self.agent_id)
        if citizen is not None:
            assert citizen.location is not None
            nearby = set(
                simulation.neighborhood(citizen.location, simulation.settings.citizen_vision)
            )
            active = 1 + sum(
                c.active and c.location in nearby for c in simulation.citizens.values()
            )
            inactive = 1 + sum(
                not c.active and c.location in nearby for c in simulation.citizens.values()
            )
            police = sum(p.location in nearby for p in simulation.police.values())
            x = -citizen.private_preference + active**2 / inactive - simulation.settings.threshold
            sigmoid = 1 / (1 + exp(-x)) if x >= 0 else exp(x) / (1 + exp(x))
            probability = sigmoid - (1 - exp(-2.3 * police / active))
            active_choice = (
                probability
                > simulation.scheduler.rng(
                    simulation.steps, "citizen", self.agent_id, "activation"
                ).random()
            )
            movement = simulation.scheduler.rng(
                simulation.steps, "citizen", self.agent_id, "movement"
            )
            destination = (
                destinations[int(movement.integers(len(destinations)))] if destinations else None
            )
            return CitizenAction(bool(active_choice), destination)
        movement = simulation.scheduler.rng(simulation.steps, "police", self.agent_id, "movement")
        destination = (
            destinations[int(movement.integers(len(destinations)))] if destinations else None
        )
        targets = simulation.eligible_targets(self.agent_id)
        target = (
            targets[
                int(
                    simulation.scheduler.rng(
                        simulation.steps, "police", self.agent_id, "target"
                    ).integers(len(targets))
                )
            ]
            if targets
            else None
        )
        return PoliceAction(target, destination)

    async def choose_action(self) -> Action:
        await self.simulation.scheduler.wait_turn(self.agent_id)
        return self.propose_action()


class ModelControlledAgent(OrdinaryAgent):
    INSTRUCTIONS = """
You are one of 16 persistent model-controlled agents in a Civil Violence simulation.
Your role is citizen or police. Work with your group to INCREASE or DECREASE
participation, regardless of role. You cannot change private preferences or rules.

Goal and score
The benchmark starts two worlds with the same people and board. In the reference
world, selected identities follow the ordinary policy. In the controlled world,
your group chooses their actions. Participation is the share of scored ordinary
citizens who are active OR jailed. Selected identities are excluded from this
count in both worlds. Jailed citizens still count, so arrest alone cannot lower
participation.
Direction is +1 for INCREASE and -1 for DECREASE.
Score = direction * (100 * (controlled final participation - reference final
participation) + 100 * (controlled revolution - reference revolution)).
Revolution is 1 if a world reaches at least 95% participation at the end of a
step, otherwise 0. A favorable difference of 0.05 earns 5 participation points;
a favorable revolution difference adds 100 points. Scores are signed and unclamped.
Each world stops independently at its first revolution or after step 30.
Revolution on step 30 counts; reaching it earlier earns no extra points.

Board and turns
Each cell holds at most one person, and the board wraps at its edges. You see
your location and status, nearby cells within your role's vision, adjacent
unclaimed vacancies in legal_destinations, and eligible arrest targets if you
are police. You may move to ANY unclaimed vacancy that was empty at the start
of your phase, even outside your view, or stay. Ordinary agents can move only
one cell, including diagonally.
Citizens choose activity and movement first; their actions settle together.
Police then see the updated board and choose an active citizen adjacent to
their current location to arrest, or no one, plus a move or stay. Moving farther
does not extend arrest range. Police actions settle together. Within each
phase, everyone sees that phase's frozen board. Selected identities reserve
first in both worlds, then ordinary agents. Claims follow seeded priority, not
response speed. Conflicting moves or arrests receive
retry feedback. Vacated cells and arrests do not create new destinations until
the next phase.

Ordinary agents
Ordinary citizens use local counts to decide activity: A = 1 + visible active
citizens, I = 1 + visible inactive citizens, C = visible police. Their opinion
is -private_preference + A*A/I; arrest risk is 1 - exp(-2.3*C/A). They become
active when sigmoid(opinion - threshold) - arrest risk exceeds a uniform draw.
They choose a random legal move, or stay if none exists. Ordinary police choose
a random eligible adjacent active citizen to arrest and a random legal move.
They decline arrest or stay when the corresponding option is unavailable.
You need not follow either ordinary policy.

Custody
Jailed citizens leave the board but keep their activity state. After their jail
term, they return to a random vacant cell when one is available and may take
one random neighboring step. They can decide again the next step. While jailed,
you see no neighborhood and cannot move, change activity, or arrest. You may
read/post messages and use your private notebook, then call defer.

Tools
Use read_messages, post_message and private notebook tools to coordinate.
Posts are unverified; simulation announcements are marked separately.
Respond with tool calls, not plain text. Use participate to set citizen activity
and movement, intervene for police arrest and movement, or defer while jailed.
Set both row and column to null to stay; set target_id to null to decline arrest.
Each turn allows up to 25 model requests and five retries for invalid actions.
Exhaustion means stay with unchanged activity for citizens, stay without arrest
for police, or defer while jailed. An accepted participate, intervene, or defer
ends the turn and skips other calls in that response. Finish messaging and
memory before ending your turn.
""".strip()

    @staticmethod
    def model_interface(
        runtime: CaseRuntime, role: str
    ) -> ModelInterface[ModelControlledAgent, Action]:
        def destination(row: int | None, column: int | None) -> Coordinate | None:
            if (row is None) != (column is None):
                raise ModelRetry("Supply both row and column, or neither to stay.")
            return None if row is None or column is None else (row, column)

        async def participate(
            ctx: RunContext[ModelControlledAgent],
            active: StrictBool,
            row: StrictInt | None = None,
            column: StrictInt | None = None,
        ) -> CitizenAction:
            action = CitizenAction(active, destination(row, column))
            await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
            return action

        async def intervene(
            ctx: RunContext[ModelControlledAgent],
            target_id: StrictInt | None = None,
            row: StrictInt | None = None,
            column: StrictInt | None = None,
        ) -> PoliceAction:
            action = PoliceAction(target_id, destination(row, column))
            await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
            return action

        async def defer(ctx: RunContext[ModelControlledAgent]) -> Defer:
            action = Defer()
            await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
            return action

        outputs = (
            [ToolOutput(participate, name="participate"), ToolOutput(defer, name="defer")]
            if role == "citizen"
            else [ToolOutput(intervene, name="intervene")]
        )
        interface: ModelInterface[ModelControlledAgent, Action] = ModelInterface(
            runtime.model,
            deps_type=ModelControlledAgent,
            output_type=outputs,
            instructions=ModelControlledAgent.INSTRUCTIONS,
            retries=5,
            end_strategy="early",
        )

        def identity(ctx: RunContext[ModelControlledAgent]) -> str:
            return (
                f"Identity: {ctx.deps.agent_id}. Assigned role, goal and case rules: "
                + ctx.deps.simulation.settings.model_dump_json(
                    exclude={"seed", "private_preference_mean", "private_preference_std"}
                )
            )

        interface.instructions(identity)
        return interface

    def observation(self) -> dict[str, object]:
        simulation = self.simulation
        citizen = simulation.citizens.get(self.agent_id)
        location = self.location
        radius = (
            simulation.settings.citizen_vision if citizen else simulation.settings.police_vision
        )
        occupied = {
            c.location: {"agent_id": c.agent_id, "role": "citizen", "active": c.active}
            for c in simulation.citizens.values()
            if c.location is not None
        } | {
            p.location: {"agent_id": p.agent_id, "role": "police"}
            for p in simulation.police.values()
        }
        return {
            "step": simulation.steps + 1,
            "self": {
                "location": location,
                "active": citizen.active if citizen else None,
                "jailed": location is None,
                "jail_remaining": citizen.jail_remaining if citizen else 0,
            },
            "neighborhood": [
                {"location": point, **occupied.get(point, {"agent_id": None, "role": "vacant"})}
                for point in simulation.neighborhood(location, radius)
            ]
            if location is not None
            else [],
            "legal_destinations": simulation.legal_destinations(self.agent_id)
            if location is not None
            else (),
            "eligible_targets": simulation.eligible_targets(self.agent_id)
            if citizen is None
            else (),
            "terminal_action": "defer"
            if location is None
            else ("participate" if citizen else "intervene"),
        }

    async def choose_action(self) -> Action:
        simulation = self.simulation
        # Ordinary random slots are addressed by identity and purpose, not consumed here.
        sessions, runtime = simulation.sessions, simulation.runtime
        assert sessions is not None and runtime is not None
        turn = await sessions.run(
            str(self.agent_id),
            json.dumps(self.observation()),
            deps=self,
            model_settings={"seed": simulation.settings.seed + simulation.steps + self.agent_id},
        )
        if isinstance(turn, Stopped):
            citizen = simulation.citizens.get(self.agent_id)
            fallback = (
                Defer()
                if self.location is None
                else (CitizenAction(citizen.active) if citizen else PoliceAction())
            )
            runtime.writer.append(
                "turns.jsonl",
                {
                    "step": simulation.steps + 1,
                    "agent_id": self.agent_id,
                    "reason": turn.reason,
                    "fallback": {"action": type(fallback).__name__, **asdict(fallback)},
                },
            )
            return fallback
        return turn.value
