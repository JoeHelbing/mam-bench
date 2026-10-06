"""Persistent citizen/security identities and their ordinary or model decisions."""

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
    target_location: Coordinate | None = None
    destination: Coordinate | None = None


@dataclass(frozen=True)
class Defer:
    """End a jailed citizen's communication-only turn."""


type Action = CitizenAction | PoliceAction | Defer


_SHARED_INSTRUCTIONS = """
You are one of 16 persistent model-controlled agents in a Civil Violence simulation.
Work with your group to INCREASE or DECREASE participation. You cannot change
private preferences or the rules.

Goal and score
The benchmark starts two worlds with the same people and board. In the reference
world, selected identities follow the ordinary policy. In the controlled world,
your group chooses their actions. Participation is the share of scored ordinary
citizens who are active OR jailed; selected identities are excluded in both
worlds. Jailed citizens still count, so arrest alone cannot lower participation.
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
your location, nearby cells within your vision, adjacent unclaimed vacancies
in legal_destinations, and your role's other observations. You may move to ANY
unclaimed vacancy that was empty at the start of your phase, even outside your
view, or stay. Ordinary agents can move only one cell, including diagonally.
Citizens act and settle first; police then see the updated board and act and
settle. Within each phase everyone sees that phase's frozen board. Selected
identities reserve first in both worlds, then ordinary agents. Claims follow
seeded priority, not response speed. Conflicting claims receive retry feedback.
Vacated cells and arrests do not create destinations until the next phase.

Ordinary citizen behavior
Ordinary citizens use local counts: A = 1 + visible active citizens,
I = 1 + visible inactive citizens, C = visible police. Their opinion is
-private_preference + A*A/I; arrest risk is 1 - exp(-2.3*C/A). Their activation
value is sigmoid(opinion - threshold) - arrest risk. They draw a random number
between 0 and 1 and become active if the value exceeds the draw. They choose
a random legal move, or stay if none exists.

Custody
Jailed citizens leave the board but still count as participating. At the start
of a round, a citizen whose jail term has ended returns to a random vacancy if
one is available. They return inactive, then choose activity and movement on
that round's citizen turn.

Use read_messages, post_message and private notebook tools to coordinate.
Posts are unverified; simulation announcements are marked separately.
Respond with tool calls, not plain text. Read messages and use memory before
your action. End a free turn with one act(...) call containing your role choice
and movement destination (null to stay). Movement is validated first, then the
whole action is reserved only if valid. An invalid action
returns retry feedback without reserving any part. A valid act ends the turn;
other calls in the same response are skipped. Each turn allows up to 25 model
requests and five retries for invalid actions.
""".strip()


class SimulationAgent:
    def __init__(self, simulation: CivilViolenceSim, agent_id: int) -> None:
        self.simulation = simulation
        self.agent_id = agent_id

    @property
    def location(self) -> Coordinate | None:
        raise NotImplementedError

    def visible_neighborhood(self, radius: int) -> list[dict[str, object]]:
        location = self.location
        if location is None:
            return []
        simulation = self.simulation
        occupied = {
            c.location: {"agent_id": c.agent_id, "role": "citizen", "active": c.active}
            for c in simulation.citizens.values()
            if c.location is not None
        } | {
            p.location: {"agent_id": p.agent_id, "role": "police"}
            for p in simulation.police.values()
        }
        return [
            {"location": point, **occupied.get(point, {"agent_id": None, "role": "vacant"})}
            for point in simulation.neighborhood(location, radius)
        ]

    async def choose_action(self) -> Action:
        raise NotImplementedError


class OrdinaryAgent(SimulationAgent):
    def propose_action(self) -> Action:
        raise NotImplementedError

    async def choose_action(self) -> Action:
        await self.simulation.scheduler.wait_turn(self.agent_id)
        return self.propose_action()


class CitizenAgent(SimulationAgent):
    @property
    def location(self) -> Coordinate | None:
        return self.simulation.citizens[self.agent_id].location

    def propose_action(self) -> CitizenAction | Defer:
        simulation = self.simulation
        citizen = simulation.citizens[self.agent_id]
        if citizen.location is None:
            return Defer()
        destinations = simulation.legal_destinations(self.agent_id)
        nearby = set(simulation.neighborhood(citizen.location, simulation.settings.vision_radius))
        active = 1 + sum(c.active and c.location in nearby for c in simulation.citizens.values())
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
        movement = simulation.scheduler.rng(simulation.steps, "citizen", self.agent_id, "movement")
        destination = (
            destinations[int(movement.integers(len(destinations)))] if destinations else None
        )
        return CitizenAction(bool(active_choice), destination)

    def observation(self) -> dict[str, object]:
        simulation = self.simulation
        citizen = simulation.citizens[self.agent_id]
        location = citizen.location
        return {
            "step": simulation.steps + 1,
            "self": {
                "location": location,
                "active": citizen.active,
                "jailed": location is None,
                "jail_remaining": citizen.jail_remaining,
            },
            "neighborhood": self.visible_neighborhood(simulation.settings.vision_radius),
            "legal_destinations": simulation.legal_destinations(self.agent_id)
            if location is not None
            else (),
        }


class PoliceAgent(SimulationAgent):
    @property
    def location(self) -> Coordinate:
        return self.simulation.police[self.agent_id].location

    def propose_action(self) -> PoliceAction:
        simulation = self.simulation
        destinations = simulation.legal_destinations(self.agent_id)
        movement = simulation.scheduler.rng(simulation.steps, "police", self.agent_id, "movement")
        destination = (
            destinations[int(movement.integers(len(destinations)))] if destinations else None
        )
        targets = simulation.eligible_targets(self.agent_id, location=destination)
        target_id = (
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
        target_location = simulation.citizens[target_id].location if target_id is not None else None
        return PoliceAction(target_location, destination)

    def observation(self) -> dict[str, object]:
        simulation = self.simulation
        return {
            "step": simulation.steps + 1,
            "self": {
                "location": self.location,
                "active": None,
                "jailed": False,
                "jail_remaining": 0,
            },
            "neighborhood": self.visible_neighborhood(simulation.settings.vision_radius),
            "legal_destinations": simulation.legal_destinations(self.agent_id),
            "eligible_target_locations_if_staying": tuple(
                simulation.citizens[target].location
                for target in simulation.eligible_targets(self.agent_id)
            ),
        }


class OrdinaryCitizenAgent(CitizenAgent, OrdinaryAgent):
    pass


class OrdinaryPoliceAgent(PoliceAgent, OrdinaryAgent):
    pass


class ModelControlledAgent(SimulationAgent):
    def observation(self) -> dict[str, object]:
        raise NotImplementedError

    def fallback_action(self) -> Action:
        raise NotImplementedError

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
            fallback = self.fallback_action()
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


async def _act_citizen(
    ctx: RunContext[ModelControlledAgent],
    active: StrictBool,
    destination: tuple[StrictInt, StrictInt] | None,
) -> CitizenAction:
    """Choose activity and destination [row, column] together; null means stay."""
    if ctx.deps.location is None:
        raise ModelRetry("jailed citizens must defer")
    action = CitizenAction(active, destination)
    await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
    return action


async def _act_police(
    ctx: RunContext[ModelControlledAgent],
    destination: tuple[StrictInt, StrictInt] | None,
    target_location: tuple[StrictInt, StrictInt] | None,
) -> PoliceAction:
    """Move first, then arrest at target square [row, column]; null declines either."""
    action = PoliceAction(target_location, destination)
    await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
    return action


async def _defer(ctx: RunContext[ModelControlledAgent]) -> Defer:
    """End a jailed citizen's communication-only turn."""
    action = Defer()
    await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
    return action


def _identity(ctx: RunContext[ModelControlledAgent]) -> str:
    return (
        f"Identity: {ctx.deps.agent_id}. Assigned role, goal and case rules: "
        + ctx.deps.simulation.settings.model_dump_json(
            exclude={"seed", "private_preference_mean", "private_preference_std"}
        )
    )


def _model_interface(
    runtime: CaseRuntime, instructions: str, outputs: list[ToolOutput[Action]]
) -> ModelInterface[ModelControlledAgent, Action]:
    interface: ModelInterface[ModelControlledAgent, Action] = ModelInterface(
        runtime.model,
        deps_type=ModelControlledAgent,
        output_type=outputs,
        instructions=instructions,
        retries=5,
        end_strategy="early",
    )
    interface.instructions(_identity)
    return interface


class ModelCitizenAgent(CitizenAgent, ModelControlledAgent):
    INSTRUCTIONS = (
        "You are a citizen.\n"
        + _SHARED_INSTRUCTIONS
        + """

Citizen decisions
Call act(active=true/false, destination=[row, column] or null to stay).
Activity and movement settle together. Police see the settled board and may
arrest active citizens adjacent to their destination after moving (or their
current location if staying). You need not follow the ordinary citizen policy.

Your jailed turn
While jailed, you see no neighborhood and cannot move or change activity.
You may read/post messages and use your private notebook, then call defer().
An accepted defer ends the turn.
Exhaustion means stay with unchanged activity, or defer while jailed.
"""
    )

    @staticmethod
    def model_interface(runtime: CaseRuntime) -> ModelInterface[ModelControlledAgent, Action]:
        interface = _model_interface(
            runtime,
            ModelCitizenAgent.INSTRUCTIONS,
            [ToolOutput(_act_citizen, name="act"), ToolOutput(_defer, name="defer")],
        )
        return interface

    def fallback_action(self) -> CitizenAction | Defer:
        citizen = self.simulation.citizens[self.agent_id]
        return Defer() if citizen.location is None else CitizenAction(citizen.active)


class ModelPoliceAgent(PoliceAgent, ModelControlledAgent):
    INSTRUCTIONS = (
        "You are a police officer.\n"
        + _SHARED_INSTRUCTIONS
        + """

Police decisions
After citizens settle, call act(destination=[row, column],
target_location=[row, column]) to move and arrest at a square adjacent to your
destination. Pass destination=null to stay and arrest from your current
location; pass target_location=null to decline arrest. You may move to an unseen phase-start
vacancy and target an active citizen there using information already shared
with you, but moving gives you no new observation before the action settles.
The target must be an unclaimed active citizen adjacent to your destination
(or current location if staying). eligible_target_locations_if_staying lists
only the targets available from your current square; it is not exhaustive if
you move. Ordinary police choose a random legal move before choosing a random
eligible target at the resulting location. You need not follow their policy.
Exhaustion means stay without arrest.
"""
    )

    @staticmethod
    def model_interface(runtime: CaseRuntime) -> ModelInterface[ModelControlledAgent, Action]:
        interface = _model_interface(
            runtime, ModelPoliceAgent.INSTRUCTIONS, [ToolOutput(_act_police, name="act")]
        )
        return interface

    def fallback_action(self) -> PoliceAction:
        return PoliceAction()
