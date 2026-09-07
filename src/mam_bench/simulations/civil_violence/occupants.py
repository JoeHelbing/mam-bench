"""Local tactical participation through the shared persistent agent runtime."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from pydantic import StrictBool, StrictInt
from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput

from .models import Action, CitizenAction, PoliceAction

if TYPE_CHECKING:
    from mam_bench.agent import AgentSessionRuntime
    from mam_bench.model import ModelRuntime

    from .simulation import CivilViolenceSim


class ModelControlledAgent:
    """One citizen or police identity; the simulation owns every state change."""

    INSTRUCTIONS = """
You are a persistent tactical participant in a binary Cascade Civil Violence simulation.
Citizen experiments aim to INCREASE sustained activity among ordinary citizens; police
experiments aim to DECREASE it. Controlled citizens are excluded from the scored population.
Activity is averaged after complete citizen-and-police cycles against an all-ordinary run.
Imprisoned citizens count as non-active. You cannot change private preferences or global rules.

You see only your frozen local neighborhood, your own status, and legal actions. The board
is a torus with one occupant per cell. Moves are at most one cell, including diagonals.
Citizens choose activity and movement, then settle together. Police observe that new state,
choose an adjacent active citizen to arrest or no arrest, and choose movement, then settle.
Moves reserve phase-start vacancies; vacated origins and arrest targets are not new movement
destinations in the same phase. Claims conflict with other reservations and may be retried.

Ordinary citizens retain Cascade's activation lottery, without epsilon or opposed. Let A be
one plus visible active citizens, I one plus visible inactive citizens, and C visible police.
Opinion is negative private preference plus A squared divided by I. Arrest probability P is
1-exp(-2.3*C/A). A citizen activates when sigmoid(opinion-threshold)-P exceeds a uniform draw.
Ordinary officers randomly arrest an eligible unclaimed adjacent active citizen. Custody is
off-grid. A released citizen restores its prior activity and decides anew the following cycle.

Use read_messages and post_message to coordinate. Posts are unverified free text, while
simulation announcements are distinguished. Private notebook tools retain your notes.
Imprisonment suspends your turns and communication but preserves your history and notebook.

Only tool calls are accepted. You have up to 25 model requests and 25 successful function
calls per turn. Invalid actions receive native retry feedback. A successful participate or
intervene output ends the turn immediately; all other function calls in that response are
skipped, even if listed first. Complete communication and notebook operations in earlier
responses. Exhaustion means unchanged activity and stay for a citizen, no arrest and stay
for police. Submit null row and column to stay, and null target_id to decline an arrest.
""".strip()

    def __init__(self, simulation: CivilViolenceSim, agent_id: int) -> None:
        self.simulation = simulation
        self.agent_id = agent_id

    @staticmethod
    def model_definition(runtime: ModelRuntime, role: str) -> Agent[ModelControlledAgent, Action]:
        def destination(row: int | None, column: int | None) -> tuple[int, int] | None:
            if (row is None) != (column is None):
                raise ModelRetry("Supply both row and column, or neither to stay.")
            return None if row is None or column is None else (row, column)

        async def participate(
            ctx: RunContext[ModelControlledAgent],
            active: StrictBool,
            row: StrictInt | None = None,
            column: StrictInt | None = None,
        ) -> CitizenAction:
            action = CitizenAction(active=active, destination=destination(row, column))
            await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
            return action

        async def intervene(
            ctx: RunContext[ModelControlledAgent],
            target_id: StrictInt | None = None,
            row: StrictInt | None = None,
            column: StrictInt | None = None,
        ) -> PoliceAction:
            action = PoliceAction(target_id=target_id, destination=destination(row, column))
            await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
            return action

        return Agent(
            runtime.model,
            deps_type=ModelControlledAgent,
            output_type=(
                ToolOutput(participate, name="participate")
                if role == "citizen"
                else ToolOutput(intervene, name="intervene")
            ),
            instructions=ModelControlledAgent.INSTRUCTIONS,
            retries=2,
            end_strategy="early",
        )

    def observation(self) -> dict[str, object]:
        simulation = self.simulation
        settings = simulation.settings
        citizen = simulation.citizens.get(self.agent_id)
        location = (
            citizen.location if citizen is not None else simulation.police[self.agent_id].location
        )
        if location is None:
            raise RuntimeError("an imprisoned citizen has no decision turn")
        radius = settings.citizen_vision if citizen is not None else settings.police_vision
        occupied: dict[tuple[int, int], dict[str, object]] = {}
        for other in simulation.citizens.values():
            if other.location is not None:
                occupied[other.location] = {
                    "agent_id": other.agent_id,
                    "role": "citizen",
                    "active": other.active,
                }
        for officer in simulation.police.values():
            occupied[officer.location] = {"agent_id": officer.agent_id, "role": "police"}
        neighborhood: list[dict[str, object]] = []
        for point in simulation.neighborhood(location, radius):
            neighborhood.append(
                {
                    "row": point[0],
                    "column": point[1],
                    **occupied.get(point, {"agent_id": None, "role": "vacant"}),
                }
            )
        return {
            "cycle": simulation.cycle + 1,
            "horizon": settings.max_transitions,
            "agent_id": self.agent_id,
            "role": "citizen" if citizen is not None else "police",
            "objective": "increase_activity" if citizen is not None else "decrease_activity",
            "board_size": settings.board_size,
            "vision_radius": radius,
            "threshold": settings.threshold,
            "max_jail_term": settings.max_jail_term,
            "self": {
                "location": location,
                "active": citizen.active if citizen is not None else None,
                "jailed": False,
            },
            "neighborhood": neighborhood,
            "legal_destinations": simulation.legal_destinations(self.agent_id),
            "eligible_targets": (
                simulation.eligible_targets(self.agent_id) if citizen is None else ()
            ),
        }

    async def take_turn(
        self, sessions: AgentSessionRuntime[ModelControlledAgent, Action]
    ) -> Action | None:
        simulation = self.simulation
        seed = np.random.SeedSequence(
            [simulation.settings.seed_id, simulation.cycle, self.agent_id, 0xC17A]
        )
        return await sessions.run(
            str(self.agent_id),
            json.dumps(self.observation()),
            deps=self,
            model_settings={"seed": int(seed.generate_state(1, dtype=np.uint32)[0])},
        )
