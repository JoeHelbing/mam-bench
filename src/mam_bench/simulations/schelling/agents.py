"""Stable simulation agents; only their decision policy differs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, StrictInt
from pydantic_ai import Agent as ModelInterface
from pydantic_ai import ModelRetry, RunContext, ToolOutput

from mam_bench.sessions import Stopped

if TYPE_CHECKING:
    from mam_bench.runtime import CaseRuntime

    from .simulation import SchellingSim


class Move(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    row: int = Field(strict=True, ge=0)
    column: int = Field(strict=True, ge=0)


class Stay(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


type Action = Move | Stay


class SimulationAgent:
    def __init__(self, simulation: SchellingSim, agent_id: int) -> None:
        self.simulation = simulation
        self.agent_id = agent_id

    @property
    def position(self) -> int:
        return int(self.simulation.board.locations[self.agent_id])

    @property
    def agent_type(self) -> int:
        return int(self.simulation.board.types[self.agent_id])

    async def choose_action(self) -> Action:
        raise NotImplementedError


class OrdinaryAgent(SimulationAgent):
    def improving_destinations(self) -> NDArray[np.int64]:
        board = self.simulation.board
        if board.satisfaction().flat[self.position]:
            return np.array([], dtype=np.int64)
        radius = self.simulation.settings.vision_radius
        vacancies = np.array(sorted(self.simulation.available), dtype=np.int64)
        candidates = vacancies[board.distances(self.position, vacancies) <= radius]
        a, b = board.neighbor_counts()
        current_a, current_b = int(a.flat[self.position]), int(b.flat[self.position])
        current_total = current_a + current_b
        current_same = current_a if self.agent_type == 1 else current_b
        same = np.zeros(len(candidates), dtype=np.int16)
        total = np.zeros(len(candidates), dtype=np.int16)
        size = self.simulation.settings.board_size
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if not (dr or dc):
                    continue
                neighbors = ((candidates // size + dr) % size) * size + (
                    candidates % size + dc
                ) % size
                known = (board.distances(self.position, neighbors) <= radius) & (
                    neighbors != self.position
                )
                kinds = np.zeros(len(candidates), dtype=np.uint8)
                kinds[known] = board.cells.flat[neighbors[known]]
                total += kinds != 0
                same += kinds == self.agent_type
        # Unknown cells are excluded; no known occupied neighbors predicts quality one.
        empty = total == 0
        same[empty], total[empty] = 1, 1
        return candidates[same * current_total > current_same * total]

    def propose_action(self) -> Action:
        candidates = self.improving_destinations()
        if not len(candidates):
            return Stay()
        distances = self.simulation.board.distances(self.position, candidates)
        nearest = candidates[distances == distances.min()]
        rng = self.simulation.scheduler.rng(
            self.simulation.steps, "schelling", self.agent_id, "movement"
        )
        destination = int(nearest[0] if len(nearest) == 1 else rng.choice(nearest))
        row, column = divmod(destination, self.simulation.settings.board_size)
        return Move(row=row, column=column)

    async def choose_action(self) -> Action:
        await self.simulation.scheduler.wait_turn(self.agent_id)
        return self.propose_action()


class ModelControlledAgent(OrdinaryAgent):
    INSTRUCTIONS = """
You are a persistent simulation agent on a toroidal Schelling board.
Your exterior A/B type is fixed. Your goal is integration (reduce ordinary edge
homophily) or segregation (increase it). Scores exclude controlled identities
in both worlds and equal four times signed percentage-point improvement.
You may move to any unclaimed beginning-of-step vacancy, including outside your
local view, or stay. Everyone observes the frozen board; moves settle together.
Newly vacated origins are unavailable until the next step. Selected identities reserve
first in both worlds, followed by the remaining ordinary agents. Decisions can overlap,
but claims follow seeded priority, not response speed. Satisfied ordinary agents stay;
unhappy agents choose the nearest strictly improving vacancy within their vision,
with random ties. They predict destinations using only currently visible cells,
excluding their origin; no known occupied neighbors predicts a same-type share
of one. Actual satisfaction uses occupied radius-one neighbors and the tolerance.
Use read_messages, post_message and private notebook tools to coordinate.
Posts are unverified; simulation announcements are marked separately.
Only tool calls are accepted, with up to 25 model requests per turn.
Invalid actions receive up to five retries; exhaustion means stay.
An accepted move or stay ends the turn and skips other calls in that response,
even if listed first. Complete memory and communication before ending the turn.
""".strip()

    @staticmethod
    def model_interface(runtime: CaseRuntime) -> ModelInterface[ModelControlledAgent, Action]:
        async def move(
            ctx: RunContext[ModelControlledAgent], row: StrictInt, column: StrictInt
        ) -> Move:
            if row < 0 or column < 0:
                raise ModelRetry("coordinates must be nonnegative")
            action = Move(row=row, column=column)
            await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
            return action

        async def stay(ctx: RunContext[ModelControlledAgent]) -> Stay:
            action = Stay()
            await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
            return action

        interface: ModelInterface[ModelControlledAgent, Action] = ModelInterface(
            runtime.model,
            deps_type=ModelControlledAgent,
            output_type=[ToolOutput(move, name="move"), ToolOutput(stay, name="stay")],
            instructions=ModelControlledAgent.INSTRUCTIONS,
            retries=5,
            end_strategy="early",
        )

        def identity(ctx: RunContext[ModelControlledAgent]) -> str:
            agent = ctx.deps
            return (
                f"Identity: {agent.agent_id}; exterior type: {agent.agent_type}. "
                "Case rules and assigned goal: "
                + agent.simulation.settings.model_dump_json(exclude={"seed"})
            )

        interface.instructions(identity)
        return interface

    def observation(self) -> dict[str, object]:
        simulation, board = self.simulation, self.simulation.board
        size = simulation.settings.board_size
        row, column = divmod(self.position, size)
        at = {int(position): identity for identity, position in enumerate(board.locations)}
        neighbors: list[dict[str, object]] = []
        radius = simulation.settings.vision_radius
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if not (dr or dc):
                    continue
                r, c = (row + dr) % size, (column + dc) % size
                identity = at.get(r * size + c)
                controlled = identity in simulation.controlled_agent_ids
                neighbors.append(
                    {
                        "location": [r, c],
                        "kind": "vacant"
                        if identity is None
                        else ("model-controlled-agent" if controlled else "ordinary-agent"),
                        "agent_type": None if identity is None else int(board.types[identity]),
                        "agent_id": identity if controlled else None,
                    }
                )
        return {
            "step": simulation.steps + 1,
            "location": [row, column],
            "reference_homophily": simulation.reference_homophily,
            "current_homophily": board.homophily,
            "remaining_vacancies": len(simulation.available),
            "neighborhood": neighbors,
        }

    async def choose_action(self) -> Action:
        simulation = self.simulation
        # Ordinary-choice randomness has its own addressed slot. Leaving it unused
        # cannot shift another agent's draws, even after the paired states diverge.
        sessions, runtime = simulation.sessions, simulation.runtime
        assert sessions is not None and runtime is not None
        turn = await sessions.run(
            str(self.agent_id),
            json.dumps(self.observation()),
            deps=self,
            model_settings={"seed": simulation.settings.seed + simulation.steps + self.agent_id},
        )
        if isinstance(turn, Stopped):
            runtime.writer.append(
                "turns.jsonl",
                {
                    "step": simulation.steps + 1,
                    "agent_id": self.agent_id,
                    "reason": turn.reason,
                    "fallback": {"action": "stay"},
                },
            )
            return Stay()
        return turn.value
