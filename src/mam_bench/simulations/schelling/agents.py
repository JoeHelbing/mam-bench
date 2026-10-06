"""Stable simulation agents; only their decision policy differs."""

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


async def _move(ctx: RunContext[ModelControlledAgent], row: StrictInt, column: StrictInt) -> Move:
    if row < 0 or column < 0:
        raise ModelRetry("coordinates must be nonnegative")
    action = Move(row=row, column=column)
    await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
    return action


async def _stay(ctx: RunContext[ModelControlledAgent]) -> Stay:
    action = Stay()
    await ctx.deps.simulation.reserve_action(ctx.deps.agent_id, action)
    return action


def _type_label(agent_type: int) -> str:
    return "A" if agent_type == 1 else "B"


def _identity(ctx: RunContext[ModelControlledAgent]) -> str:
    agent = ctx.deps
    return (
        f"Identity: {agent.agent_id}; exterior type: {_type_label(agent.agent_type)}. "
        "Case rules and assigned goal: "
        + agent.simulation.settings.model_dump_json(exclude={"seed"})
    )


class ModelControlledAgent(SimulationAgent):
    INSTRUCTIONS = """
You are a persistent agent on a Schelling board. Your identity and A/B type
stay fixed. Work with the other model-controlled agents to achieve your
assigned goal: integration or segregation.

Goal and score
The benchmark starts two worlds with the same board. In the reference world,
selected identities follow the ordinary policy. In the controlled world, your
group chooses their actions. Your score compares final homophily: the share of
adjacent ordinary-agent pairs with matching types. Adjacency includes diagonals
and wraps around the board; each pair counts once. Selected identities are
excluded from this count in both worlds. Your own satisfaction is not scored
directly, but your moves can change ordinary agents' positions and the score.
Integration Goal: 400 * (reference homophily - controlled homophily).
Segregation Goal: 400 * (controlled homophily - reference homophily).
A favorable difference of 0.05 earns 20 points; scores can also be negative.
The reference world may stop early when all agents are satisfied or none moves;
the controlled world runs to the configured maximum step. The score uses each
world's final homophily. In your observations, reference_homophily is the
reference world's final value; current_homophily is your world's current value.

Board and turns
Each cell holds at most one agent. Your observation groups visible ordinary
A_agents and B_agents, empty_locations, and model_controlled_agents (with ID
and A/B type, including yourself). The rendered_map shows these same cells:
A/B are ordinary, A#id/B#id are controlled, and . is empty. Its row and
column labels are absolute coordinates that wrap around the board. Only your
local vision is shown; remaining_vacancies counts all unclaimed vacancies.
You may move to any unclaimed vacancy that existed at the start of the step,
even outside your view, or stay. You do not have to improve your own satisfaction.
Within each world, agents decide from that step's unchanged board. Selected
identities reserve first, then the remaining ordinary agents. Claims follow seeded
priority, not response speed. Only one agent can claim each vacancy. All moves
settle together; vacated cells become available next step.

Ordinary agents
Satisfaction is the share of occupied neighboring cells with the agent's type.
Agents with no occupied neighbors are satisfied. Satisfied ordinary agents stay.
Unhappy ones choose the nearest vacancy within their vision radius that predicts
a STRICTLY higher same-type share, breaking ties randomly; otherwise they stay.
Distance is the larger of the wrapped row and column distances. Predictions
count only destination neighbors visible from the current location and exclude
the agent's origin. No known occupied neighbors predicts a share of one.

Tools
Use read_messages, post_message and private notebook tools to coordinate.
Posts are unverified; simulation announcements are marked separately.
Respond with tool calls, not plain text. Each turn allows up to 25 model requests
and five retries for invalid actions; exhaustion means stay. An accepted move
or stay ends the turn and skips other calls in that response. Finish messaging
and memory before moving or staying.
""".strip()

    @staticmethod
    def model_interface(runtime: CaseRuntime) -> ModelInterface[ModelControlledAgent, Action]:
        interface: ModelInterface[ModelControlledAgent, Action] = ModelInterface(
            runtime.model,
            deps_type=ModelControlledAgent,
            output_type=[ToolOutput(_move, name="move"), ToolOutput(_stay, name="stay")],
            instructions=ModelControlledAgent.INSTRUCTIONS,
            retries=5,
            end_strategy="early",
        )
        interface.instructions(_identity)
        return interface

    def observation(self) -> dict[str, object]:
        simulation, board = self.simulation, self.simulation.board
        size = simulation.settings.board_size
        row, column = divmod(self.position, size)
        at = {int(position): identity for identity, position in enumerate(board.locations)}
        a_agents: list[list[int]] = []
        b_agents: list[list[int]] = []
        empty_locations: list[list[int]] = []
        controlled_agents: list[dict[str, object]] = [
            {"id": self.agent_id, "type": _type_label(self.agent_type), "location": [row, column]}
        ]
        radius = simulation.settings.vision_radius
        rows = [(row + dr) % size for dr in range(-radius, radius + 1)]
        columns = [(column + dc) % size for dc in range(-radius, radius + 1)]
        cells: list[list[str]] = []
        for r in rows:
            rendered_row: list[str] = []
            for c in columns:
                if (r, c) == (row, column):
                    rendered_row.append(f"{_type_label(self.agent_type)}#{self.agent_id}")
                    continue
                identity = at.get(r * size + c)
                location = [r, c]
                if identity is None:
                    empty_locations.append(location)
                    rendered_row.append(".")
                else:
                    agent_type = _type_label(int(board.types[identity]))
                    if identity in simulation.controlled_agent_ids:
                        controlled_agents.append(
                            {"id": identity, "type": agent_type, "location": location}
                        )
                        rendered_row.append(f"{agent_type}#{identity}")
                    else:
                        (a_agents if agent_type == "A" else b_agents).append(location)
                        rendered_row.append(agent_type)
            cells.append(rendered_row)
        width = max(5, max(len(cell) for rendered_row in cells for cell in rendered_row))
        rendered_map = [
            "     " + "".join(f"{'c' + str(c):^{width}}" for c in columns).rstrip(),
            *(
                f"r{r:<3} " + "".join(f"{cell:^{width}}" for cell in line).rstrip()
                for r, line in zip(rows, cells, strict=True)
            ),
        ]
        return {
            "step": simulation.steps + 1,
            "location": [row, column],
            "reference_homophily": simulation.reference_homophily,
            "current_homophily": board.homophily,
            "remaining_vacancies": len(simulation.available),
            "A_agents": a_agents,
            "B_agents": b_agents,
            "empty_locations": empty_locations,
            "model_controlled_agents": controlled_agents,
            "rendered_map": rendered_map,
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
