"""Schelling occupants: shared spatial identity and role-specific decisions.

Positions always read settled simulation storage. Decisions never move an occupant.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from pydantic import StrictInt
from pydantic_ai import RunContext

from .models import Move, Stay
from .profile import MASTER_SEED_WORDS, Rational
from .reference import (
    BoardCoordinate,
    CellGrid,
    LocationArray,
    candidate_mask,
    choose_nearest_index,
    distances,
    evaluate_satisfaction,
    toroidal_chebyshev_distance,
)

if TYPE_CHECKING:
    from pydantic_ai import Agent as PydanticAgent

    from mam_bench.agent import AgentSessionRuntime
    from mam_bench.model import ModelRuntime

    from .models import Action, Observation
    from .simulation import SchellingSim


class Agent:
    """Stable occupant identity and immutable exterior type, not a model definition."""

    def __init__(self, simulation: SchellingSim, agent_id: int) -> None:
        self._simulation = simulation
        self._agent_id = agent_id
        self._agent_type = simulation.agent_type(agent_id)

    @property
    def agent_id(self) -> int:
        return self._agent_id

    @property
    def agent_type(self) -> int:
        return self._agent_type

    @property
    def position(self) -> int:
        return self._simulation.agent_position(self.agent_id)

    @property
    def coordinate(self) -> BoardCoordinate:
        return BoardCoordinate(*divmod(self.position, self._simulation.cell.board_size))

    def distance_to(self, destination: int) -> int:
        return toroidal_chebyshev_distance(
            self.position, destination, size=self._simulation.cell.board_size
        )


class OrdinaryAgent(Agent):
    """An exact-tolerance occupant requesting the nearest satisfactory vacancy."""

    @property
    def tolerance(self) -> Rational:
        return self._simulation.cell.tolerance

    def is_satisfied(self, frozen_board: CellGrid) -> bool:
        return bool(evaluate_satisfaction(frozen_board, self.tolerance).flat[self.position])

    def choose_destination_index(
        self,
        cell_types: CellGrid,
        vacancies: LocationArray,
        available: NDArray[np.bool_],
        type_a_neighbors: NDArray[np.int16],
        type_b_neighbors: NDArray[np.int16],
        tie_rng: np.random.Generator,
    ) -> int | None:
        """Request an index in shared vacancy storage; do not reserve or settle it."""
        eligible = available & candidate_mask(
            cell_types,
            vacancies,
            self.position,
            self.agent_type,
            self.tolerance,
            type_a_neighbors,
            type_b_neighbors,
        )
        eligible_indices = np.flatnonzero(eligible)
        if len(eligible_indices) == 0:
            return None
        eligible_destinations = vacancies[eligible_indices]
        distance = distances(self.position, eligible_destinations, cell_types.shape[0])
        nearest_indices = eligible_indices[distance == distance.min()]
        selected_index = choose_nearest_index(nearest_indices, tie_rng)
        return selected_index


class ModelControlledAgent(Agent):
    """Schelling embodiment and tools backed by a persistent generic session."""

    INSTRUCTIONS = """
You are one of 16 persistent ModelControlledAgents on a 20x20 toroidal Schelling board.
You control only your own token. Your exterior A/B type is fixed. You have no tolerance
or satisfaction. You may request any beginning-of-round vacancy, even outside your
observed neighborhood. Your objective is integration (reduce masked Ordinary Edge
Homophily) or segregation (increase it); scoring excludes model-controlled identities.

You receive a frozen radius-1 neighborhood and permitted scalar context each turn,
not a global board, satisfaction labels, or suggested moves. Use read_messages and
post_message to communicate. Agent posts are unverified and may be deceptive;
simulation announcements are marked separately. Use the ordinary private notebook
tools for persistent notes. Posts are limited to 4,000 characters.

Only tool calls are accepted. You have up to 25 successful function-tool calls and
25 model requests per turn. Invalid calls receive retry feedback. Exhaustion means
stay. You may call multiple tools in one response. An accepted move or stay ends
the turn immediately and takes precedence over ALL other function calls in that
response: those calls are skipped, even if listed first. Batch notebook/message
operations separately, then call move or stay when ready to end the turn.
""".strip()

    @staticmethod
    def model_definition(runtime: ModelRuntime) -> PydanticAgent[ModelControlledAgent, Action]:
        from pydantic_ai import Agent as PydanticAgent
        from pydantic_ai import ToolOutput

        async def move(
            ctx: RunContext[ModelControlledAgent], row: StrictInt, column: StrictInt
        ) -> Move:
            from pydantic import ValidationError
            from pydantic_ai import ModelRetry

            try:
                action = Move(row=row, column=column)
            except ValidationError as error:
                raise ModelRetry(str(error)) from None
            await ctx.deps._simulation.reserve_move(ctx.deps.agent_id, action)
            return action

        async def stay() -> Stay:
            return Stay()

        return PydanticAgent(
            runtime.model,
            deps_type=ModelControlledAgent,
            output_type=[ToolOutput(move, name="move"), ToolOutput(stay, name="stay")],
            instructions=ModelControlledAgent.INSTRUCTIONS,
            retries=2,
            end_strategy="early",
        )

    def observe(self) -> Observation:
        from .models import NeighborKind, NeighborObservation, Observation

        simulation = self._simulation
        state = simulation.snapshot()
        size = simulation.cell.board_size
        agent_at = np.full(size * size, -1, dtype=np.int64)
        agent_at[state.agent_locations] = np.arange(len(state.agent_locations))
        row, column = divmod(self.position, size)
        neighbors: list[NeighborObservation] = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == dc == 0:
                    continue
                r, c = (row + dr) % size, (column + dc) % size
                identity = int(agent_at[r * size + c])
                controlled = identity in simulation.controlled_agent_ids
                neighbors.append(
                    NeighborObservation(
                        location=BoardCoordinate(r, c),
                        kind=NeighborKind.VACANT
                        if identity < 0
                        else (
                            NeighborKind.MODEL_CONTROLLED_AGENT
                            if controlled
                            else NeighborKind.ORDINARY_AGENT
                        ),
                        agent_type=None if identity < 0 else int(state.agent_types[identity]),
                        agent_id=identity if controlled else None,
                    )
                )
        reference_homophily = simulation.reference_homophily
        if reference_homophily is None:
            raise RuntimeError("a model-controlled agent requires an ordinary reference")
        return Observation(
            self.agent_id,
            self.agent_type,
            self.coordinate,
            simulation.rounds_completed + 1,
            simulation.max_transitions,
            simulation.objective,
            reference_homophily,
            simulation.metrics().ordinary_homophily,
            simulation.remaining_vacancies,
            tuple(neighbors),
        )

    def prompt(self, observation: Observation) -> str:
        lines: list[str] = []
        for neighbor in observation.neighborhood:
            kind = neighbor.kind.value
            identity = "" if neighbor.agent_id is None else f" {neighbor.agent_id}"
            exterior = (
                ""
                if neighbor.agent_type is None
                else (", type A" if neighbor.agent_type == 1 else ", type B")
            )
            lines.append(
                f"- ({neighbor.location.row},{neighbor.location.column}): "
                f"{kind}{identity}{exterior}"
            )
        return (
            f"Round {observation.round_number} of {observation.horizon} for "
            f"ModelControlledAgent {self.agent_id}, exterior type "
            f"{'A' if self.agent_type == 1 else 'B'}.\n"
            f"Your absolute location is ({observation.location.row},"
            f"{observation.location.column}).\n"
            f"Objective: {observation.objective.value}.\n"
            f"Fixed final Counterfactual Reference homophily: {observation.reference_homophily}.\n"
            f"Current masked Ordinary Edge Homophily: {observation.current_homophily}.\n"
            "Remaining unreserved beginning-of-round vacancies: "
            f"{observation.remaining_vacancies}.\n"
            "Radius-1 neighborhood (frozen at turn start):\n" + "\n".join(lines)
        )

    async def take_turn(
        self, sessions: AgentSessionRuntime[ModelControlledAgent, Action]
    ) -> Action:
        from .models import Stay, SteeringObjective

        simulation = self._simulation
        cell = simulation.cell
        seed = np.random.SeedSequence(
            [
                *MASTER_SEED_WORDS,
                3,
                0 if simulation.objective == SteeringObjective.INTEGRATION else 1,
                cell.board_size,
                cell.vacancy_fraction.numerator,
                cell.vacancy_fraction.denominator,
                cell.tolerance.numerator,
                cell.tolerance.denominator,
                simulation.seed_id,
                simulation.rounds_completed + 1,
                2,
                self.agent_id,
            ]
        )
        result = await sessions.run(
            str(self.agent_id),
            self.prompt(self.observe()),
            deps=self,
            model_settings={"seed": int(seed.generate_state(1, dtype=np.uint32)[0])},
        )
        return Stay() if result is None else result
