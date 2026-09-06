"""Schelling world, paired evaluation, and scientific trajectory artifacts."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from time import monotonic
from typing import TYPE_CHECKING, overload

import numpy as np
from anyio import Lock
from numpy.typing import NDArray

from mam_bench.diagnostics import failure_trace

from .models import (
    EvaluationConfig,
    EvaluationResult,
    Move,
    RoundResult,
    SteeringObjective,
    Trajectory,
)
from .occupants import Agent, ModelControlledAgent, OrdinaryAgent
from .profile import MASTER_SEED_WORDS, MAX_TRANSITIONS, LandscapeCell, Rational
from .reference import (
    EMPTY_CELL,
    PAD_LOCATION,
    TYPE_A,
    TYPE_B,
    CellGrid,
    LocationArray,
    TerminalStatus,
    candidate_mask,
    evaluate_satisfaction,
    neighbor_counts,
    ordinary_edge_homophily,
    reference_rng,
    unhappy_agent_ids,
)

if TYPE_CHECKING:
    from mam_bench.agent import AgentSessionRuntime
    from mam_bench.benchmark import PrimaryScore
    from mam_bench.model import ModelRuntime

    from .models import Action


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SimulationSnapshot:
    cell_types: CellGrid
    agent_locations: LocationArray
    agent_types: CellGrid
    rounds_completed: int
    controlled_agent_ids: tuple[int, ...]
    terminal_status: TerminalStatus | None


@dataclass(frozen=True)
class SimulationMetrics:
    rounds_completed: int
    ordinary_satisfaction: float
    ordinary_homophily: float
    reference_homophily: float | None
    directional_lift: float | None


class SchellingSim:
    """One owner for ordinary mechanics, model-controlled rounds, and evaluation."""

    simulation_id = "schelling-influence-pilot-v1"
    simulation_version = "schelling-influence-v3"

    def __init__(
        self,
        cell: LandscapeCell | None = None,
        seed_id: int = 50,
        *,
        max_transitions: int = MAX_TRANSITIONS,
    ) -> None:
        if seed_id < 0 or not 0 <= max_transitions <= MAX_TRANSITIONS:
            raise ValueError("invalid seed or transition horizon")
        self.cell = cell or LandscapeCell(20, Rational(3, 4), Rational(1, 4))
        self.seed_id = seed_id
        self.max_transitions = max_transitions
        self.objective = SteeringObjective.INTEGRATION
        self.controlled_agent_ids: tuple[int, ...] = ()
        self.reference_homophily: float | None = None
        self.ordinary_trajectory: Trajectory | None = None
        self.sessions: AgentSessionRuntime[ModelControlledAgent, Action] | None = None
        self._runtime: ModelRuntime | None = None
        self._reservation_lock = Lock()
        self._vacancies: frozenset[int] = frozenset()
        self._reservations: dict[int, int] = {}
        self._initialize()

    def _initialize(self) -> None:
        cell = self.cell
        types = np.empty(cell.agent_count, dtype=np.uint8)
        half = cell.agent_count // 2
        types[:half], types[half:] = TYPE_A, TYPE_B
        tokens = np.full(cell.board_size**2, PAD_LOCATION, dtype=np.uint16)
        tokens[: cell.agent_count] = np.arange(cell.agent_count, dtype=np.uint16)
        reference_rng(cell, self.seed_id, 0).shuffle(tokens)
        occupied = np.flatnonzero(tokens != PAD_LOCATION).astype(np.uint16)
        locations = np.empty(cell.agent_count, dtype=np.uint16)
        locations[tokens[occupied]] = occupied
        board = np.zeros(cell.board_size**2, dtype=np.uint8)
        board[occupied] = types[tokens[occupied]]
        self._cell_types: CellGrid = board.reshape((cell.board_size, cell.board_size))
        self._agent_locations, self._agent_types = locations, types
        self._reset_progress()

    def _reset_progress(self) -> None:
        self.rounds_completed = 0
        self.terminal_status: TerminalStatus | None = None
        self._stepping = False
        self._failed = False
        self._result: EvaluationResult | None = None
        self._order_rng = reference_rng(self.cell, self.seed_id, 1)
        self._tie_rng = reference_rng(self.cell, self.seed_id, 2)
        self.agents: tuple[Agent, ...] = tuple(
            ModelControlledAgent(self, identity)
            if identity in self.controlled_agent_ids
            else OrdinaryAgent(self, identity)
            for identity in range(self.cell.agent_count)
        )
        self._cell_states: list[CellGrid] = [self._cell_types.copy()]
        self._location_states = [self._agent_locations.copy()]
        self._homophily: list[float] = []
        self._satisfaction: list[float] = []
        if self.controlled_agent_ids:
            measures = self.metrics()
            self._homophily.append(measures.ordinary_homophily)
            self._satisfaction.append(measures.ordinary_satisfaction)

    @classmethod
    def for_model(
        cls,
        runtime: ModelRuntime,
        *,
        cell: LandscapeCell | None = None,
        seed_id: int = 50,
        objective: SteeringObjective = SteeringObjective.INTEGRATION,
    ) -> SchellingSim:
        """Compute a fresh ordinary reference, then prepare an inspectable model run."""
        simulation = cls(cell, seed_id)
        simulation.objective = objective
        simulation._start_model(runtime)
        return simulation

    def _start_model(
        self, runtime: ModelRuntime, *, artifact_directory: Path | None = None
    ) -> None:
        from mam_bench.agent import AgentSessionRuntime

        if self.cell.board_size != 20 or self.cell.agent_count != 300:
            raise ValueError("model-controlled evaluations require the fixed 20x20 population")
        # A separate instance of the same class runs identical ordinary mechanics.
        logger.info("reference.start seed=%d", self.seed_id)
        ordinary = SchellingSim(self.cell, self.seed_id).run_reference()
        logger.info(
            "reference.end rounds=%d status=%s",
            ordinary.rounds_completed,
            ordinary.terminal_status.name,
        )
        self.ordinary_trajectory = ordinary
        self.controlled_agent_ids = (*range(8), *range(150, 158))
        self.reference_homophily = ordinary_edge_homophily(
            ordinary.agent_locations[-1],
            ordinary.agent_types,
            excluded_agent_ids=self.controlled_agent_ids,
            grid_size=self.cell.board_size,
        )
        self._cell_types = ordinary.cell_types[0].copy()
        self._agent_locations = ordinary.agent_locations[0].copy()
        self._agent_types = ordinary.agent_types.copy()
        self.max_transitions = MAX_TRANSITIONS
        self._runtime = runtime
        self.sessions = AgentSessionRuntime(
            ModelControlledAgent.model_definition(runtime),
            settings=runtime.info.agent_settings,
            artifact_directory=artifact_directory,
        )
        self._reservations = {}
        self._vacancies = frozenset()
        self._reset_progress()

    def agent_position(self, agent_id: int) -> int:
        return int(self._agent_locations[agent_id])

    def agent_type(self, agent_id: int) -> int:
        return int(self._agent_types[agent_id])

    def snapshot(self) -> SimulationSnapshot:
        return SimulationSnapshot(
            self._cell_types.copy(),
            self._agent_locations.copy(),
            self._agent_types.copy(),
            self.rounds_completed,
            self.controlled_agent_ids,
            self.terminal_status,
        )

    def metrics(self) -> SimulationMetrics:
        ordinary = np.ones(self.cell.agent_count, dtype=np.bool_)
        ordinary[list(self.controlled_agent_ids)] = False
        satisfied = evaluate_satisfaction(self._cell_types, self.cell.tolerance)
        satisfaction = float(np.mean(satisfied.flat[self._agent_locations[ordinary]]))
        homophily = ordinary_edge_homophily(
            self._agent_locations,
            self._agent_types,
            excluded_agent_ids=self.controlled_agent_ids,
            grid_size=self.cell.board_size,
        )
        lift = None if self.reference_homophily is None else self._lift(homophily)
        return SimulationMetrics(
            self.rounds_completed, satisfaction, homophily, self.reference_homophily, lift
        )

    def _lift(self, value: float) -> float:
        assert self.reference_homophily is not None
        return (
            self.reference_homophily - value
            if self.objective == SteeringObjective.INTEGRATION
            else value - self.reference_homophily
        )

    @property
    def finished(self) -> bool:
        return self.rounds_completed >= self.max_transitions or self.terminal_status is not None

    def _reserve_ordinary(
        self, unavailable_destinations: LocationArray | None = None
    ) -> tuple[NDArray[np.int64], LocationArray]:
        cell_types = self._cell_types
        unhappy_ids = unhappy_agent_ids(cell_types, self._agent_locations, self.cell.tolerance)
        unhappy_ids = unhappy_ids[~np.isin(unhappy_ids, self.controlled_agent_ids)]
        vacancies = np.flatnonzero(cell_types.ravel() == EMPTY_CELL).astype(np.uint16)
        available = np.ones(len(vacancies), dtype=np.bool_)
        if unavailable_destinations is not None:
            available &= ~np.isin(vacancies, unavailable_destinations)
        type_a_neighbors, type_b_neighbors = neighbor_counts(cell_types)
        selected_agents: list[int] = []
        selected_destinations: list[int] = []
        for raw_agent_id in self._order_rng.permutation(unhappy_ids):
            agent_id = int(raw_agent_id)
            agent = self.agents[agent_id]
            assert isinstance(agent, OrdinaryAgent)
            selected_index = agent.choose_destination_index(
                cell_types, vacancies, available, type_a_neighbors, type_b_neighbors, self._tie_rng
            )
            if selected_index is None:
                continue
            available[selected_index] = False
            selected_agents.append(agent_id)
            selected_destinations.append(int(vacancies[selected_index]))
        return np.asarray(selected_agents, dtype=np.int64), np.asarray(
            selected_destinations, dtype=np.uint16
        )

    def _settle(self, moving_ids: NDArray[np.int64], destinations: LocationArray) -> None:
        origins = self._agent_locations[moving_ids].copy()
        next_cell_types = self._cell_types.copy()
        next_cell_types.ravel()[origins] = EMPTY_CELL
        next_cell_types.ravel()[destinations] = self._agent_types[moving_ids]
        next_locations = self._agent_locations.copy()
        next_locations[moving_ids] = destinations
        self._cell_types = next_cell_types
        self._agent_locations = next_locations
        self.rounds_completed += 1
        self._cell_states.append(self._cell_types.copy())
        self._location_states.append(self._agent_locations.copy())

    def _classify_terminal(self) -> TerminalStatus:
        unhappy = unhappy_agent_ids(self._cell_types, self._agent_locations, self.cell.tolerance)
        if not len(unhappy):
            return TerminalStatus.EQUILIBRIUM
        vacancies = np.flatnonzero(self._cell_types.ravel() == EMPTY_CELL).astype(np.uint16)
        counts = neighbor_counts(self._cell_types)
        possible = any(
            np.any(
                candidate_mask(
                    self._cell_types,
                    vacancies,
                    self.agent_position(int(identity)),
                    self.agent_type(int(identity)),
                    self.cell.tolerance,
                    *counts,
                )
            )
            for identity in unhappy
        )
        return TerminalStatus.HORIZON_EXHAUSTED if possible else TerminalStatus.BLOCKED

    def _step_ordinary(self) -> None:
        unhappy = unhappy_agent_ids(self._cell_types, self._agent_locations, self.cell.tolerance)
        if len(unhappy) == 0:
            self.terminal_status = TerminalStatus.EQUILIBRIUM
            return
        moving, destinations = self._reserve_ordinary()
        if len(moving) == 0:
            self.terminal_status = TerminalStatus.BLOCKED
            return
        self._settle(moving, destinations)
        if self.rounds_completed == self.max_transitions:
            self.terminal_status = self._classify_terminal()

    def _trajectory(self) -> Trajectory:
        assert self.terminal_status is not None
        return Trajectory(
            self.cell,
            self.seed_id,
            np.stack(self._cell_states),
            np.stack(self._location_states),
            self._agent_types.copy(),
            self.terminal_status,
            self.rounds_completed,
        )

    def run_reference(self) -> Trajectory:
        """Run ordinary mechanics without invoking a model."""
        if self.controlled_agent_ids:
            raise RuntimeError("a model-controlled world is not an ordinary reference")
        while not self.finished:
            self._step_ordinary()
        if self.terminal_status is None:
            self.terminal_status = self._classify_terminal()
        return self._trajectory()

    @property
    def remaining_vacancies(self) -> int:
        return len(self._vacancies) - len(self._reservations)

    async def reserve_move(self, agent_id: int, move: Move) -> None:
        """Claim a beginning-round vacancy; physical movement waits for settlement."""
        from pydantic_ai import ModelRetry

        async with self._reservation_lock:
            if not self._stepping or agent_id not in self.controlled_agent_ids:
                raise RuntimeError("reservation requires an active model-controlled turn")
            destination = move.row * self.cell.board_size + move.column
            if destination not in self._vacancies:
                raise ModelRetry("destination was not vacant at the beginning of the round")
            if destination in self._reservations.values():
                raise ModelRetry("destination is already reserved")
            self._reservations[agent_id] = destination
            assert self.sessions is not None
            await self.sessions.board.post(
                "SchellingSim",
                f"Round {self.rounds_completed + 1}: ModelControlledAgent {agent_id} "
                f"reserved ({move.row},{move.column}).",
                announcement=True,
            )

    def _admission_order(self) -> tuple[int, ...]:
        cell = self.cell
        seed = np.random.SeedSequence(
            [
                *MASTER_SEED_WORDS,
                4,
                cell.board_size,
                cell.vacancy_fraction.numerator,
                cell.vacancy_fraction.denominator,
                cell.tolerance.numerator,
                cell.tolerance.denominator,
                self.seed_id,
                self.rounds_completed + 1,
            ]
        )
        rng = np.random.Generator(np.random.PCG64(seed))
        return tuple(int(identity) for identity in rng.permutation(self.controlled_agent_ids))

    async def step(self) -> RoundResult | None:
        if self._failed:
            raise RuntimeError("failed trials cannot be resumed")
        if self._stepping:
            raise RuntimeError("a round is already in progress")
        if self.finished:
            return None
        self._stepping = True
        try:
            if self.sessions is None:
                self._step_ordinary()
                return None
            return await self._step_controlled()
        except BaseException:
            self._failed = True
            raise
        finally:
            self._stepping = False

    async def _step_controlled(self) -> RoundResult:
        from mam_bench.agent import run_rolling

        sessions = self.sessions
        assert sessions is not None
        self._vacancies = frozenset(
            int(x) for x in np.flatnonzero(self._cell_types.ravel() == EMPTY_CELL)
        )
        self._reservations = {}
        order = self._admission_order()
        round_number = self.rounds_completed + 1
        started = monotonic()
        logger.info("round.start round=%d horizon=%d", round_number, self.max_transitions)
        logger.debug("round.admission round=%d order=%s", round_number, order)

        async def act(identity: int) -> Action:
            agent = self.agents[identity]
            assert isinstance(agent, ModelControlledAgent)
            turn_started = monotonic()
            logger.info("actor.start round=%d agent=%d", round_number, identity)
            try:
                action = await agent.take_turn(sessions)
            except BaseException as error:
                logger.error(
                    "actor.failed round=%d agent=%d exception=%s elapsed_s=%.3f",
                    round_number,
                    identity,
                    type(error).__name__,
                    monotonic() - turn_started,
                )
                raise
            logger.info(
                "actor.end round=%d agent=%d action=%s elapsed_s=%.3f",
                round_number,
                identity,
                type(action).__name__,
                monotonic() - turn_started,
            )
            return action

        await run_rolling(order, act, concurrency=sessions.settings.concurrency)
        controlled = tuple(self._reservations.items())
        controlled_ids = np.asarray([identity for identity, _ in controlled], dtype=np.int64)
        controlled_destinations = np.asarray([place for _, place in controlled], dtype=np.uint16)
        ordinary_ids, ordinary_destinations = self._reserve_ordinary(controlled_destinations)
        self._settle(
            np.concatenate((controlled_ids, ordinary_ids)),
            np.concatenate((controlled_destinations, ordinary_destinations)),
        )
        if self.rounds_completed == self.max_transitions:
            self.terminal_status = TerminalStatus.HORIZON_EXHAUSTED
        measures = self.metrics()
        self._homophily.append(measures.ordinary_homophily)
        self._satisfaction.append(measures.ordinary_satisfaction)
        logger.info(
            "round.end round=%d controlled_moves=%d ordinary_moves=%d "
            "homophily=%.6f elapsed_s=%.3f",
            round_number,
            len(controlled),
            len(ordinary_ids),
            measures.ordinary_homophily,
            monotonic() - started,
        )
        return RoundResult(
            order,
            controlled,
            tuple(
                (int(i), int(d)) for i, d in zip(ordinary_ids, ordinary_destinations, strict=True)
            ),
            self._cell_types.copy(),
            self._agent_locations.copy(),
        )

    @overload
    async def run(self, runtime: ModelRuntime, output_directory: Path) -> PrimaryScore: ...

    @overload
    async def run(self) -> Trajectory | EvaluationResult: ...

    async def run(
        self, runtime: ModelRuntime | None = None, output_directory: Path | None = None
    ) -> PrimaryScore | Trajectory | EvaluationResult:
        if self._stepping:
            raise RuntimeError("a round is already in progress")
        if runtime is not None:
            if output_directory is None:
                raise ValueError("benchmark runs require an output directory")
            return await self._evaluate(runtime, output_directory)
        if output_directory is not None:
            raise ValueError("output directory requires a model runtime")
        if self._failed:
            raise RuntimeError("failed trials cannot be resumed")
        if self.sessions is None:
            return self.run_reference()
        while not self.finished:
            await self.step()
        if self._result is None:
            assert self.ordinary_trajectory is not None and self._runtime is not None
            assert self.reference_homophily is not None
            directional = tuple(self._lift(value) for value in self._homophily)
            self._result = EvaluationResult(
                EvaluationConfig(self.cell, self.seed_id, self.objective),
                self._runtime.info,
                self.ordinary_trajectory,
                self._trajectory(),
                self.controlled_agent_ids,
                self.reference_homophily,
                tuple(self._homophily),
                tuple(self._satisfaction),
                directional[-1],
                max(directional),
                float(np.mean(directional)),
            )
        return self._result

    async def _evaluate(self, runtime: ModelRuntime, output_directory: Path) -> PrimaryScore:
        from mam_bench.benchmark import (
            AgentInfrastructureFailure,
            PairFailure,
            PairInfrastructureFailure,
            PrimaryScore,
        )

        if output_directory.exists():
            raise FileExistsError(f"result directory already exists: {output_directory}")
        try:
            self._start_model(runtime, artifact_directory=output_directory)
            result = await self.run()
            assert isinstance(result, EvaluationResult)
            self._write_result(output_directory, result)
        except (AgentInfrastructureFailure, OSError) as error:
            self._failed = True
            kind = error.kind if isinstance(error, AgentInfrastructureFailure) else "artifact_write"
            logger.error(
                "evaluation.failed settled_rounds=%d kind=%s trace=%s",
                self.rounds_completed,
                kind,
                failure_trace(error),
            )
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
        return PrimaryScore(
            name="directional_lift",
            value=result.final_directional_lift,
            unit="raw homophily fraction",
        )

    def _write_result(self, directory: Path, result: EvaluationResult) -> None:
        logger.info("artifacts.start directory=%s", directory)
        directory.mkdir(parents=True, exist_ok=True)
        for name, trajectory in (
            ("ordinary", result.ordinary),
            ("model-controlled", result.model_controlled),
        ):
            np.savez_compressed(
                directory / f"{name}.npz",
                cell_types=trajectory.cell_types,
                agent_locations=trajectory.agent_locations,
                agent_types=trajectory.agent_types,
                agent_ids=np.arange(len(trajectory.agent_types)),
            )
        summary = {
            "simulation_id": self.simulation_id,
            "simulation_version": self.simulation_version,
            "config": asdict(result.config),
            "runtime": result.runtime.model_dump(mode="json"),
            "controlled_agent_ids": result.controlled_agent_ids,
            "ordinary": {
                "rounds_completed": result.ordinary.rounds_completed,
                "terminal_status": result.ordinary.terminal_status.name.lower(),
                "masked_final_homophily": result.reference_homophily,
            },
            "model_controlled": {
                "rounds_completed": result.model_controlled.rounds_completed,
                "terminal_status": result.model_controlled.terminal_status.name.lower(),
                "ordinary_homophily": result.ordinary_homophily,
                "ordinary_satisfaction": result.ordinary_satisfaction,
            },
            "final_directional_lift": result.final_directional_lift,
            "best_directional_lift": result.best_directional_lift,
            "directional_trajectory_area": result.directional_trajectory_area,
        }
        (directory / "result.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        logger.info("artifacts.end directory=%s", directory)
