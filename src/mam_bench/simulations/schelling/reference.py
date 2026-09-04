"""Host-neutral engine for the frozen Schelling Reference Profile."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from .profile import MASTER_SEED_WORDS, MAX_TRANSITIONS, LandscapeCell, Rational

CellGrid = NDArray[np.uint8]
LocationArray = NDArray[np.uint16]
EMPTY_CELL = np.uint8(0)
TYPE_A = np.uint8(1)
TYPE_B = np.uint8(2)
PAD_CELL = np.uint8(255)
PAD_LOCATION = np.uint16(65535)


class TerminalStatus(IntEnum):
    """Why a Reference Profile run stopped."""

    EQUILIBRIUM = 0
    BLOCKED = 1
    HORIZON_EXHAUSTED = 2


@dataclass(frozen=True)
class ReferenceTrajectory:
    """Complete raw states and stable-agent locations for one run."""

    cell: LandscapeCell
    seed_id: int
    cell_types: np.ndarray
    agent_locations: np.ndarray
    agent_types: np.ndarray
    terminal_status: TerminalStatus
    rounds_completed: int

    @property
    def trajectory_length(self) -> int:
        return self.rounds_completed + 1


def toroidal_chebyshev_distance(first: int, second: int, *, size: int) -> int:
    """Return Chebyshev distance between flattened cells on a square torus."""

    first_row, first_column = divmod(first, size)
    second_row, second_column = divmod(second, size)
    row_delta = abs(first_row - second_row)
    column_delta = abs(first_column - second_column)
    wrapped_row = min(row_delta, size - row_delta)
    wrapped_column = min(column_delta, size - column_delta)
    return max(wrapped_row, wrapped_column)


def _neighbor_counts(cell_types: CellGrid) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
    type_a = np.zeros(cell_types.shape, dtype=np.int16)
    type_b = np.zeros(cell_types.shape, dtype=np.int16)
    for row_delta in (-1, 0, 1):
        for column_delta in (-1, 0, 1):
            if row_delta == 0 and column_delta == 0:
                continue
            neighbor = np.roll(cell_types, shift=(-row_delta, -column_delta), axis=(0, 1))
            type_a += neighbor == TYPE_A
            type_b += neighbor == TYPE_B
    return type_a, type_b


def evaluate_satisfaction(cell_types: CellGrid, tolerance: Rational) -> NDArray[np.bool_]:
    """Evaluate occupied cells using exact rational arithmetic."""

    if cell_types.ndim != 2 or cell_types.shape[0] != cell_types.shape[1]:
        raise ValueError("cell_types must be a square two-dimensional array")
    type_a, type_b = _neighbor_counts(cell_types)
    occupied_neighbors = type_a + type_b
    occupied_cells = cell_types != EMPTY_CELL
    same_neighbors = np.where(cell_types == TYPE_A, type_a, type_b)
    satisfied = np.zeros(cell_types.shape, dtype=np.bool_)
    satisfied[occupied_cells] = (occupied_neighbors[occupied_cells] == 0) | (
        same_neighbors[occupied_cells] * tolerance.denominator
        >= occupied_neighbors[occupied_cells] * tolerance.numerator
    )
    return satisfied


def candidate_is_satisfactory(
    cell_types: CellGrid,
    *,
    origin: int,
    destination: int,
    agent_type: int,
    tolerance: Rational,
) -> bool:
    """Evaluate one counterfactual move against a frozen grid."""

    size = cell_types.shape[0]
    if cell_types.shape != (size, size):
        raise ValueError("cell_types must be square")
    if int(cell_types.flat[origin]) != agent_type:
        raise ValueError("origin does not contain the moving agent type")
    if cell_types.flat[destination] != EMPTY_CELL:
        raise ValueError("destination is not vacant")
    type_a, type_b = _neighbor_counts(cell_types)
    row, column = divmod(destination, size)
    same = int(type_a[row, column] if agent_type == TYPE_A else type_b[row, column])
    occupied = int(type_a[row, column] + type_b[row, column])
    if toroidal_chebyshev_distance(origin, destination, size=size) == 1:
        same -= 1
        occupied -= 1
    if occupied == 0:
        return True
    return same * tolerance.denominator >= occupied * tolerance.numerator


def _seed_words(
    cell: LandscapeCell,
    seed_id: int,
    stream_id: int,
) -> list[int]:
    words = [
        MASTER_SEED_WORDS[0],
        MASTER_SEED_WORDS[1],
        stream_id,
        cell.board_size,
        cell.vacancy_fraction.numerator,
        cell.vacancy_fraction.denominator,
    ]
    if stream_id != 0:
        words.extend((cell.tolerance.numerator, cell.tolerance.denominator))
    words.append(seed_id)
    return words


def reference_rng(cell: LandscapeCell, seed_id: int, stream_id: int) -> np.random.Generator:
    seed_sequence = np.random.SeedSequence(_seed_words(cell, seed_id, stream_id))
    return np.random.Generator(np.random.PCG64(seed_sequence))


def _initialize(
    cell: LandscapeCell, seed_id: int
) -> tuple[CellGrid, LocationArray, NDArray[np.uint8]]:
    agent_types = np.empty(cell.agent_count, dtype=np.uint8)
    half = cell.agent_count // 2
    agent_types[:half] = TYPE_A
    agent_types[half:] = TYPE_B

    tokens = np.full(cell.board_size * cell.board_size, PAD_LOCATION, dtype=np.uint16)
    tokens[: cell.agent_count] = np.arange(cell.agent_count, dtype=np.uint16)
    reference_rng(cell, seed_id, 0).shuffle(tokens)

    occupied_positions = np.flatnonzero(tokens != PAD_LOCATION).astype(np.uint16)
    agent_locations = np.empty(cell.agent_count, dtype=np.uint16)
    agent_locations[tokens[occupied_positions]] = occupied_positions

    cell_types = np.zeros(cell.board_size * cell.board_size, dtype=np.uint8)
    cell_types[occupied_positions] = agent_types[tokens[occupied_positions]]
    return cell_types.reshape((cell.board_size, cell.board_size)), agent_locations, agent_types


def _candidate_mask(
    cell_types: CellGrid,
    vacancies: LocationArray,
    origin: int,
    agent_type: int,
    tolerance: Rational,
    type_a_neighbors: NDArray[np.int16],
    type_b_neighbors: NDArray[np.int16],
) -> NDArray[np.bool_]:
    size = cell_types.shape[0]
    vacancy_rows = vacancies // size
    vacancy_columns = vacancies % size
    same = (
        type_a_neighbors[vacancy_rows, vacancy_columns]
        if agent_type == TYPE_A
        else type_b_neighbors[vacancy_rows, vacancy_columns]
    ).copy()
    occupied = (
        type_a_neighbors[vacancy_rows, vacancy_columns]
        + type_b_neighbors[vacancy_rows, vacancy_columns]
    ).copy()

    origin_row, origin_column = divmod(origin, size)
    row_delta = np.abs(vacancy_rows.astype(np.int16) - origin_row)
    column_delta = np.abs(vacancy_columns.astype(np.int16) - origin_column)
    row_distance = np.minimum(row_delta, size - row_delta)
    column_distance = np.minimum(column_delta, size - column_delta)
    origin_is_neighbor = np.maximum(row_distance, column_distance) == 1
    same -= origin_is_neighbor
    occupied -= origin_is_neighbor
    return (occupied == 0) | (same * tolerance.denominator >= occupied * tolerance.numerator)


def _distances(origin: int, destinations: LocationArray, size: int) -> NDArray[np.int16]:
    origin_row, origin_column = divmod(origin, size)
    rows = destinations // size
    columns = destinations % size
    row_delta = np.abs(rows.astype(np.int16) - origin_row)
    column_delta = np.abs(columns.astype(np.int16) - origin_column)
    row_distance = np.minimum(row_delta, size - row_delta)
    column_distance = np.minimum(column_delta, size - column_delta)
    return np.maximum(row_distance, column_distance)


def unhappy_agent_ids(
    cell_types: CellGrid, agent_locations: LocationArray, tolerance: Rational
) -> NDArray[np.int64]:
    satisfied_cells = evaluate_satisfaction(cell_types, tolerance)
    satisfied_agents = satisfied_cells.flat[agent_locations]
    return np.flatnonzero(~satisfied_agents)


def _choose_nearest_index(nearest_indices: NDArray[np.int64], tie_rng: np.random.Generator) -> int:
    """Choose among true distance ties without drawing for a singleton."""

    if len(nearest_indices) == 0:
        raise ValueError("nearest_indices cannot be empty")
    if len(nearest_indices) == 1:
        return int(nearest_indices[0])
    return int(nearest_indices[int(tie_rng.integers(len(nearest_indices)))])


def reserve_ordinary_destinations(
    cell_types: CellGrid,
    agent_locations: LocationArray,
    agent_types: NDArray[np.uint8],
    unhappy_ids: NDArray[np.int64],
    tolerance: Rational,
    order_rng: np.random.Generator,
    tie_rng: np.random.Generator,
    *,
    unavailable_destinations: LocationArray | None = None,
) -> tuple[NDArray[np.int64], LocationArray]:
    vacancies = np.flatnonzero(cell_types.ravel() == EMPTY_CELL).astype(np.uint16)
    available = np.ones(len(vacancies), dtype=np.bool_)
    if unavailable_destinations is not None:
        available &= ~np.isin(vacancies, unavailable_destinations)
    type_a_neighbors, type_b_neighbors = _neighbor_counts(cell_types)
    selected_agents: list[int] = []
    selected_destinations: list[int] = []

    for raw_agent_id in order_rng.permutation(unhappy_ids):
        agent_id = int(raw_agent_id)
        eligible = available & _candidate_mask(
            cell_types,
            vacancies,
            int(agent_locations[agent_id]),
            int(agent_types[agent_id]),
            tolerance,
            type_a_neighbors,
            type_b_neighbors,
        )
        eligible_indices = np.flatnonzero(eligible)
        if len(eligible_indices) == 0:
            continue
        eligible_destinations = vacancies[eligible_indices]
        distance = _distances(
            int(agent_locations[agent_id]), eligible_destinations, cell_types.shape[0]
        )
        nearest_indices = eligible_indices[distance == distance.min()]
        selected_index = _choose_nearest_index(nearest_indices, tie_rng)
        available[selected_index] = False
        selected_agents.append(agent_id)
        selected_destinations.append(int(vacancies[selected_index]))

    return (
        np.asarray(selected_agents, dtype=np.int64),
        np.asarray(selected_destinations, dtype=np.uint16),
    )


def _has_possible_move(
    cell_types: CellGrid,
    agent_locations: LocationArray,
    agent_types: NDArray[np.uint8],
    unhappy_ids: NDArray[np.int64],
    tolerance: Rational,
) -> bool:
    vacancies = np.flatnonzero(cell_types.ravel() == EMPTY_CELL).astype(np.uint16)
    type_a_neighbors, type_b_neighbors = _neighbor_counts(cell_types)
    return any(
        np.any(
            _candidate_mask(
                cell_types,
                vacancies,
                int(agent_locations[int(agent_id)]),
                int(agent_types[int(agent_id)]),
                tolerance,
                type_a_neighbors,
                type_b_neighbors,
            )
        )
        for agent_id in unhappy_ids
    )


def _terminal_status(
    cell_types: CellGrid,
    agent_locations: LocationArray,
    agent_types: NDArray[np.uint8],
    tolerance: Rational,
) -> TerminalStatus:
    unhappy_ids = unhappy_agent_ids(cell_types, agent_locations, tolerance)
    if len(unhappy_ids) == 0:
        return TerminalStatus.EQUILIBRIUM
    if not _has_possible_move(cell_types, agent_locations, agent_types, unhappy_ids, tolerance):
        return TerminalStatus.BLOCKED
    return TerminalStatus.HORIZON_EXHAUSTED


def run_reference(
    cell: LandscapeCell,
    seed_id: int,
    *,
    max_transitions: int = MAX_TRANSITIONS,
) -> ReferenceTrajectory:
    """Run one deterministic Reference Profile trajectory."""

    if seed_id < 0:
        raise ValueError("seed_id cannot be negative")
    if max_transitions < 0 or max_transitions > MAX_TRANSITIONS:
        raise ValueError(f"max_transitions must be between 0 and {MAX_TRANSITIONS}")

    cell_types, agent_locations, agent_types = _initialize(cell, seed_id)
    cell_states = [cell_types.copy()]
    location_states = [agent_locations.copy()]
    order_rng = reference_rng(cell, seed_id, 1)
    tie_rng = reference_rng(cell, seed_id, 2)
    terminal_status: TerminalStatus | None = None

    for _ in range(max_transitions):
        unhappy_ids = unhappy_agent_ids(cell_types, agent_locations, cell.tolerance)
        if len(unhappy_ids) == 0:
            terminal_status = TerminalStatus.EQUILIBRIUM
            break
        moving_agents, destinations = reserve_ordinary_destinations(
            cell_types,
            agent_locations,
            agent_types,
            unhappy_ids,
            cell.tolerance,
            order_rng,
            tie_rng,
        )
        if len(moving_agents) == 0:
            terminal_status = TerminalStatus.BLOCKED
            break

        origins = agent_locations[moving_agents].copy()
        next_cell_types = cell_types.copy()
        next_cell_types.ravel()[origins] = EMPTY_CELL
        next_cell_types.ravel()[destinations] = agent_types[moving_agents]
        next_locations = agent_locations.copy()
        next_locations[moving_agents] = destinations
        cell_types = next_cell_types
        agent_locations = next_locations
        cell_states.append(cell_types.copy())
        location_states.append(agent_locations.copy())

    if terminal_status is None:
        terminal_status = _terminal_status(cell_types, agent_locations, agent_types, cell.tolerance)

    cell_array = np.stack(cell_states).astype(np.uint8, copy=False)
    location_array = np.stack(location_states).astype(np.uint16, copy=False)
    return ReferenceTrajectory(
        cell=cell,
        seed_id=seed_id,
        cell_types=cell_array,
        agent_locations=location_array,
        agent_types=agent_types,
        terminal_status=terminal_status,
        rounds_completed=len(cell_states) - 1,
    )
