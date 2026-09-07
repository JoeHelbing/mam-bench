"""Private numerical calculations shared by Schelling domain classes."""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

from .profile import MASTER_SEED_WORDS, LandscapeCell, Rational

CellGrid = NDArray[np.uint8]
LocationArray = NDArray[np.uint16]
EMPTY_CELL = np.uint8(0)
TYPE_A = np.uint8(1)
TYPE_B = np.uint8(2)
PAD_CELL = np.uint8(255)
PAD_LOCATION = np.uint16(65535)


@dataclass(frozen=True)
class BoardCoordinate:
    """One absolute row and column on the toroidal board."""

    row: int
    column: int


class TerminalStatus(IntEnum):
    """Why a Reference Profile run stopped."""

    EQUILIBRIUM = 0
    BLOCKED = 1
    HORIZON_EXHAUSTED = 2


def toroidal_chebyshev_distance(first: int, second: int, *, size: int) -> int:
    """Return Chebyshev distance between flattened cells on a square torus."""

    first_row, first_column = divmod(first, size)
    second_row, second_column = divmod(second, size)
    row_delta = abs(first_row - second_row)
    column_delta = abs(first_column - second_column)
    wrapped_row = min(row_delta, size - row_delta)
    wrapped_column = min(column_delta, size - column_delta)
    return max(wrapped_row, wrapped_column)


def neighbor_counts(cell_types: CellGrid) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
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
    type_a, type_b = neighbor_counts(cell_types)
    occupied_neighbors = type_a + type_b
    occupied_cells = cell_types != EMPTY_CELL
    same_neighbors = np.where(cell_types == TYPE_A, type_a, type_b)
    satisfied = np.zeros(cell_types.shape, dtype=np.bool_)
    satisfied[occupied_cells] = (occupied_neighbors[occupied_cells] == 0) | (
        same_neighbors[occupied_cells].astype(np.int64) * tolerance.denominator
        >= occupied_neighbors[occupied_cells].astype(np.int64) * tolerance.numerator
    )
    return satisfied


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


def candidate_mask(
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


def distances(origin: int, destinations: LocationArray, size: int) -> NDArray[np.int16]:
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


def choose_nearest_index(nearest_indices: NDArray[np.int64], tie_rng: np.random.Generator) -> int:
    """Choose among true distance ties without drawing for a singleton."""

    if len(nearest_indices) == 0:
        raise ValueError("nearest_indices cannot be empty")
    if len(nearest_indices) == 1:
        return int(nearest_indices[0])
    return int(nearest_indices[int(tie_rng.integers(len(nearest_indices)))])


def ordinary_edge_homophily(
    agent_locations: LocationArray,
    agent_types: NDArray[np.uint8],
    *,
    excluded_agent_ids: tuple[int, ...],
    grid_size: int,
) -> float:
    """Return same-type share of undirected scored Ordinary-Ordinary edges."""

    if agent_locations.shape != agent_types.shape:
        raise ValueError("agent locations and types must have matching shapes")
    scored = np.ones(len(agent_types), dtype=np.bool_)
    if excluded_agent_ids:
        excluded = np.asarray(excluded_agent_ids, dtype=np.int64)
        if np.any(excluded < 0) or np.any(excluded >= len(agent_types)):
            raise ValueError("excluded agent ID is outside the population")
        scored[excluded] = False
    grid = np.zeros((grid_size, grid_size), dtype=np.uint8)
    grid.ravel()[agent_locations[scored]] = agent_types[scored]
    same_edges = 0
    edge_count = 0
    for row_delta, column_delta in ((0, 1), (1, -1), (1, 0), (1, 1)):
        neighbor = np.roll(grid, shift=(-row_delta, -column_delta), axis=(0, 1))
        edges = (grid != EMPTY_CELL) & (neighbor != EMPTY_CELL)
        edge_count += int(np.count_nonzero(edges))
        same_edges += int(np.count_nonzero(edges & (grid == neighbor)))
    if edge_count == 0:
        raise ValueError("Ordinary Edge Homophily requires at least one scored edge")
    return same_edges / edge_count
