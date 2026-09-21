"""Schelling spatial state and measurements, shared by both worlds."""

from fractions import Fraction

import numpy as np
from numpy.typing import NDArray

from .settings import SchellingSettings

type CellGrid = NDArray[np.uint8]
type Locations = NDArray[np.int64]


class Board:
    def __init__(self, settings: SchellingSettings, rng: np.random.Generator) -> None:
        self.settings = settings
        self.types: CellGrid = np.repeat(
            np.array([1, 2], dtype=np.uint8), settings.agent_count // 2
        )
        self.locations: Locations = rng.permutation(settings.board_size**2)[: settings.agent_count]
        self.cells: CellGrid = np.zeros((settings.board_size, settings.board_size), dtype=np.uint8)
        self.cells.flat[self.locations] = self.types

    def distances(self, origin: int, destinations: Locations) -> NDArray[np.int64]:
        size = self.settings.board_size
        row, column = divmod(origin, size)
        dr = np.abs(destinations // size - row)
        dc = np.abs(destinations % size - column)
        return np.maximum(np.minimum(dr, size - dr), np.minimum(dc, size - dc))

    def neighbor_counts(self) -> tuple[NDArray[np.int16], NDArray[np.int16]]:
        a = np.zeros(self.cells.shape, dtype=np.int16)
        b = np.zeros(self.cells.shape, dtype=np.int16)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr or dc:
                    neighbor = np.roll(self.cells, shift=(-dr, -dc), axis=(0, 1))
                    a += neighbor == 1
                    b += neighbor == 2
        return a, b

    def satisfaction(self) -> NDArray[np.bool_]:
        a, b = self.neighbor_counts()
        total = a + b
        same = np.where(self.cells == 1, a, b)
        tolerance = Fraction(self.settings.tolerance)
        return (self.cells != 0) & (
            (total == 0)
            | (
                same.astype(np.int64) * tolerance.denominator
                >= total.astype(np.int64) * tolerance.numerator
            )
        )

    @property
    def homophily(self) -> float:
        """Undirected radius-one edges between the fixed scored identities."""
        scored = np.ones(len(self.types), dtype=np.bool_)
        scored[list(self.settings.controlled_agent_ids)] = False
        grid = np.zeros_like(self.cells)
        grid.flat[self.locations[scored]] = self.types[scored]
        same = total = 0
        for dr, dc in ((0, 1), (1, -1), (1, 0), (1, 1)):
            neighbor = np.roll(grid, shift=(-dr, -dc), axis=(0, 1))
            edges = (grid != 0) & (neighbor != 0)
            total += int(np.count_nonzero(edges))
            same += int(np.count_nonzero(edges & (grid == neighbor)))
        if not total:
            raise ValueError("homophily requires a scored edge")
        return same / total

    @property
    def ordinary_satisfaction(self) -> float:
        scored = np.ones(len(self.types), dtype=np.bool_)
        scored[list(self.settings.controlled_agent_ids)] = False
        return float(np.mean(self.satisfaction().flat[self.locations[scored]]))

    def settle(self, moves: dict[int, int]) -> None:
        identities = np.array(list(moves), dtype=np.int64)
        destinations = np.array(list(moves.values()), dtype=np.int64)
        self.cells.flat[self.locations[identities]] = 0
        self.cells.flat[destinations] = self.types[identities]
        self.locations[identities] = destinations
