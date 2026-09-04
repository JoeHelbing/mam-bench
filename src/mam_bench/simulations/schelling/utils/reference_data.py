"""Build and serialize the fixed Schelling evaluation reference.

The fixture is a held-out all-Ordinary-Agent run for the benchmark's exact cell
and seed. It retains the initial identities and board, terminal comparator state,
termination facts, and scoring metrics while discarding unused intermediate
states. The arrays are stored in a compressed NPZ archive, and readable JSON
metadata records the parameters, metrics, archive name, and SHA-256 digest.

Runtime loading lives in ``schelling.fixture`` so benchmark execution does not
depend on this offline publishing package.
"""

import hashlib
import io
from pathlib import Path

import numpy as np

from ..fixture import ARCHIVE_NAME, METADATA_NAME, EvaluationReferenceMetadata
from ..models import CounterfactualReference
from ..profile import PROFILE_VERSION, LandscapeCell
from ..reference import run_reference
from ..runtime import (
    influence_actor_ids,
    ordinary_edge_homophily,
    ordinary_satisfaction_fraction,
)


def build_evaluation_reference(cell: LandscapeCell, seed_id: int) -> CounterfactualReference:
    trajectory = run_reference(cell, seed_id)
    actor_ids = influence_actor_ids(cell.agent_count)
    terminal_locations = trajectory.agent_locations[-1]
    terminal_grid = trajectory.cell_types[-1]
    return CounterfactualReference(
        cell=cell,
        seed_id=seed_id,
        terminal_status=trajectory.terminal_status,
        rounds_completed=trajectory.rounds_completed,
        initial_cell_types=trajectory.cell_types[0],
        initial_agent_locations=trajectory.agent_locations[0],
        terminal_cell_types=terminal_grid,
        terminal_agent_locations=terminal_locations,
        agent_types=trajectory.agent_types,
        masked_agent_count=cell.agent_count - len(actor_ids),
        masked_final_homophily=ordinary_edge_homophily(
            terminal_locations,
            trajectory.agent_types,
            excluded_agent_ids=actor_ids,
            grid_size=cell.board_size,
        ),
        unmasked_final_homophily=ordinary_edge_homophily(
            terminal_locations,
            trajectory.agent_types,
            excluded_agent_ids=(),
            grid_size=cell.board_size,
        ),
        ordinary_satisfaction=ordinary_satisfaction_fraction(
            terminal_grid,
            terminal_locations,
            cell=cell,
        ),
    )


def write_evaluation_reference(root: Path, reference: CounterfactualReference) -> None:
    root.mkdir(parents=True, exist_ok=True)
    stream = io.BytesIO()
    np.savez_compressed(
        stream,
        initial_cell_types=reference.initial_cell_types,
        initial_agent_locations=reference.initial_agent_locations,
        terminal_cell_types=reference.terminal_cell_types,
        terminal_agent_locations=reference.terminal_agent_locations,
        agent_types=reference.agent_types,
    )
    archive = stream.getvalue()
    (root / ARCHIVE_NAME).write_bytes(archive)
    metadata = EvaluationReferenceMetadata(
        profile_version=PROFILE_VERSION,
        board_size=reference.cell.board_size,
        tolerance=str(reference.cell.tolerance),
        vacancy_fraction=str(reference.cell.vacancy_fraction),
        vacancy_count=reference.cell.vacancy_count,
        seed_id=reference.seed_id,
        terminal_status=reference.terminal_status.name.lower(),
        rounds_completed=reference.rounds_completed,
        masked_agent_count=reference.masked_agent_count,
        masked_final_homophily=reference.masked_final_homophily,
        unmasked_final_homophily=reference.unmasked_final_homophily,
        ordinary_satisfaction=reference.ordinary_satisfaction,
        archive=ARCHIVE_NAME,
        archive_sha256=hashlib.sha256(archive).hexdigest(),
    )
    (root / METADATA_NAME).write_text(f"{metadata.model_dump_json(indent=2)}\n", encoding="utf-8")
