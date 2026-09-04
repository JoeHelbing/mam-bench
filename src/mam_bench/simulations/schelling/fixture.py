"""Load the compact reference used by the fixed evaluation."""

import hashlib
import io
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from .models import CounterfactualReference
from .profile import LandscapeCell, Rational
from .reference import TerminalStatus

ARCHIVE_NAME = "evaluation-reference.npz"
METADATA_NAME = "evaluation-reference.json"
ARRAY_NAMES = {
    "initial_cell_types",
    "initial_agent_locations",
    "terminal_cell_types",
    "terminal_agent_locations",
    "agent_types",
}


class EvaluationReferenceMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    profile_version: str
    board_size: int
    tolerance: str
    vacancy_fraction: str
    vacancy_count: int
    seed_id: int
    terminal_status: str
    rounds_completed: int
    masked_agent_count: int
    masked_final_homophily: float
    unmasked_final_homophily: float
    ordinary_satisfaction: float
    archive: str
    archive_sha256: str


def load_evaluation_reference(root: Path) -> CounterfactualReference:
    """Load the packaged initial and terminal reference fixture."""

    metadata = EvaluationReferenceMetadata.model_validate_json(
        (root / METADATA_NAME).read_text(encoding="utf-8")
    )
    archive = (root / metadata.archive).read_bytes()
    if hashlib.sha256(archive).hexdigest() != metadata.archive_sha256:
        raise ValueError("evaluation reference archive SHA-256 differs")
    with np.load(io.BytesIO(archive), allow_pickle=False) as loaded:
        if set(loaded.files) != ARRAY_NAMES:
            raise ValueError("evaluation reference archive has unexpected arrays")
        arrays = {name: loaded[name].copy() for name in loaded.files}

    return CounterfactualReference(
        cell=LandscapeCell(
            metadata.board_size,
            Rational.parse(metadata.tolerance),
            Rational.parse(metadata.vacancy_fraction),
        ),
        seed_id=metadata.seed_id,
        terminal_status=TerminalStatus[metadata.terminal_status.upper()],
        rounds_completed=metadata.rounds_completed,
        initial_cell_types=arrays["initial_cell_types"],
        initial_agent_locations=arrays["initial_agent_locations"],
        terminal_cell_types=arrays["terminal_cell_types"],
        terminal_agent_locations=arrays["terminal_agent_locations"],
        agent_types=arrays["agent_types"],
        masked_agent_count=metadata.masked_agent_count,
        masked_final_homophily=metadata.masked_final_homophily,
        unmasked_final_homophily=metadata.unmasked_final_homophily,
        ordinary_satisfaction=metadata.ordinary_satisfaction,
    )
