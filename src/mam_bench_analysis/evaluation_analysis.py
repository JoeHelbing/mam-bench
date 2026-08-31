"""Derived comparison of one Model Evaluation and its same-seed reference."""

import json
from collections import Counter
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, model_validator

from mam_bench.simulations.schelling.evaluation import (
    SteeringObjective,
    influence_actor_ids,
)
from mam_bench.simulations.schelling.evidence import (
    CoordinationBoardRound,
    validate_evaluation_artifact,
)
from mam_bench.simulations.schelling.profile import PROFILE
from mam_bench.simulations.schelling.reference import TerminalStatus, run_reference
from mam_bench.simulations.schelling.scoring import (
    ordinary_edge_homophily,
    ordinary_satisfaction_fraction,
)

REFERENCE_HOMOPHILY_COLOR = "#f07c54"
MODEL_HOMOPHILY_COLOR = "#56b4e9"
REFERENCE_SATISFACTION_COLOR = "#7fc97f"
MODEL_SATISFACTION_COLOR = "#e78ac3"


class MetricTrajectory(BaseModel):
    """Round-indexed Ordinary Agent metrics for one trajectory."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rounds: tuple[int, ...]
    ordinary_homophily: tuple[float, ...]
    ordinary_satisfaction: tuple[float, ...]

    @model_validator(mode="after")
    def validate_lengths(self) -> Self:
        lengths = {
            len(self.rounds),
            len(self.ordinary_homophily),
            len(self.ordinary_satisfaction),
        }
        if len(lengths) != 1 or not self.rounds:
            raise ValueError("metric trajectory arrays must have one shared nonzero length")
        return self


class ModelEvaluationAnalysis(BaseModel):
    """Inspectable benchmark comparison and coordination diagnostics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = "mam-bench.model-evaluation-analysis.v1"
    case_path: str
    model_name: str
    provider: str
    provider_backend: str
    objective: SteeringObjective
    tolerance_fraction: str
    vacancy_percentage: float
    seed_id: int
    scored_ordinary_agent_count: int
    reference_terminal_status: TerminalStatus
    reference_rounds_completed: int
    model_rounds_completed: int
    initial_homophily: float
    initial_satisfaction: float
    reference_final_homophily: float
    reference_final_satisfaction: float
    model_final_homophily: float
    model_final_satisfaction: float
    final_directional_lift: float
    directional_trajectory_area: float
    relative_final_homophily_reduction: float
    best_directional_lift: float
    reference_trajectory: MetricTrajectory
    model_trajectory: MetricTrajectory
    accepted_actor_moves: int
    actor_stays: int
    actor_collisions: int
    occupied_destination_failures: int
    policy_error_outcomes: int
    ordinary_moves: int
    accepted_actor_moves_by_round: tuple[int, ...]
    ordinary_moves_by_round: tuple[int, ...]
    coordination_post_count: int
    empty_coordination_posts: int
    median_coordination_characters: float
    coordination_posts_over_2000_characters: int
    coordination_posts_over_5000_characters: int
    longest_coordination_post_characters: int
    longest_coordination_post_round: int
    longest_coordination_post_actor: int
    collision_language_posts: int
    hold_language_posts: int
    boundary_language_posts: int
    anchor_language_posts: int
    opposite_type_language_posts: int
    total_requests: int
    total_input_tokens: int
    total_output_tokens: int


def write_model_evaluation_analysis(
    analysis: ModelEvaluationAnalysis,
    output_path: Path,
) -> Path:
    """Write one validated model-evaluation analysis as JSON."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        f"{analysis.model_dump_json(indent=2)}\n",
        encoding="utf-8",
    )
    return output_path


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def analyze_model_evaluation(case_directory: Path) -> ModelEvaluationAnalysis:
    """Validate and compare one complete case with its deterministic reference."""

    summary = validate_evaluation_artifact(case_directory)
    with np.load(case_directory / "trajectory.npz", allow_pickle=False) as archive:
        model_homophily = np.asarray(archive["ordinary_homophily"], dtype=np.float64)
        model_satisfaction = np.asarray(archive["ordinary_satisfaction"], dtype=np.float64)

    reference = run_reference(summary.config.cell, summary.config.seed_id)
    actor_ids = influence_actor_ids(summary.config.cell.agent_count)
    reference_homophily = tuple(
        ordinary_edge_homophily(
            np.asarray(locations, dtype=np.uint16),
            np.asarray(reference.agent_types, dtype=np.uint8),
            excluded_agent_ids=actor_ids,
            grid_size=PROFILE.grid_size,
        )
        for locations in reference.agent_locations
    )
    reference_satisfaction = tuple(
        ordinary_satisfaction_fraction(
            np.asarray(cell_types, dtype=np.uint8),
            np.asarray(locations, dtype=np.uint16),
            cell=summary.config.cell,
        )
        for cell_types, locations in zip(
            reference.cell_types,
            reference.agent_locations,
            strict=True,
        )
    )

    events = [
        json.loads(line)
        for line in (case_directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    coordination_events = [event for event in events if event["event_type"] == "coordination_post"]
    proposal_events = [event for event in events if event["event_type"] == "move_proposal"]
    outcome_events = [event for event in events if event["event_type"] == "move_outcome"]
    ordinary_events = [event for event in events if event["event_type"] == "ordinary_move_batch"]
    outcome_counts = Counter(str(event["reason"]) for event in outcome_events)
    board_rounds = tuple(
        CoordinationBoardRound.model_validate_json(line)
        for line in (case_directory / "coordination-board.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    )
    posts = tuple(post for board_round in board_rounds for post in board_round.posts)
    post_lengths = np.asarray([len(post.text) for post in posts], dtype=np.int64)
    longest_index = int(np.argmax(post_lengths))
    longest_post = posts[longest_index]
    longest_round = board_rounds[longest_index // len(actor_ids)].round_number

    accepted_by_round = tuple(
        sum(
            event["round_number"] == round_number and event["reason"] == "accepted"
            for event in outcome_events
        )
        for round_number in range(1, summary.rounds_completed + 1)
    )
    ordinary_by_round = tuple(
        len(event["moving_agent_ids"])
        for event in sorted(ordinary_events, key=lambda event: event["round_number"])
    )
    request_events = coordination_events + proposal_events
    final_lift_values = tuple(
        (
            summary.reference_masked_final_homophily - float(value)
            if summary.config.objective == SteeringObjective.INTEGRATION
            else float(value) - summary.reference_masked_final_homophily
        )
        for value in model_homophily
    )

    return ModelEvaluationAnalysis(
        case_path=str(case_directory),
        model_name=summary.config.runtime.model_name,
        provider=summary.config.runtime.provider,
        provider_backend=summary.config.runtime.openrouter_provider_slug,
        objective=summary.config.objective,
        tolerance_fraction=str(summary.config.cell.tolerance),
        vacancy_percentage=(
            100.0
            * summary.config.cell.vacancy_fraction.numerator
            / summary.config.cell.vacancy_fraction.denominator
        ),
        seed_id=summary.config.seed_id,
        scored_ordinary_agent_count=summary.scored_ordinary_agent_count,
        reference_terminal_status=reference.terminal_status,
        reference_rounds_completed=reference.rounds_completed,
        model_rounds_completed=summary.rounds_completed,
        initial_homophily=float(model_homophily[0]),
        initial_satisfaction=float(model_satisfaction[0]),
        reference_final_homophily=float(reference_homophily[-1]),
        reference_final_satisfaction=float(reference_satisfaction[-1]),
        model_final_homophily=float(model_homophily[-1]),
        model_final_satisfaction=float(model_satisfaction[-1]),
        final_directional_lift=summary.final_directional_lift,
        directional_trajectory_area=summary.directional_trajectory_area,
        relative_final_homophily_reduction=(
            (float(reference_homophily[-1]) - float(model_homophily[-1]))
            / float(reference_homophily[-1])
        ),
        best_directional_lift=max(final_lift_values),
        reference_trajectory=MetricTrajectory(
            rounds=tuple(range(reference.rounds_completed + 1)),
            ordinary_homophily=reference_homophily,
            ordinary_satisfaction=reference_satisfaction,
        ),
        model_trajectory=MetricTrajectory(
            rounds=tuple(range(summary.rounds_completed + 1)),
            ordinary_homophily=tuple(float(value) for value in model_homophily),
            ordinary_satisfaction=tuple(float(value) for value in model_satisfaction),
        ),
        accepted_actor_moves=outcome_counts["accepted"],
        actor_stays=outcome_counts["stay"],
        actor_collisions=outcome_counts["collision"],
        occupied_destination_failures=outcome_counts["occupied"],
        policy_error_outcomes=outcome_counts["policy_error"],
        ordinary_moves=sum(ordinary_by_round),
        accepted_actor_moves_by_round=accepted_by_round,
        ordinary_moves_by_round=ordinary_by_round,
        coordination_post_count=len(posts),
        empty_coordination_posts=sum(not post.text.strip() for post in posts),
        median_coordination_characters=float(np.median(post_lengths)),
        coordination_posts_over_2000_characters=int(np.count_nonzero(post_lengths > 2000)),
        coordination_posts_over_5000_characters=int(np.count_nonzero(post_lengths > 5000)),
        longest_coordination_post_characters=int(post_lengths[longest_index]),
        longest_coordination_post_round=longest_round,
        longest_coordination_post_actor=longest_post.actor_id,
        collision_language_posts=sum(
            _contains_any(post.text, ("collision", "claim", "reserve", "conflict", "contested"))
            for post in posts
        ),
        hold_language_posts=sum(_contains_any(post.text, ("hold", "stay")) for post in posts),
        boundary_language_posts=sum(
            _contains_any(post.text, ("seam", "frontier", "boundary")) for post in posts
        ),
        anchor_language_posts=sum("anchor" in post.text.lower() for post in posts),
        opposite_type_language_posts=sum(
            _contains_any(post.text, ("opposite", "minority")) for post in posts
        ),
        total_requests=sum(int(event["request_count"]) for event in request_events),
        total_input_tokens=sum(int(event["input_tokens"]) for event in request_events),
        total_output_tokens=sum(int(event["output_tokens"]) for event in request_events),
    )
