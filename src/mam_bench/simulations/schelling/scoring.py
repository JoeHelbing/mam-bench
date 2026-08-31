"""Schelling Primary Score inputs and simulation-native metrics."""

from .evaluation import (
    CounterfactualReference,
    ModelEvaluationSummary,
    build_counterfactual_reference,
    ordinary_edge_homophily,
    ordinary_satisfaction_fraction,
)

__all__ = [
    "CounterfactualReference",
    "ModelEvaluationSummary",
    "build_counterfactual_reference",
    "ordinary_edge_homophily",
    "ordinary_satisfaction_fraction",
]
