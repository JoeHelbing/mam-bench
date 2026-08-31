"""Command-line interface for machine-readable post-benchmark analysis."""

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from .analysis import (
    analyze_final_satisfaction,
    write_reference_landscape_analysis,
)
from .evaluation_analysis import (
    analyze_model_evaluation,
    write_model_evaluation_analysis,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mam-bench-analysis")
    commands = parser.add_subparsers(dest="command", required=True)

    reference = commands.add_parser(
        "reference-landscape",
        help="derive machine-readable statistics from a validated Reference Dataset",
    )
    reference.add_argument("dataset", type=Path)
    reference.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="directory for the derived JSON and CSV artifacts",
    )

    evaluation = commands.add_parser(
        "model-evaluation",
        help="derive machine-readable diagnostics from one validated evaluation case",
    )
    evaluation.add_argument("case", type=Path)
    evaluation.add_argument(
        "--output",
        type=Path,
        required=True,
        help="path for the derived JSON artifact",
    )
    return parser


def _analyze_reference_landscape(args: argparse.Namespace) -> None:
    analysis = analyze_final_satisfaction(args.dataset)
    json_path, csv_path = write_reference_landscape_analysis(
        analysis,
        args.output_dir,
    )
    print(
        json.dumps(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "selected_spots": [
                    spot.model_dump(mode="json") for spot in analysis.selected_spots
                ],
            },
            indent=2,
        )
    )


def _analyze_model_evaluation(args: argparse.Namespace) -> None:
    analysis = analyze_model_evaluation(args.case)
    output_path = write_model_evaluation_analysis(analysis, args.output)
    print(
        json.dumps(
            {
                "json": str(output_path),
                "model": analysis.model_name,
                "objective": analysis.objective,
                "final_directional_lift": analysis.final_directional_lift,
            },
            indent=2,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "reference-landscape":
            _analyze_reference_landscape(args)
        elif args.command == "model-evaluation":
            _analyze_model_evaluation(args)
        else:
            parser.error(f"unsupported command: {args.command}")
    except (OSError, RuntimeError, ValueError, ValidationError) as exc:
        parser.error(str(exc))
    return 0
