"""Publish the compact Schelling resources shipped with MAM-Bench.

This command validates and aggregates the complete trajectory sweep, writes the
621-record ``landscape.jsonl``, generates the held-out Seed 50 counterfactual,
and writes its initial and terminal states as the evaluation fixture. It then
loads that fixture once to verify its archive hash and format.

Run this only after intentionally regenerating the full dataset. Its destination
is ``src/mam_bench/data/schelling-reference-v2`` and its three files are consumed
by ordinary benchmark installations.
"""

import argparse
from pathlib import Path

from mam_bench.simulations.schelling.fixture import load_evaluation_reference
from mam_bench.simulations.schelling.profile import LandscapeCell, Rational
from mam_bench.simulations.schelling.utils.analysis import analyze_landscape, write_landscape
from mam_bench.simulations.schelling.utils.reference_data import (
    build_evaluation_reference,
    write_evaluation_reference,
)

EXPECTED_OUTPUTS = {
    "evaluation-reference.json",
    "evaluation-reference.npz",
    "landscape.jsonl",
}


def publish_reference_data(full_dataset: Path, output_directory: Path) -> None:
    """Validate full trajectories and publish only evaluation-sized products."""

    if output_directory.exists():
        unexpected = {path.name for path in output_directory.iterdir()} - EXPECTED_OUTPUTS
        if unexpected:
            raise FileExistsError(
                f"compact output contains unexpected entries: {sorted(unexpected)}"
            )
    output_directory.mkdir(parents=True, exist_ok=True)
    write_landscape(analyze_landscape(full_dataset), output_directory / "landscape.jsonl")
    reference = build_evaluation_reference(LandscapeCell(20, Rational(3, 4), Rational(1, 4)), 50)
    write_evaluation_reference(output_directory, reference)
    load_evaluation_reference(output_directory)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full-dataset",
        type=Path,
        default=Path(".scratch/schelling-reference-v2-full"),
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("src/mam_bench/data/schelling-reference-v2"),
    )
    arguments = parser.parse_args()
    publish_reference_data(arguments.full_dataset, arguments.output_directory)


if __name__ == "__main__":
    main()
