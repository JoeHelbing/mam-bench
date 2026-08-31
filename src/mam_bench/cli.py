"""Command-line interface for Reference Landscape generation and validation."""

import argparse
import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

from pydantic import ValidationError

from mam_bench.benchmark import BenchmarkTopline
from mam_bench.config import load_benchmark_config
from mam_bench.pydantic_runtime import PydanticModelRuntime, PydanticRuntimeSettings
from mam_bench.registry import resolve_benchmark_plan
from mam_bench.runner import BenchmarkError, preflight_benchmark, run_benchmark
from mam_bench.runtime import RuntimeInfrastructureError
from mam_bench.simulations.schelling import SchellingBenchmarkSimulation
from mam_bench.simulations.schelling.dataset import (
    PartialSweep,
    build_cell_archive,
    finalize_dataset,
    validate_cell_archive,
    validate_dataset,
    write_artifact_atomically,
)
from mam_bench.simulations.schelling.evaluation import InfluenceInfrastructureError
from mam_bench.simulations.schelling.evidence import validate_evaluation_artifact
from mam_bench.simulations.schelling.profile import PROFILE, landscape_cell


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="mam-bench")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("profile", help="print the frozen Reference Profile as JSON")

    run = commands.add_parser(
        "run",
        help="run one selection-only YAML benchmark matrix",
    )
    run.add_argument("config", type=Path, help="path to mam-bench.run.v1 YAML")

    generate = commands.add_parser(
        "generate-cell", help="generate one canonical 20-seed Landscape Cell artifact"
    )
    generate.add_argument("--tolerance-index", type=int, required=True)
    generate.add_argument("--vacancy-index", type=int, required=True)
    generate.add_argument("--output-dir", type=Path, required=True)

    validate_cell = commands.add_parser("validate-cell", help="validate one NPZ cell artifact")
    validate_cell.add_argument("artifact", type=Path)
    validate_cell.add_argument("--tolerance-index", type=int, required=True)
    validate_cell.add_argument("--vacancy-index", type=int, required=True)

    finalize = commands.add_parser(
        "finalize-dataset",
        help="validate all Modal artifacts and publish the complete manifest",
    )
    finalize.add_argument("dataset", type=Path)
    finalize.add_argument(
        "--partial",
        type=Path,
        help="partial receipt path (defaults to DATASET/.partial-artifacts.json)",
    )

    validate_all = commands.add_parser(
        "validate-dataset", help="validate a complete manifested Reference Landscape"
    )
    validate_all.add_argument("dataset", type=Path)

    pilot = commands.add_parser(
        "agent-pilot",
        help="run the fixed 3/4 preference, 25 percent empty, Seed 22 integration pilot",
        description=(
            "Run the fixed live Influence Profile pilot: 3/4 preference, 25% empty, "
            "Evaluation Seed 22, integration objective."
        ),
    )
    pilot.add_argument("--output-dir", type=Path, required=True)
    pilot.add_argument(
        "--provider",
        choices=("openrouter", "openai-compatible"),
        default="openrouter",
    )
    pilot.add_argument(
        "--base-url",
        default="https://openrouter.ai/api/v1",
        help="recorded model API base URL",
    )
    pilot.add_argument("--model-name", default="qwen/qwen3.8-27b")
    pilot.add_argument("--openrouter-provider", default="phala")
    validate_evaluation = commands.add_parser(
        "validate-evaluation",
        help="reload and recompute one complete Influence Profile case artifact",
    )
    validate_evaluation.add_argument("case", type=Path)
    return parser


def _profile_record() -> dict[str, object]:
    record = PROFILE.model_dump(mode="json")
    record["tolerances"] = [str(value) for value in PROFILE.tolerances]
    record["vacancy_fractions"] = [str(value) for value in PROFILE.vacancy_fractions]
    return record


def format_grouped_topline(topline: BenchmarkTopline) -> str:
    """Format simulation-local scores without cross-simulation aggregation."""

    lines: list[str] = []
    current_simulation: str | None = None
    for entry in topline.entries:
        if entry.simulation_id != current_simulation:
            if lines:
                lines.append("")
            lines.append(entry.simulation_id)
            current_simulation = entry.simulation_id
        lines.append(
            f"  {entry.model_id}  {entry.primary_score.name} = "
            f"{entry.primary_score.value:.6f} {entry.primary_score.unit}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "profile":
            print(json.dumps(_profile_record(), indent=2, sort_keys=True))
        elif args.command == "run":
            config = load_benchmark_config(args.config)
            resolved = resolve_benchmark_plan(config)
            prepared = preflight_benchmark(
                resolved.selection,
                simulations=resolved.simulation_registry,
                runtimes=resolved.runtime_factories,
            )
            topline = asyncio.run(run_benchmark(prepared))
            print(format_grouped_topline(topline))
        elif args.command == "generate-cell":
            cell = landscape_cell(args.tolerance_index, args.vacancy_index)
            archive = build_cell_archive(cell)
            record = write_artifact_atomically(args.output_dir, cell, archive)
            print(record.model_dump_json(indent=2))
        elif args.command == "validate-cell":
            cell = landscape_cell(args.tolerance_index, args.vacancy_index)
            validate_cell_archive(args.artifact.read_bytes(), cell)
            print(f"valid: {args.artifact}")
        elif args.command == "finalize-dataset":
            partial_path = args.partial or args.dataset / ".partial-artifacts.json"
            partial = PartialSweep.model_validate_json(partial_path.read_text(encoding="utf-8"))
            manifest = finalize_dataset(
                args.dataset,
                list(partial.artifacts),
                modal_client_version=partial.modal_client_version,
                generator_versions=partial.generator_versions,
            )
            print(f"finalized: {len(manifest.artifacts)} cells, {manifest.expected_run_count} runs")
        elif args.command == "validate-dataset":
            manifest = validate_dataset(args.dataset)
            print(f"valid: {len(manifest.artifacts)} cells, {manifest.expected_run_count} runs")
        elif args.command == "agent-pilot":
            simulation = SchellingBenchmarkSimulation()
            prepared = simulation.prepare()
            runtime = PydanticModelRuntime(
                PydanticRuntimeSettings(
                    provider=args.provider,
                    base_url=args.base_url,
                    model_name=args.model_name,
                    credential_environment=(
                        "OPENROUTER_API_KEY" if args.provider == "openrouter" else "OPENAI_API_KEY"
                    ),
                    openrouter_provider_slug=args.openrouter_provider,
                )
            )
            asyncio.run(
                prepared.execute(
                    runtime=runtime,
                    output_directory=args.output_dir,
                    retain_diagnostic_artifacts=True,
                )
            )
            result = simulation.validate(
                output_directory=args.output_dir,
                runtime=runtime.descriptor,
            )
            print(result.model_dump_json(indent=2))
        elif args.command == "validate-evaluation":
            summary = validate_evaluation_artifact(args.case)
            print(summary.model_dump_json(indent=2))
        else:
            parser.error(f"unsupported command: {args.command}")
    except (
        BenchmarkError,
        InfluenceInfrastructureError,
        OSError,
        RuntimeInfrastructureError,
        ValueError,
        ValidationError,
    ) as exc:
        parser.error(str(exc))
    return 0
