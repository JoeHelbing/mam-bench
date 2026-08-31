import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from mam_bench.benchmark import BenchmarkTopline, EvidenceReceipt, PrimaryScore, ToplineEntry
from mam_bench.cli import format_grouped_topline


class CliTests(unittest.TestCase):
    def _run(self, *arguments: str, module: str = "mam_bench") -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
        return subprocess.run(
            [sys.executable, "-m", module, *arguments],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

    def test_profile_prints_the_frozen_reference_contract(self) -> None:
        completed = self._run("profile")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        profile = json.loads(completed.stdout)
        self.assertEqual(profile["profile_version"], "schelling-reference-v1")
        self.assertEqual(profile["grid_size"], 20)
        self.assertEqual(profile["max_transitions"], 500)
        self.assertEqual(len(profile["tolerances"]), 23)
        self.assertEqual(profile["vacancy_counts"], [40, 60, 80, 100, 120, 140, 160])

    def test_yaml_run_help_exposes_no_simulation_mechanics(self) -> None:
        completed = self._run("run", "--help")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("mam-bench.run.v1", completed.stdout)
        for forbidden in ("--rounds", "--objective", "--prompt", "--dataset"):
            self.assertNotIn(forbidden, completed.stdout)

    def test_yaml_run_rejects_unknown_simulation_without_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "benchmark.yaml"
            config.write_text(
                """\
schema_version: mam-bench.run.v1
simulations: [unknown-simulation]
models:
  - id: model-a
    runtime: openrouter
    model: provider/model
    provider: provider-a
output:
  directory: results
""",
                encoding="utf-8",
            )

            completed = self._run("run", str(config))

            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("unknown-simulation", completed.stderr)
            self.assertFalse((root / "results").exists())

    def test_yaml_run_formats_scores_grouped_by_simulation(self) -> None:
        topline = BenchmarkTopline(
            entries=(
                ToplineEntry(
                    simulation_id="simulation-a",
                    simulation_version="simulation-a-v1",
                    model_id="model-a",
                    runtime="fake",
                    provider="test",
                    model="test/model-a",
                    primary_score=PrimaryScore(
                        name="score_a",
                        value=1.25,
                        objective="increase a",
                        meaning="a score",
                        unit="points",
                        semantics_version="score-a-v1",
                    ),
                    evidence=EvidenceReceipt(
                        relative_directory="runs/simulation-a/model-a",
                        manifest_path="evidence-manifest.json",
                        manifest_sha256="a" * 64,
                    ),
                ),
                ToplineEntry(
                    simulation_id="simulation-b",
                    simulation_version="simulation-b-v1",
                    model_id="model-a",
                    runtime="fake",
                    provider="test",
                    model="test/model-a",
                    primary_score=PrimaryScore(
                        name="score_b",
                        value=2.5,
                        objective="increase b",
                        meaning="b score",
                        unit="ratio",
                        semantics_version="score-b-v1",
                    ),
                    evidence=EvidenceReceipt(
                        relative_directory="runs/simulation-b/model-a",
                        manifest_path="evidence-manifest.json",
                        manifest_sha256="b" * 64,
                    ),
                ),
            )
        )

        self.assertEqual(
            format_grouped_topline(topline),
            "simulation-a\n"
            "  model-a  score_a = 1.250000 points\n"
            "\n"
            "simulation-b\n"
            "  model-a  score_b = 2.500000 ratio",
        )

    def test_agent_pilot_help_exposes_the_fixed_live_case(self) -> None:
        completed = self._run("agent-pilot", "--help")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("3/4", completed.stdout)
        self.assertIn("Seed 22", completed.stdout)
        self.assertIn("--output-dir", completed.stdout)
        self.assertNotIn("--allow-openrouter-fallbacks", completed.stdout)

    def test_benchmark_cli_excludes_post_benchmark_analysis(self) -> None:
        completed = self._run("--help")

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertNotIn("satisfaction-manifold", completed.stdout)
        self.assertNotIn("benchmark-story", completed.stdout)

    def test_importing_runtime_cli_does_not_import_analysis_or_plotly(self) -> None:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
        completed = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import mam_bench.cli; "
                    "assert 'mam_bench_analysis' not in sys.modules; "
                    "assert 'plotly' not in sys.modules"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_analysis_cli_exposes_only_machine_readable_commands(self) -> None:
        reference = self._run("reference-landscape", "--help", module="mam_bench_analysis")
        self.assertEqual(reference.returncode, 0, reference.stderr)
        self.assertIn("--output-dir", reference.stdout)
        self.assertNotIn("--html", reference.stdout)
        self.assertNotIn("--evaluation-case", reference.stdout)

        evaluation = self._run("model-evaluation", "--help", module="mam_bench_analysis")
        self.assertEqual(evaluation.returncode, 0, evaluation.stderr)
        self.assertIn("--output", evaluation.stdout)

        top_level = self._run("--help", module="mam_bench_analysis")
        self.assertEqual(top_level.returncode, 0, top_level.stderr)
        self.assertIn("reference-landscape", top_level.stdout)
        self.assertIn("model-evaluation", top_level.stdout)
        self.assertNotIn("satisfaction-manifold", top_level.stdout)
        self.assertNotIn("benchmark-story", top_level.stdout)

    def test_analysis_cli_writes_reference_landscape_data(self) -> None:
        dataset = Path(__file__).parents[1] / "data" / "reference-landscape" / "v1"
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            completed = self._run(
                "reference-landscape",
                str(dataset),
                "--output-dir",
                str(output),
                module="mam_bench_analysis",
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(
                set(result),
                {"json", "csv", "selected_spots"},
            )
            self.assertTrue((output / "final-satisfaction-manifold.json").is_file())
            self.assertTrue((output / "final-satisfaction-manifold.csv").is_file())

    def test_generate_cell_writes_a_valid_canonical_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)

            completed = self._run(
                "generate-cell",
                "--tolerance-index",
                "0",
                "--vacancy-index",
                "3",
                "--output-dir",
                str(output),
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            record = json.loads(completed.stdout)
            artifact = output / record["path"]
            self.assertTrue(artifact.is_file())
            self.assertEqual(record["tolerance_index"], 0)
            self.assertEqual(record["vacancy_index"], 3)

            validated = self._run(
                "validate-cell",
                str(artifact),
                "--tolerance-index",
                "0",
                "--vacancy-index",
                "3",
            )
            self.assertEqual(validated.returncode, 0, validated.stderr)
            self.assertIn("valid", validated.stdout)


if __name__ == "__main__":
    unittest.main()
