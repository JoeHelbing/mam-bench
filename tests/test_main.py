import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

import main as application

from mam_bench.benchmark import BenchmarkTopline, EvidenceReceipt, PrimaryScore, ToplineEntry


class MainTests(unittest.TestCase):
    def test_requires_exactly_one_yaml_path(self) -> None:
        for arguments in ((), ("one.yaml", "two.yaml"), ("benchmark.yml",)):
            with self.subTest(arguments=arguments):
                stderr = StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as raised:
                    application.main(arguments)

                self.assertEqual(raised.exception.code, 2)
                self.assertTrue(stderr.getvalue())

    def test_unknown_simulation_fails_without_creating_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / "benchmark.yaml"
            config.write_text(
                """\
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
            stderr = StringIO()

            with redirect_stderr(stderr):
                result = application.main((str(config),))

            self.assertEqual(result, 2)
            self.assertIn("unknown-simulation", stderr.getvalue())
            self.assertFalse((root / "results").exists())

    def test_formats_scores_grouped_by_simulation(self) -> None:
        topline = BenchmarkTopline(
            entries=(
                self._entry("simulation-a", "score_a", 1.25, "points", "a"),
                self._entry("simulation-b", "score_b", 2.5, "ratio", "b"),
            )
        )

        self.assertEqual(
            application.format_topline(topline),
            "simulation-a\n"
            "  model-a  score_a = 1.250000 points\n"
            "\n"
            "simulation-b\n"
            "  model-a  score_b = 2.500000 ratio",
        )

    @staticmethod
    def _entry(
        simulation_id: str,
        score_name: str,
        score_value: float,
        unit: str,
        digest_character: str,
    ) -> ToplineEntry:
        return ToplineEntry(
            simulation_id=simulation_id,
            simulation_version=f"{simulation_id}-v1",
            model_id="model-a",
            runtime="fake",
            provider="test",
            model="test/model-a",
            primary_score=PrimaryScore(
                name=score_name,
                value=score_value,
                objective=f"increase {score_name}",
                meaning=f"{score_name} score",
                unit=unit,
                semantics_version=f"{score_name}-v1",
            ),
            evidence=EvidenceReceipt(
                relative_directory=f"runs/{simulation_id}/model-a",
                manifest_path="evidence-manifest.json",
                manifest_sha256=digest_character * 64,
            ),
        )


if __name__ == "__main__":
    unittest.main()
