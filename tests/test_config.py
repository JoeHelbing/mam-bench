import tempfile
import unittest
from pathlib import Path

from mam_bench.config import BenchmarkConfigError, load_benchmark_config
from mam_bench.registry import RegistryPreflightError, resolve_benchmark_plan
from mam_bench.runner import CompatibilityPreflightError, preflight_benchmark

VALID_YAML = """\
schema_version: mam-bench.run.v1
simulations:
  - schelling-influence-pilot-v1
models:
  - id: qwen-3-8-27b
    runtime: openrouter
    model: qwen/qwen3.8-27b
    provider: phala
  - id: local-model
    runtime: openai-compatible
    model: local/model
    base_url: http://127.0.0.1:8000/v1
    api_key_env: LOCAL_MODEL_API_KEY
output:
  directory: results/comparison-001
  retain_diagnostic_artifacts: false
"""


class BenchmarkConfigTests(unittest.TestCase):
    def _write(self, directory: str, content: str) -> Path:
        path = Path(directory) / "benchmark.yaml"
        path.write_text(content, encoding="utf-8")
        return path

    def test_loads_only_selection_and_resolves_output_relative_to_yaml(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, VALID_YAML)

            config = load_benchmark_config(path)

            self.assertEqual(config.schema_version, "mam-bench.run.v1")
            self.assertEqual(config.simulations, ("schelling-influence-pilot-v1",))
            self.assertEqual(
                tuple(model.id for model in config.models),
                (
                    "qwen-3-8-27b",
                    "local-model",
                ),
            )
            self.assertEqual(
                config.output.directory,
                Path(directory).resolve() / "results" / "comparison-001",
            )
            self.assertFalse(config.output.directory.exists())

    def test_rejects_duplicate_keys_aliases_merges_tags_and_documents(self) -> None:
        invalid_documents = {
            "duplicate key": VALID_YAML.replace(
                "schema_version: mam-bench.run.v1",
                "schema_version: mam-bench.run.v1\nschema_version: mam-bench.run.v1",
            ),
            "alias": VALID_YAML.replace(
                "simulations:\n  - schelling-influence-pilot-v1",
                "simulations: &simulations\n  - schelling-influence-pilot-v1\ncopy: *simulations",
            ),
            "merge": VALID_YAML.replace(
                "output:\n  directory: results/comparison-001",
                "defaults: &defaults\n"
                "  retain_diagnostic_artifacts: false\n"
                "output:\n"
                "  <<: *defaults\n"
                "  directory: results/comparison-001",
            ),
            "custom tag": VALID_YAML.replace(
                "schema_version: mam-bench.run.v1",
                "schema_version: !custom mam-bench.run.v1",
            ),
            "multiple documents": f"{VALID_YAML}---\n{VALID_YAML}",
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, content in invalid_documents.items():
                with self.subTest(name=name), self.assertRaises(BenchmarkConfigError):
                    load_benchmark_config(self._write(directory, content))

    def test_rejects_unsafe_or_duplicate_selections_and_nonempty_output(self) -> None:
        invalid_documents = {
            "unsafe model id": VALID_YAML.replace("id: qwen-3-8-27b", "id: ../model"),
            "duplicate simulation": VALID_YAML.replace(
                "  - schelling-influence-pilot-v1\nmodels:",
                "  - schelling-influence-pilot-v1\n  - schelling-influence-pilot-v1\nmodels:",
            ),
            "duplicate model": VALID_YAML.replace("id: local-model", "id: qwen-3-8-27b"),
            "unsafe environment": VALID_YAML.replace(
                "LOCAL_MODEL_API_KEY", "not-an-environment-name"
            ),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, content in invalid_documents.items():
                with self.subTest(name=name), self.assertRaises(BenchmarkConfigError):
                    load_benchmark_config(self._write(directory, content))

            credential_url = VALID_YAML.replace(
                "http://127.0.0.1:8000/v1",
                "https://user:do-not-store@example.test/v1",
            )
            with self.assertRaises(BenchmarkConfigError) as caught:
                load_benchmark_config(self._write(directory, credential_url))
            self.assertNotIn("do-not-store", str(caught.exception))

            path = self._write(directory, VALID_YAML)
            output = Path(directory) / "results" / "comparison-001"
            output.mkdir(parents=True)
            (output / "existing.txt").write_text("occupied", encoding="utf-8")
            with self.assertRaises(BenchmarkConfigError):
                load_benchmark_config(path)

    def test_rejects_mechanics_and_unknown_runtime_fields(self) -> None:
        invalid_documents = {
            "root mechanics": VALID_YAML.replace(
                "output:\n", "rounds: 20\nobjective: integration\noutput:\n"
            ),
            "model prompt": VALID_YAML.replace(
                "    provider: phala", "    provider: phala\n    prompt: steer"
            ),
            "generic parameters": VALID_YAML.replace(
                "    provider: phala", "    provider: phala\n    parameters: {}"
            ),
            "unknown runtime": VALID_YAML.replace("runtime: openrouter", "runtime: python-plugin"),
        }
        with tempfile.TemporaryDirectory() as directory:
            for name, content in invalid_documents.items():
                with self.subTest(name=name), self.assertRaises(BenchmarkConfigError):
                    load_benchmark_config(self._write(directory, content))

    def test_reports_every_unknown_runtime_before_output(self) -> None:
        content = VALID_YAML.replace("runtime: openrouter", "runtime: unknown-one").replace(
            "runtime: openai-compatible", "runtime: unknown-two"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = self._write(directory, content)

            with self.assertRaises(BenchmarkConfigError) as caught:
                load_benchmark_config(path)

            self.assertTrue(any("models.0" in error for error in caught.exception.errors))
            self.assertTrue(any("models.1" in error for error in caught.exception.errors))
            self.assertFalse((Path(directory) / "results").exists())

    def test_requires_nonempty_lists_and_existing_environment_names_only(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            no_simulations = VALID_YAML.replace(
                "simulations:\n  - schelling-influence-pilot-v1", "simulations: []"
            )
            with self.assertRaises(BenchmarkConfigError):
                load_benchmark_config(self._write(directory, no_simulations))

            no_models = VALID_YAML.replace(
                VALID_YAML[VALID_YAML.index("models:") : VALID_YAML.index("output:")],
                "models: []\n",
            )
            with self.assertRaises(BenchmarkConfigError):
                load_benchmark_config(self._write(directory, no_models))


class RegistryTests(unittest.TestCase):
    def test_resolves_builtins_in_deterministic_matrix_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.yaml"
            path.write_text(VALID_YAML, encoding="utf-8")
            config = load_benchmark_config(path)

            plan = resolve_benchmark_plan(config)

            self.assertEqual(plan.simulations, ("schelling-influence-pilot-v1",))
            self.assertEqual(plan.runtimes, ("qwen-3-8-27b", "local-model"))
            self.assertEqual(tuple(plan.simulation_registry), plan.simulations)
            self.assertEqual(tuple(plan.runtime_factories), plan.runtimes)
            self.assertEqual(
                plan.runtime_factories["qwen-3-8-27b"].descriptor.required_environment,
                ("OPENROUTER_API_KEY",),
            )
            self.assertEqual(
                plan.runtime_factories["local-model"].descriptor.required_environment,
                ("LOCAL_MODEL_API_KEY",),
            )

    def test_complete_preflight_reports_every_missing_environment_without_output(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.yaml"
            path.write_text(VALID_YAML, encoding="utf-8")
            config = load_benchmark_config(path)
            resolved = resolve_benchmark_plan(config)

            with self.assertRaises(CompatibilityPreflightError) as caught:
                preflight_benchmark(
                    resolved.selection,
                    simulations=resolved.simulation_registry,
                    runtimes=resolved.runtime_factories,
                    environment={},
                )

            self.assertEqual(
                [issue.code for issue in caught.exception.issues],
                [
                    "missing-environment:OPENROUTER_API_KEY",
                    "missing-environment:LOCAL_MODEL_API_KEY",
                ],
            )
            self.assertFalse(config.output.directory.exists())

    def test_reports_every_unknown_simulation_without_constructing_output(self) -> None:
        content = VALID_YAML.replace(
            "  - schelling-influence-pilot-v1",
            "  - unknown-one\n  - unknown-two",
            1,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.yaml"
            path.write_text(content, encoding="utf-8")
            config = load_benchmark_config(path)

            with self.assertRaises(RegistryPreflightError) as caught:
                resolve_benchmark_plan(config)

            self.assertEqual(
                tuple(issue.simulation_id for issue in caught.exception.issues),
                ("unknown-one", "unknown-two"),
            )
            self.assertFalse(config.output.directory.exists())


if __name__ == "__main__":
    unittest.main()
