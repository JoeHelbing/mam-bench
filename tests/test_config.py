import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from mam_bench.config import load_benchmark_config


class BenchmarkConfigTests(unittest.TestCase):
    def test_loads_both_provider_paths_and_resolves_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "benchmark.yaml"
            config_path.write_text(
                """simulations:
  - schelling-influence-pilot-v1
models:
  - id: openrouter-model
    runtime: openrouter
    model: vendor/model
    provider: provider-a
  - id: local-model
    runtime: openai-compatible
    model: local/model
    base_url: http://127.0.0.1:8000/v1
    api_key_env: LOCAL_API_KEY
output_directory: results
""",
                encoding="utf-8",
            )

            config = load_benchmark_config(config_path)

            self.assertEqual(len(config.models), 2)
            self.assertEqual(config.models[0].runtime, "openrouter")
            self.assertEqual(config.models[1].runtime, "openai-compatible")
            self.assertEqual(config.output_directory, root / "results")

    def test_rejects_duplicate_model_ids(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.yaml"
            path.write_text(
                """simulations: [schelling-influence-pilot-v1]
models:
  - {id: duplicate, runtime: openrouter, model: vendor/a, provider: provider-a}
  - {id: duplicate, runtime: openrouter, model: vendor/b, provider: provider-b}
output_directory: results
""",
                encoding="utf-8",
            )

            with self.assertRaises(ValidationError):
                load_benchmark_config(path)

    def test_rejects_credentials_in_openai_compatible_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.yaml"
            path.write_text(
                """simulations: [schelling-influence-pilot-v1]
models:
  - id: unsafe
    runtime: openai-compatible
    model: local/model
    base_url: https://user:password@example.test/v1
    api_key_env: LOCAL_API_KEY
output_directory: results
""",
                encoding="utf-8",
            )

            with self.assertRaises(ValidationError):
                load_benchmark_config(path)


if __name__ == "__main__":
    unittest.main()
