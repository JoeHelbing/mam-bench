import tempfile
import unittest
from contextlib import chdir
from pathlib import Path

from pydantic import ValidationError

from mam_bench.config import (
    AgentSettings,
    BenchmarkConfig,
    OpenAICompatibleModel,
    OpenRouterModel,
    load_benchmark_config,
)
from mam_bench.simulations.schelling.settings import SchellingSettings


class BenchmarkConfigTests(unittest.TestCase):
    def test_loads_schelling_parameters_and_preserves_fraction_seed_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "benchmark.yaml"
            path.write_text(
                "simulations: [schelling-influence-pilot-v1]\n"
                "models: [{id: test, runtime: openrouter, model: test, provider: test}]\n"
                "output_directory: results\n"
                "schelling:\n"
                "  board_size: 12\n"
                "  tolerance: 6/8\n"
                "  vacancy_fraction: 2/8\n"
                "  seed_id: 9\n"
                "  max_transitions: 45\n"
                "  vision_radius: 2\n"
                "  controlled_agent_count: 4\n"
                "  objective: segregation\n",
                encoding="utf-8",
            )
            settings = load_benchmark_config(path).schelling
        self.assertEqual(settings.board_size, 12)
        self.assertEqual(settings.seed_id, 9)
        self.assertEqual(settings.max_transitions, 45)
        self.assertEqual(settings.vision_radius, 2)
        self.assertEqual(settings.objective, "segregation")
        self.assertEqual(str(settings.cell.tolerance), "6/8")
        self.assertEqual(str(settings.cell.vacancy_fraction), "2/8")
        self.assertEqual(settings.cell.agent_count, 108)
        self.assertEqual(settings.controlled_agent_ids, (0, 1, 54, 55))

    def test_existing_yaml_selections_receive_radius_three_defaults(self) -> None:
        config = BenchmarkConfig.model_validate(
            {
                "simulations": ["schelling-influence-pilot-v1"],
                "models": [
                    {"id": "test", "runtime": "openrouter", "model": "test", "provider": "test"}
                ],
                "output_directory": "results",
            }
        )
        self.assertEqual(config.schelling, SchellingSettings())
        self.assertEqual(config.schelling.vision_radius, 3)
        self.assertEqual(config.schelling.cell.agent_count, 300)
        self.assertEqual(config.schelling.controlled_agent_ids, (*range(8), *range(150, 158)))

    def test_rejects_invalid_schelling_parameters(self) -> None:
        for values in (
            {"board_size": 2},
            {"board_size": 256},
            {"board_size": 20.0},
            {"tolerance": "3/0"},
            {"tolerance": "-1/4"},
            {"tolerance": "5/4"},
            {"tolerance": "0.75"},
            {"tolerance": "3/4/5"},
            {"tolerance": 0.75},
            {"tolerance": "1/1000000001"},
            {"vacancy_fraction": "1/1"},
            {"vacancy_fraction": "0/1"},
            {"vacancy_fraction": "1/401"},
            {"vacancy_fraction": "1/400"},
            {"seed_id": -1},
            {"seed_id": True},
            {"seed_id": "50"},
            {"max_transitions": 0},
            {"vision_radius": 0},
            {"vision_radius": 10},
            {"controlled_agent_count": 1},
            {"controlled_agent_count": 3},
            {"controlled_agent_count": 300},
            {"board_size": 7, "vacancy_fraction": "1/49", "controlled_agent_count": 46},
            {"vacancy_fraction": "3/4", "controlled_agent_count": 2},
            {"objective": "unknown"},
            {"vision_raduis": 3},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                SchellingSettings.model_validate(values)

    def test_output_paths_use_working_directory_and_preserve_absolute_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            configs = root / "configs"
            configs.mkdir()
            working = root / "working"
            working.mkdir()
            path = configs / "benchmark.yaml"
            for value, expected in (
                ("results/run", working / "results/run"),
                ("../results/run", root / "results/run"),
                (str(root / "absolute/run"), root / "absolute/run"),
            ):
                with self.subTest(output=value):
                    path.write_text(
                        "simulations: [schelling-influence-pilot-v1]\n"
                        "models: [{id: test, runtime: openrouter, "
                        "model: vendor/model, provider: test}]\n"
                        f"output_directory: {value}\n",
                        encoding="utf-8",
                    )
                    with chdir(working):
                        for config_path in (path, Path("../configs/benchmark.yaml")):
                            self.assertEqual(
                                load_benchmark_config(config_path).output_directory, expected
                            )

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
    settings:
      concurrency: 16
      tool_choice: auto
      context_window_tokens: 65536
      max_completion_tokens: 8192
      summary_completion_tokens: 8192
  - id: local-model
    runtime: openai-compatible
    model: local/model
    base_url: http://127.0.0.1:8000/v1
    api_key_env: LOCAL_API_KEY
output_directory: results
log_level: DEBUG
""",
                encoding="utf-8",
            )

            invocation_directory = root / "working-directory"
            invocation_directory.mkdir()
            with chdir(invocation_directory):
                config = load_benchmark_config(config_path)

            self.assertEqual(len(config.models), 2)
            self.assertEqual(config.models[0].runtime, "openrouter")
            self.assertEqual(config.models[1].runtime, "openai-compatible")
            self.assertEqual(config.output_directory, invocation_directory / "results")
            self.assertEqual(config.log_level, "DEBUG")
            self.assertEqual(config.models[0].settings.concurrency, 16)
            self.assertEqual(config.models[0].settings.tool_choice, "auto")
            self.assertEqual(config.models[1].settings.tool_choice, "required")
            self.assertEqual(config.models[0].settings.context_window_tokens, 65536)
            self.assertEqual(config.models[1].settings, AgentSettings())

    def test_rejects_invalid_settings_for_either_provider(self) -> None:
        invalid_settings = (
            {"tool_choice": "none"},
            {"concurency": 16},
            {"concurrency": 0},
            {"temperature": -1},
            {"temperature": float("nan")},
            {"top_p": 1.1},
            {"top_k": -1},
            {"timeout_seconds": 0},
            {"timeout_seconds": float("inf")},
            {"max_completion_tokens": 0},
            {"summary_completion_tokens": 0},
            {"memory_injection_tokens": 0},
            {"context_window_tokens": 1000},
            {"compaction_trigger_fraction": 1},
            {"reasoning_effort": "unknown"},
        )
        for settings in invalid_settings:
            for selection in (
                {"runtime": "openrouter", "provider": "provider-a"},
                {
                    "runtime": "openai-compatible",
                    "base_url": "http://localhost/v1",
                    "api_key_env": "LOCAL_API_KEY",
                },
            ):
                with self.subTest(settings=settings, selection=selection):
                    cls = (
                        OpenRouterModel
                        if selection["runtime"] == "openrouter"
                        else OpenAICompatibleModel
                    )
                    with self.assertRaises(ValidationError):
                        cls.model_validate(
                            {"id": "test", "model": "test", **selection, "settings": settings}
                        )

    def test_names_do_not_require_regex_spelling(self) -> None:
        router = OpenRouterModel(
            id="My Model", runtime="openrouter", model="vendor/model", provider="Provider/Region"
        )
        local = OpenAICompatibleModel(
            id="Local Model",
            runtime="openai-compatible",
            model="local/model",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="local.api-key",
        )
        self.assertEqual(router.provider, "Provider/Region")
        self.assertEqual(local.api_key_env, "local.api-key")

    def test_model_values_are_left_to_live_preflight(self) -> None:
        router = OpenRouterModel(id="router", runtime="openrouter", model="", provider="")
        local = OpenAICompatibleModel(
            id="local",
            runtime="openai-compatible",
            model="",
            base_url="http://127.0.0.1:8000/v1",
            api_key_env="",
        )
        self.assertEqual(router.model, "")
        self.assertEqual(router.provider, "")
        self.assertEqual(local.model, "")
        self.assertEqual(local.api_key_env, "")

    def test_rejects_model_ids_that_are_not_directory_components(self) -> None:
        for model_id in (
            "",
            ".",
            "..",
            "../escape",
            "/absolute",
            "a/b",
            "a\\b",
            "C:escape",
            "a\x00b",
        ):
            with self.subTest(model_id=model_id):
                for runtime in ("openrouter", "openai-compatible"):
                    with self.subTest(runtime=runtime), self.assertRaises(ValidationError):
                        if runtime == "openrouter":
                            OpenRouterModel(
                                id=model_id,
                                runtime=runtime,
                                model="vendor/a",
                                provider="provider-a",
                            )
                        else:
                            OpenAICompatibleModel(
                                id=model_id,
                                runtime=runtime,
                                model="local/a",
                                base_url="http://127.0.0.1:8000/v1",
                                api_key_env="LOCAL_API_KEY",
                            )

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

    def test_preserves_base_url_without_local_validation(self) -> None:
        for base_url in (
            "http://127.0.0.1:8000/v1/",
            "https://example.test/v1?region=test#fragment",
            "relative/endpoint",
            "ftp://example.test/v1",
            "",
        ):
            with self.subTest(base_url=base_url):
                selection = OpenAICompatibleModel(
                    id="local-model",
                    runtime="openai-compatible",
                    model="local/model",
                    base_url=base_url,
                    api_key_env="LOCAL_API_KEY",
                )
                self.assertEqual(selection.base_url, base_url)


if __name__ == "__main__":
    unittest.main()
