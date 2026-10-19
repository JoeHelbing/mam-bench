"""Complete upfront validation and custom-suite replacement."""

import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from mam_bench.config import (
    AgentSettings,
    BenchmarkConfig,
    RepetitionDetection,
    load_benchmark_config,
)
from support import civil, schelling, selection


class ConfigurationTests(unittest.TestCase):
    def test_model_yaml_keeps_connections_specific_to_the_selected_service(self) -> None:
        valid = (
            {"runtime": "openai", "model": "example"},
            {"runtime": "openrouter", "model": "vendor/model", "provider": "provider"},
            selection().model_dump(),
        )
        for model in valid:
            config = BenchmarkConfig.model_validate(
                {"model": model, "cases": [schelling()], "output_directory": "unused"}
            )
            dumped = config.model.model_dump()
            self.assertEqual(dumped, {**model, "settings": config.model.settings.model_dump()})
            unexpected = {"base_url", "api_key_env", "provider", "repetition_detection"} - set(
                model
            )
            if model["runtime"] == "vllm":
                unexpected.discard("repetition_detection")
            for field in unexpected:
                with (
                    self.subTest(runtime=model["runtime"], field=field),
                    self.assertRaises(ValidationError),
                ):
                    BenchmarkConfig.model_validate(
                        {
                            "model": {**model, field: None},
                            "cases": [schelling()],
                            "output_directory": "unused",
                        }
                    )

    def test_repetition_detection_rejects_invalid_pattern_limits(self) -> None:
        for values in (
            {"min_pattern_size": 0},
            {"min_pattern_size": 129},
            {"max_pattern_size": 0},
            {"min_count": 1},
            {"min_count": True},
            {"min_count": "8"},
            {"max_pattern_size": 128.5},
            {"unknown_setting": 1},
        ):
            with self.subTest(values=values), self.assertRaises(ValidationError):
                RepetitionDetection.model_validate(values)

    def test_required_case_fields_no_defaults_and_no_extra_parameters(self) -> None:
        for case in (schelling(), civil()):
            complete = case.model_dump()
            for field in complete:
                with self.subTest(simulation=case.simulation, field=field):
                    incomplete = {key: value for key, value in complete.items() if key != field}
                    with self.assertRaises(ValidationError):
                        BenchmarkConfig.model_validate(
                            {
                                "model": selection().model_dump(),
                                "cases": [incomplete],
                                "output_directory": "unused",
                            }
                        )
            with self.assertRaises(ValidationError):
                type(case).model_validate({**complete, "obsolete_override": 1})

    def test_population_and_role_constraints(self) -> None:
        for updates in (
            {"controlled_role": "police"},
            {"controlled_agent_count": 2},
            {"citizen_density": 0.25},
            {"citizen_density": 0.8, "police_density": 0.3},
            {"private_preference_std": -1},
            {"vision_radius": 0},
            {"vision_radius": True},
            {"citizen_vision": 1, "police_vision": 1},
            {"threshold": float("nan")},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                civil(**updates)
        for max_steps in (1, 7, 30, 60):
            with self.subTest(max_steps=max_steps):
                self.assertEqual(civil(max_steps=max_steps).max_steps, max_steps)
        for max_steps in (0, -1, True, "30", 1.5):
            with self.subTest(max_steps=max_steps), self.assertRaises(ValidationError):
                civil(max_steps=max_steps)
        for updates in (
            {"controlled_agent_count": 3},
            {"controlled_agent_count": 40},
            {"vision_radius": 4},
            {"vacancy_fraction": 1.0},
            {"max_steps": 0},
            {"seed": -1},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                schelling(**updates)

    def test_custom_suite_replaces_cases_preserves_order_and_explicit_model(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.yaml"
            suite = root / "suite.yaml"
            model.write_text(yaml.safe_dump(selection().model_dump()))
            cases = [
                civil(max_steps=7).model_dump(),
                schelling(seed=19).model_dump(),
                civil().model_dump(),
            ]
            suite.write_text(yaml.safe_dump({"cases": cases}))
            config = load_benchmark_config(model, suite, output_directory=Path("results/custom"))
            self.assertEqual([c.model_dump() for c in config.cases], cases)
            self.assertEqual(config.output_directory, Path("results/custom").resolve())
            with self.assertRaises(ValidationError):
                BenchmarkConfig.model_validate({"cases": cases, "output_directory": "unused"})
            self.assertEqual(len(load_benchmark_config(model, output_directory=root).cases), 6)

    def test_shipped_default_is_six_ordered_mirrored_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model.yaml"
            model.write_text(yaml.safe_dump(selection().model_dump()))
            config = load_benchmark_config(model, output_directory=root)

        shared = {"board_size": 20, "max_steps": 30, "controlled_agent_count": 16}
        schelling_settings = {
            **shared,
            "simulation": "schelling",
            "tolerance": 0.7,
            "vacancy_fraction": 0.15,
            "vision_radius": 2,
            "seed": 10002,
        }
        citizen_settings = {
            **shared,
            "simulation": "civil-violence",
            "controlled_role": "citizen",
            "citizen_density": 0.54,
            "police_density": 0.055,
            "vision_radius": 2,
            "threshold": 0.0,
            "private_preference_mean": 0.0,
            "private_preference_std": 0.7,
            "max_jail_term": 12,
            "seed": 1000,
        }
        police_settings = {
            **citizen_settings,
            "controlled_role": "police",
            "police_density": 0.05,
            "threshold": 1.0,
            "max_jail_term": 6,
            "seed": 53000,
        }
        expected = [
            {**settings, "objective": objective}
            for settings, objectives in (
                (schelling_settings, ("integration", "segregation")),
                (citizen_settings, ("increase", "decrease")),
                (police_settings, ("increase", "decrease")),
            )
            for objective in objectives
        ]
        self.assertEqual([case.model_dump() for case in config.cases], expected)

    def test_model_yaml_selects_strict_parameterless_tools_per_run(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path, suite_path = root / "model.yaml", root / "suite.yaml"
            suite_path.write_text(yaml.safe_dump({"cases": [schelling().model_dump()]}))
            model = selection().model_dump()
            model["settings"] = {"tool_choice": "auto", "strict_parameterless_tools": True}
            model_path.write_text(yaml.safe_dump(model))
            loaded = load_benchmark_config(model_path, suite_path, output_directory=root)
            self.assertEqual(loaded.model.settings.tool_choice, "auto")
            self.assertTrue(loaded.model.settings.strict_parameterless_tools)

            model["settings"] = {"tool_choice": "required"}
            model_path.write_text(yaml.safe_dump(model))
            loaded = load_benchmark_config(model_path, suite_path, output_directory=root)
            self.assertFalse(loaded.model.settings.strict_parameterless_tools)

            for invalid in ("true", 1):
                with self.subTest(invalid=invalid), self.assertRaises(ValidationError):
                    model["settings"] = {"strict_parameterless_tools": invalid}
                    model_path.write_text(yaml.safe_dump(model))
                    load_benchmark_config(model_path, suite_path, output_directory=root)

    def test_float_proportions_and_vacancy_rounding(self) -> None:
        for field in ("tolerance", "vacancy_fraction"):
            for value in (-0.1, 1.1, float("nan"), float("inf"), "3/4", "0.75", True):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    schelling(**{field: value})
        for fraction in (0.25, 15.5 / 64, 16.5 / 64, 0.24999999999999997):
            with self.subTest(fraction=fraction):
                self.assertEqual(schelling(vacancy_fraction=fraction).vacancy_count, 16)

    def test_openrouter_requires_provider_and_has_no_custom_cache_policy(self) -> None:
        from mam_bench.config import ModelSelection

        base = {"runtime": "openrouter", "model": "vendor/model", "provider": "provider"}
        ModelSelection.model_validate(base)
        for updates in (
            {"provider": ""},
            {"provider": " "},
            {"prompt_cache": True},
            {"prompt_cache": "yes"},
            {"model": "no-vendor"},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                ModelSelection.model_validate({**base, **updates})
        with self.assertRaises(ValidationError):
            ModelSelection.model_validate({"runtime": "openrouter", "model": "vendor/model"})
        selected = ModelSelection.model_validate({**base, "settings": {"temperature": 1.0}})
        self.assertEqual(selected.settings.model_fields_set, {"temperature"})

    def test_agent_budget_and_sampling_settings_validate(self) -> None:
        for settings in (
            {"concurrency": 0},
            {"timeout_seconds": 0},
            {"compaction_tail_tokens": -1},
            {"temperature": 3},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValidationError):
                AgentSettings.model_validate(settings)

    def test_compaction_uses_discovered_context_and_rejects_serving_yaml(self) -> None:
        settings = AgentSettings()
        self.assertEqual(settings.compaction_tail(10_000), 1750)
        self.assertEqual(settings.compaction_tail(260_000), 40_000)
        self.assertNotIn("compaction_tail_tokens", settings.model_fields_set)
        with self.assertRaises(ValueError):
            AgentSettings(compaction_tail_tokens=40_000).compaction_tail(10_000)
        for retired in ("context_window_tokens", "use_sampling_seed"):
            with self.subTest(field=retired), self.assertRaises(ValidationError):
                AgentSettings.model_validate({retired: 10_000})

    def test_vllm_replaces_the_generic_runtime(self) -> None:
        model = selection().model_dump()
        self.assertEqual(model["runtime"], "vllm")
        model["runtime"] = "openai-compatible"
        with self.assertRaises(ValidationError):
            BenchmarkConfig.model_validate(
                {"model": model, "cases": [schelling().model_dump()], "output_directory": "unused"}
            )
