"""Complete upfront validation and custom-suite replacement."""

import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import ValidationError

from mam_bench.config import AgentSettings, BenchmarkConfig, load_benchmark_config
from support import civil, schelling, selection


class ConfigurationTests(unittest.TestCase):
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
            "vision_radius": 1,
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

    def test_openrouter_prompt_cache_and_required_provider(self) -> None:
        from mam_bench.config import OpenRouterModel

        base = {"runtime": "openrouter", "model": "vendor/model", "provider": "provider"}
        self.assertEqual(OpenRouterModel.model_validate(base).prompt_cache, "auto")
        for value in ("auto", "required", "off"):
            self.assertEqual(
                OpenRouterModel.model_validate({**base, "prompt_cache": value}).prompt_cache, value
            )
        for updates in (
            {"provider": ""},
            {"provider": " "},
            {"prompt_cache": True},
            {"prompt_cache": "yes"},
            {"model": "no-vendor"},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                OpenRouterModel.model_validate({**base, **updates})
        with self.assertRaises(ValidationError):
            OpenRouterModel.model_validate({"runtime": "openrouter", "model": "vendor/model"})
        selected = OpenRouterModel.model_validate({**base, "settings": {"temperature": 1.0}})
        self.assertEqual(selected.settings.model_fields_set, {"temperature"})

    def test_agent_budget_and_sampling_settings_validate(self) -> None:
        for settings in (
            {"concurrency": 0},
            {"timeout_seconds": 0},
            {"compaction_tail_tokens": 200000},
            {"temperature": 3},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValidationError):
                AgentSettings.model_validate(settings)
