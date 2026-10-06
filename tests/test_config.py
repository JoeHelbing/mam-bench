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
            {"max_steps": 60},
            {"private_preference_std": -1},
            {"vision_radius": 0},
            {"vision_radius": True},
            {"citizen_vision": 1, "police_vision": 1},
            {"threshold": float("nan")},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValidationError):
                civil(**updates)
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
            cases = [civil().model_dump(), schelling(seed=19).model_dump(), civil().model_dump()]
            suite.write_text(yaml.safe_dump({"cases": cases}))
            config = load_benchmark_config(model, suite, output_directory=Path("results/custom"))
            self.assertEqual([c.model_dump() for c in config.cases], cases)
            self.assertEqual(config.output_directory, Path("results/custom").resolve())
            with self.assertRaises(ValidationError):
                BenchmarkConfig.model_validate({"cases": cases, "output_directory": "unused"})
            # The official default intentionally rejects execution until its cases are selected.
            with self.assertRaises(ValidationError):
                load_benchmark_config(model, output_directory=root)

    def test_float_proportions_and_vacancy_rounding(self) -> None:
        for field in ("tolerance", "vacancy_fraction"):
            for value in (-0.1, 1.1, float("nan"), float("inf"), "3/4", "0.75", True):
                with self.subTest(field=field, value=value), self.assertRaises(ValidationError):
                    schelling(**{field: value})
        for fraction in (0.25, 15.5 / 64, 16.5 / 64, 0.24999999999999997):
            with self.subTest(fraction=fraction):
                self.assertEqual(schelling(vacancy_fraction=fraction).vacancy_count, 16)

    def test_agent_budget_and_sampling_settings_validate(self) -> None:
        for settings in (
            {"concurrency": 0},
            {"timeout_seconds": 0},
            {"compaction_tail_tokens": 200000},
            {"temperature": 3},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValidationError):
                AgentSettings.model_validate(settings)
