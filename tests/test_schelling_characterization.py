"""Golden trajectories for strict radius-3 local improvement."""

import hashlib
import json
import unittest
from pathlib import Path

from mam_bench.simulations.schelling.profile import LandscapeCell, Rational
from mam_bench.simulations.schelling.reference import reference_rng
from mam_bench.simulations.schelling.simulation import SchellingSim


class CharacterizationTests(unittest.TestCase):
    def test_strict_vision_improvement_trajectories(self) -> None:
        cases = json.loads(
            Path(__file__).with_name("schelling_reference_characterization.json").read_text()
        )
        for case in cases:
            size, tn, td, vn, vd, seed, horizon = case["case"]
            with self.subTest(case=case["case"]):
                result = SchellingSim(
                    LandscapeCell(size, Rational(tn, td), Rational(vn, vd)),
                    seed,
                    max_transitions=horizon,
                ).run_reference()
                self.assertEqual(int(result.terminal_status), case["status"])
                self.assertEqual(result.rounds_completed, case["rounds"])
                for name, expected in case["arrays"].items():
                    self.assertEqual(
                        hashlib.sha256(getattr(result, name).tobytes()).hexdigest(), expected
                    )

    def test_exact_semantic_rng_draws(self) -> None:
        expected = [
            [
                2287940939,
                407155870,
                2369239404,
                3053754524,
                3045898625,
                3612000708,
                2086138269,
                2450546852,
            ],
            [
                1317164991,
                1093167028,
                1604527909,
                2494791736,
                1777035393,
                589281905,
                399839979,
                491357411,
            ],
            [
                595753062,
                1023275346,
                1518841924,
                2641202235,
                1466930484,
                4256282081,
                4294717265,
                1703006708,
            ],
        ]
        cell = LandscapeCell(20, Rational(3, 4), Rational(1, 4))
        for stream, draws in enumerate(expected):
            self.assertEqual(
                reference_rng(cell, 50, stream).integers(2**32, size=8).tolist(), draws
            )
