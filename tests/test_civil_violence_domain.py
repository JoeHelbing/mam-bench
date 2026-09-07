"""Civil Violence behavior through public stepping and trajectory interfaces."""

import unittest

from mam_bench.simulations.civil_violence import CivilViolenceSettings, CivilViolenceSim


class CivilViolenceDomainTests(unittest.TestCase):
    def test_ordinary_steps_replay_and_preserve_single_occupancy(self) -> None:
        settings = CivilViolenceSettings(board_size=7, max_transitions=4, seed_id=8)
        stepped = CivilViolenceSim(settings=settings)
        initial = stepped.snapshot()
        for _ in range(4):
            state = stepped.step()
            occupied = [c.location for c in state.citizens if c.location is not None]
            occupied.extend(p.location for p in state.police)
            self.assertEqual(len(occupied), len(set(occupied)))
        result = stepped.run_reference()
        self.assertEqual(result, CivilViolenceSim(settings=settings).run_reference())
        self.assertEqual(result.snapshots[0], initial)
        self.assertEqual(len(result.snapshots), 5)
        self.assertTrue(stepped.finished)
        self.assertEqual(stepped.step(), result.snapshots[-1])

    def test_activation_uses_binary_social_influence_and_strict_lottery(self) -> None:
        from mam_bench.simulations.civil_violence import Citizen

        # One isolated citizen: active=1, inactive=1, opinion=1. Threshold=1
        # makes activation exactly 0.5. Strict > rejects an equal draw.
        settings = CivilViolenceSettings(board_size=3, police_density=0, threshold=1)
        for draw, expected in [(0.49, True), (0.5, False)]:
            sim = CivilViolenceSim(settings, citizens=(Citizen(0, (0, 0), 0),), police=())
            self.assertEqual(sim.step({0: draw}).citizens[0].active, expected)

        # One active and one inactive neighbor: pseudocounts give 2**2 / 2=2.
        # Threshold=1 => sigmoid(1)=0.731058..., so 0.73 activates, 0.74 does not.
        citizens = (Citizen(0, (0, 0), 0), Citizen(1, (0, 1), 0, True), Citizen(2, (1, 0), 0))
        for draw, expected in [(0.73, True), (0.74, False)]:
            sim = CivilViolenceSim(settings, citizens=citizens, police=())
            self.assertEqual(sim.step({0: draw}).citizens[0].active, expected)

    def test_police_arrest_updated_activity_once_and_keep_prisoners_in_denominator(self) -> None:
        from mam_bench.simulations.civil_violence import Citizen, Police

        # Full board freezes movement; all citizen draws of zero activate in
        # the citizen phase. Two police can arrest exactly two distinct citizens.
        citizens = tuple(Citizen(i, divmod(i, 3), -1000) for i in range(7))
        police = (Police(7, (2, 1)), Police(8, (2, 2)))
        settings = CivilViolenceSettings(board_size=3, max_jail_term=0, max_transitions=1)
        sim = CivilViolenceSim(settings, citizens=citizens, police=police)
        state = sim.step({i: 0.0 for i in range(7)})
        self.assertEqual(sum(c.location is None for c in state.citizens), 2)
        self.assertEqual(state.activity(), 5 / 7)
        self.assertEqual(sim.run_reference().mean_activity(), 5 / 7)

    def test_custody_countdown_release_restores_activity_before_next_decision(self) -> None:
        from mam_bench.simulations.civil_violence import Citizen

        settings = CivilViolenceSettings(board_size=3, police_density=0, max_transitions=3)
        jailed = Citizen(0, None, 1000, active=True, jail_remaining=1)
        sim = CivilViolenceSim(settings, citizens=(jailed,), police=())
        first = sim.step().citizens[0]
        self.assertIsNone(first.location)
        self.assertEqual(first.jail_remaining, 0)
        second = sim.step().citizens[0]
        self.assertIsNotNone(second.location)
        self.assertTrue(second.active)
        self.assertFalse(sim.step().citizens[0].active)
        self.assertEqual(sim.run_reference().mean_activity(), 1 / 3)

    def test_releases_and_reserved_moves_cannot_overlap(self) -> None:
        from mam_bench.simulations.civil_violence import Citizen

        citizens = tuple(Citizen(i, divmod(i, 3), 0) for i in range(6)) + (
            Citizen(6, None, 0, True),
            Citizen(7, None, 0, True),
            Citizen(8, None, 0, True),
        )
        sim = CivilViolenceSim(CivilViolenceSettings(board_size=3), citizens=citizens, police=())
        state = sim.step()
        positions = [c.location for c in state.citizens]
        self.assertNotIn(None, positions)
        self.assertEqual(len(set(positions)), 9)

    def test_adjacent_movement_wraps_and_does_not_enter_a_departing_cell(self) -> None:
        from mam_bench.simulations.civil_violence import Citizen

        # All cells occupied except (0,0); (2,2) sees it diagonally across the torus.
        citizens = (Citizen(0, (2, 2), 0),) + tuple(
            Citizen(i, position, 0)
            for i, position in enumerate(
                [(r, c) for r in range(3) for c in range(3) if (r, c) not in {(0, 0), (2, 2)}],
                start=1,
            )
        )
        sim = CivilViolenceSim(CivilViolenceSettings(board_size=3), citizens=citizens, police=())
        before = sim.snapshot()
        after = sim.step()
        self.assertEqual(after.citizens[0].location, (0, 0))
        self.assertEqual(
            after.citizens[1:],
            tuple(
                type(c)(
                    c.agent_id,
                    c.location,
                    c.private_preference,
                    after.citizens[i].active,
                    c.jail_remaining,
                )
                for i, c in enumerate(before.citizens[1:], start=1)
            ),
        )

    def test_invalid_initial_occupancy_and_empty_population_are_rejected(self) -> None:
        from mam_bench.simulations.civil_violence import Citizen

        with self.assertRaises(ValueError):
            CivilViolenceSim(citizens=(), police=())
        with self.assertRaises(ValueError):
            CivilViolenceSim(citizens=(Citizen(0, (0, 0), 0), Citizen(1, (0, 0), 0)), police=())
        with self.assertRaises(ValueError):
            CivilViolenceSettings(citizen_density=0.9, police_density=0.2)

    def test_arrest_risk_uses_unrounded_ratio_and_no_epsilon_factor(self) -> None:
        from mam_bench.simulations.civil_violence import Citizen, Police

        # Isolated citizen and one visible police: sigmoid saturates at one;
        # remaining activation probability is exp(-2.3)=0.1002588437...
        settings = CivilViolenceSettings(board_size=5, citizen_vision=2)
        for draw, expected in [(0.1, True), (0.101, False)]:
            sim = CivilViolenceSim(
                settings, citizens=(Citizen(0, (0, 0), -1000),), police=(Police(1, (2, 2)),)
            )
            self.assertEqual(sim.step({0: draw}).citizens[0].active, expected)

    def test_arrests_are_adjacent_even_when_police_vision_is_large(self) -> None:
        from mam_bench.simulations.civil_violence import Citizen, Police

        # With a full board no occupant moves. Police vision sees all citizens,
        # but the only active citizen is two cells away and cannot be arrested.
        citizens = tuple(Citizen(i, divmod(i, 5), -1000 if i == 12 else 1000) for i in range(24))
        sim = CivilViolenceSim(
            CivilViolenceSettings(board_size=5, police_vision=9),
            citizens=citizens,
            police=(Police(24, (4, 4)),),
        )
        state = sim.step({i: 0 for i in range(24)})
        self.assertTrue(state.citizens[12].active)
        self.assertTrue(all(c.location is not None for c in state.citizens))

    def test_jail_terms_include_both_endpoints_and_zero_stays_offgrid_until_next_cycle(
        self,
    ) -> None:
        from mam_bench.simulations.civil_violence import Citizen, Police

        seen: set[int] = set()
        for seed in range(10):
            citizens = tuple(Citizen(i, divmod(i, 3), -1000) for i in range(8))
            sim = CivilViolenceSim(
                CivilViolenceSettings(board_size=3, max_jail_term=1, seed_id=seed),
                citizens=citizens,
                police=(Police(8, (2, 2)),),
            )
            state = sim.step({i: 0 for i in range(8)})
            jailed = [c for c in state.citizens if c.location is None]
            self.assertEqual(len(jailed), 1)
            seen.add(jailed[0].jail_remaining)
        self.assertEqual(seen, {0, 1})

    def test_npz_retains_score_replay_inputs_without_pickle(self) -> None:
        import tempfile
        from pathlib import Path

        import numpy as np

        settings = CivilViolenceSettings(board_size=5, max_transitions=4)
        result = CivilViolenceSim(settings).run_reference()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ordinary.npz"
            result.write(path)
            with np.load(path, allow_pickle=False) as saved:
                scored = saved["roles"] == 0
                active = saved["active"][1:, scored] & ~saved["jailed"][1:, scored]
                self.assertAlmostEqual(float(active.mean()), result.mean_activity())
                self.assertEqual(saved["agent_locations"].shape[0], 5)
                self.assertEqual(str(saved["settings_json"]), settings.model_dump_json())

    def test_corresponding_randomness_is_independent_of_population_iteration(self) -> None:
        from mam_bench.simulations.civil_violence import Citizen

        # A full board prevents moves. Reordering policy execution cannot shift
        # another identity's activation draw or expose a partially updated state.
        citizens = tuple(Citizen(i, divmod(i, 3), 0) for i in range(9))
        settings = CivilViolenceSettings(board_size=3, threshold=0, max_transitions=1)
        forward = CivilViolenceSim(settings, citizens=citizens, police=()).step()
        backward = CivilViolenceSim(settings, citizens=tuple(reversed(citizens)), police=()).step()
        self.assertEqual(
            {c.agent_id: c.active for c in forward.citizens},
            {c.agent_id: c.active for c in backward.citizens},
        )

    def test_ordinary_zero_police_is_valid_but_controlled_role_must_exist(self) -> None:
        settings = CivilViolenceSettings(
            board_size=3, citizen_density=0.5, police_density=0, controlled_agent_count=4
        )
        CivilViolenceSim(settings).run_reference()
        with self.assertRaises(ValueError):
            settings.validate_controlled_role("police")
        with self.assertRaises(ValueError):
            settings.validate_controlled_role("citizen")


class PopulationRoundingTests(unittest.TestCase):
    def test_rounded_populations_cannot_exceed_capacity(self) -> None:
        with self.assertRaises(ValueError):
            CivilViolenceSettings(board_size=5, citizen_density=0.3, police_density=0.7)
