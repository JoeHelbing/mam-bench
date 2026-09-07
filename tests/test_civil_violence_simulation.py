import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo

from mam_bench.config import load_benchmark_config
from mam_bench.runner import run_benchmark
from schelling_support import runtime_for


async def participate(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    return ModelResponse(parts=[ToolCallPart("participate", {"active": True})])


class CivilViolenceEvaluationTests(unittest.IsolatedAsyncioTestCase):
    async def test_citizen_yaml_runs_paired_worlds_and_scores_saved_activity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "benchmark.yaml"
            path.write_text(
                "simulations: [civil-violence-citizens-v1]\n"
                "civil_violence_citizens:\n"
                "  board_size: 5\n"
                "  citizen_density: 0.48\n"
                "  police_density: 0.12\n"
                "  citizen_vision: 1\n"
                "  police_vision: 1\n"
                "  seed_id: 7\n"
                "  max_transitions: 3\n"
                "  controlled_agent_count: 2\n"
                "models:\n"
                "  - id: offline\n"
                "    runtime: openrouter\n"
                "    model: offline\n"
                "    provider: offline\n"
                f"output_directory: {root / 'results'}\n"
            )
            config = load_benchmark_config(path)
            topline = await run_benchmark(
                config, runtime_builder=lambda _: runtime_for(participate)
            )
            summary_path = next((root / "results").rglob("result.json"))
            summary = json.loads(summary_path.read_text())
            self.assertEqual(summary["config"]["seed_id"], 7)
            self.assertEqual(summary["config"]["max_transitions"], 3)
            self.assertEqual(summary["role"], "citizen")
            self.assertEqual(len(summary["controlled_agent_ids"]), 2)
            self.assertEqual(len(summary["scored_agent_ids"]), 10)
            means: list[float] = []
            initial: list[NDArray[np.int64]] = []
            for name in ("ordinary", "model-controlled"):
                with np.load(summary_path.parent / f"{name}.npz") as trajectory:
                    self.assertEqual(len(trajectory["agent_locations"]), 4)
                    initial.append(trajectory["agent_locations"][0])
                    scored = summary["scored_agent_ids"]
                    activity = trajectory["active"][1:, scored]
                    jailed = trajectory["jailed"][1:, scored]
                    means.append(float(np.count_nonzero(activity & ~jailed)) / 30)
            np.testing.assert_array_equal(initial[0], initial[1])
            self.assertAlmostEqual(topline.entries[0].primary_score.value, means[1] - means[0])
            self.assertAlmostEqual(summary["activity_lift"], means[1] - means[0])
            self.assertTrue((summary_path.parent / "agent-messages").is_dir())

    async def test_citizen_collision_retry_preserves_frozen_state_and_local_privacy(self) -> None:
        from pydantic_ai.messages import ModelRequest, RetryPromptPart, UserPromptPart

        from mam_bench.simulations.civil_violence.models import Citizen
        from mam_bench.simulations.civil_violence.settings import CivilViolenceSettings
        from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim

        seen: dict[int, int] = {}

        async def collide(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            prompts = [
                p.content
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
                if isinstance(p, UserPromptPart)
            ]
            observation = json.loads(str(prompts[-1]))
            agent_id = observation["agent_id"]
            seen[agent_id] = seen.get(agent_id, 0) + 1
            self.assertNotIn("private_preference", str(observation))
            self.assertNotIn("ordinary_trajectory", observation)
            self.assertEqual(len(observation["neighborhood"]), 8)
            self.assertTrue(all(not c.active for c in simulation.snapshot().citizens))
            retry = any(
                isinstance(p, RetryPromptPart)
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        "participate",
                        {"active": True} if retry else {"active": True, "row": 1, "column": 1},
                    )
                ]
            )

        settings = CivilViolenceSettings(
            board_size=5,
            citizen_vision=1,
            police_density=0,
            controlled_agent_count=2,
            max_transitions=1,
        )
        simulation = CivilViolenceSim(
            settings=settings,
            citizens=(Citizen(0, (0, 0), 0), Citizen(1, (0, 2), 0), Citizen(2, (2, 0), 0)),
            police=(),
        )
        initial = simulation.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            await simulation.run(runtime_for(collide, concurrency=1), Path(directory) / "trial")
        state = simulation.snapshot()
        controlled = [c for c in state.citizens if c.agent_id in simulation.controlled_agent_ids]
        self.assertEqual(sorted(seen.values()), [1, 2])
        self.assertTrue(all(c.active for c in controlled))
        self.assertEqual(sum(c.location == (1, 1) for c in controlled), 1)
        self.assertTrue(all(not c.active for c in initial.citizens))

    async def test_jailed_session_keeps_history_notebook_and_skips_custody_turns(self) -> None:
        from pydantic_ai.messages import ModelRequest, ToolReturnPart, UserPromptPart

        from mam_bench.simulations.civil_violence.models import Citizen, Police
        from mam_bench.simulations.civil_violence.settings import CivilViolenceSettings
        from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim

        observed: list[tuple[int, int]] = []

        async def remember(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            prompts = [
                p.content
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
                if isinstance(p, UserPromptPart)
            ]
            from pydantic_ai.messages import TextContent

            texts = [
                text
                for content in prompts
                for text in (
                    [content]
                    if isinstance(content, str)
                    else [
                        item if isinstance(item, str) else item.content
                        for item in content
                        if isinstance(item, (str, TextContent))
                    ]
                )
            ]
            obs = json.loads(next(text for text in reversed(texts) if text.startswith("{")))
            if obs["cycle"] > 1:
                self.assertTrue(any("durable fact" in text for text in texts))
            observed.append((obs["agent_id"], obs["cycle"]))
            has_written = any(
                isinstance(p, ToolReturnPart) and p.tool_name == "write_memory"
                for m in messages
                if isinstance(m, ModelRequest)
                for p in m.parts
            )
            if not has_written:
                return ModelResponse(
                    parts=[
                        ToolCallPart("write_memory", {"content": "durable fact"}),
                        ToolCallPart("post_message", {"text": "All citizens are now active"}),
                    ]
                )
            return ModelResponse(parts=[ToolCallPart("participate", {"active": obs["cycle"] == 1})])

        settings = CivilViolenceSettings(
            board_size=5, controlled_agent_count=1, max_jail_term=1, max_transitions=8
        )
        sim = CivilViolenceSim(
            settings,
            citizens=(Citizen(0, (0, 0), 100), Citizen(1, (1, 0), 100)),
            police=(Police(2, (0, 1)),),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "trial"
            await sim.run(runtime_for(remember), output)
            agent_id = sim.controlled_agent_ids[0]
            with np.load(output / "model-controlled.npz") as saved:
                ids = saved["agent_ids"].tolist()
                column = ids.index(agent_id)
                jailed = saved["jailed"][:, column]
                self.assertTrue(jailed[1])  # ordinary police arrest the active model
                self.assertFalse(jailed[-1])
                expected = {cycle + 1 for cycle, in_jail in enumerate(jailed[:-1]) if not in_jail}
            self.assertEqual({cycle for _, cycle in observed}, expected)
            self.assertGreater(max(expected), 1)  # same persistent identity resumes
            self.assertLess(len(expected), settings.max_transitions)
            self.assertEqual({identity for identity, _ in observed}, {agent_id})
            archive = output / "agent-messages"
            self.assertTrue(archive.is_dir())
            assert sim.sessions is not None
            messages = await sim.sessions.board.read("observer")
            self.assertIn("message by", messages.content)
            self.assertIn("All citizens are now active", messages.content)
            self.assertIn("announcement by", messages.content)
            self.assertEqual((await sim.sessions.board.read("observer")).content, "")
            self.assertTrue((await sim.sessions.board.read("another-observer")).content)
            self.assertTrue(
                all(
                    not citizen.active
                    for citizen in sim.snapshot().citizens
                    if citizen.agent_id != agent_id
                )
            )

    async def test_invalid_turn_exhaustion_stays_and_terminal_skips_extra_posts(self) -> None:
        from mam_bench.simulations.civil_violence.models import Citizen
        from mam_bench.simulations.civil_violence.settings import CivilViolenceSettings
        from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim

        calls = 0

        async def invalid(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal calls
            calls += 1
            return ModelResponse(
                parts=[ToolCallPart("participate", {"active": False, "row": 999, "column": 999})]
            )

        settings = CivilViolenceSettings(
            board_size=5, controlled_agent_count=1, police_density=0, max_transitions=1
        )
        citizens = (Citizen(0, (0, 0), 0, True), Citizen(1, (3, 3), 0, True))
        sim = CivilViolenceSim(settings, citizens=citizens, police=())
        initial = sim.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            await sim.run(runtime_for(invalid), Path(directory) / "trial")
        agent_id = sim.controlled_agent_ids[0]
        start = next(c.location for c in initial.citizens if c.agent_id == agent_id)
        self.assertEqual(calls, 3)
        self.assertEqual(sim.citizens[agent_id].location, start)
        self.assertTrue(sim.citizens[agent_id].active)

        async def terminal(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            return ModelResponse(
                parts=[
                    ToolCallPart("post_message", {"text": "SHOULD SKIP"}),
                    ToolCallPart("participate", {"active": True}),
                ]
            )

        sim = CivilViolenceSim(settings, citizens=citizens, police=())
        with tempfile.TemporaryDirectory() as directory:
            await sim.run(runtime_for(terminal), Path(directory) / "trial")
        assert sim.sessions is not None
        board = await sim.sessions.board.read("observer")
        self.assertNotIn("SHOULD SKIP", board.content)
        self.assertIn("announcement", board.content)

    def test_selected_counts_reject_impossible_and_empty_scored_populations(self) -> None:
        from pydantic import ValidationError

        from mam_bench.config import BenchmarkConfig

        common = {
            "models": [
                {
                    "id": "offline",
                    "runtime": "openrouter",
                    "model": "offline",
                    "provider": "offline",
                }
            ],
            "output_directory": "results",
        }
        for role, section, density, count in (
            ("citizens", "civil_violence_citizens", 0.2, 5),
            ("police", "civil_violence_police", 0.0, 1),
        ):
            settings: dict[str, object] = {
                "board_size": 5,
                "controlled_agent_count": count,
                "citizen_density": density if role == "citizens" else 0.5,
                "police_density": 0.0,
            }
            with self.assertRaises(ValidationError):
                BenchmarkConfig.model_validate(
                    {**common, "simulations": [f"civil-violence-{role}-v1"], section: settings}
                )

    async def test_artifact_failure_is_unscored_and_requires_a_fresh_run(self) -> None:
        from mam_bench.benchmark import PairInfrastructureFailure
        from mam_bench.simulations.civil_violence.settings import CivilViolenceSettings
        from mam_bench.simulations.civil_violence.simulation import CivilViolenceSim

        sim = CivilViolenceSim(
            CivilViolenceSettings(
                board_size=5,
                citizen_density=0.48,
                police_density=0.12,
                controlled_agent_count=2,
                max_transitions=1,
            )
        )
        initial = sim.snapshot()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "failed"

            async def block_artifact(
                messages: list[ModelMessage], info: AgentInfo
            ) -> ModelResponse:
                (output / "ordinary.npz").mkdir(exist_ok=True)
                return await participate(messages, info)

            with self.assertRaises(PairInfrastructureFailure) as failure:
                await sim.run(runtime_for(block_artifact), output)
            self.assertEqual(failure.exception.failure.kind, "artifact_write")
            self.assertFalse((output / "result.json").exists())
            with self.assertRaisesRegex(RuntimeError, "failed trials"):
                await sim.astep()
            fresh = Path(directory) / "fresh"
            score = await sim.run(runtime_for(participate), fresh)
            self.assertTrue(np.isfinite(score.value))
            self.assertEqual(sim.snapshots[0], initial)
            self.assertTrue((fresh / "result.json").exists())
