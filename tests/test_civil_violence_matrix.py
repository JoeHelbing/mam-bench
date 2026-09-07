"""Ship calibrated settings through the real mixed benchmark, with offline responses."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pydantic_ai.exceptions import ModelAPIError
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo

from mam_bench.benchmark import BenchmarkRunFailure
from mam_bench.config import load_benchmark_config
from mam_bench.runner import run_benchmark
from mam_bench.simulations.civil_violence import CivilViolenceSim
from mam_bench.simulations.schelling.settings import SchellingSettings
from schelling_support import runtime_for

ROOT = Path(__file__).resolve().parents[1]


async def scripted(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    names = {tool.name for tool in info.output_tools}
    if "participate" in names:
        return ModelResponse(parts=[ToolCallPart("participate", {"active": True})])
    if "intervene" in names:
        return ModelResponse(parts=[ToolCallPart("intervene", {})])
    return ModelResponse(parts=[ToolCallPart("stay", {})])


def mean_from_file(path: Path, scored_ids: list[int]) -> float:
    with np.load(path, allow_pickle=False) as data:
        columns = {int(identity): column for column, identity in enumerate(data["agent_ids"])}
        selected = [columns[identity] for identity in scored_ids]
        activity = data["active"][1:, selected]
        jailed = data["jailed"][1:, selected]
        return float((activity & ~jailed).mean())


class CivilViolenceMatrixTests(unittest.IsolatedAsyncioTestCase):
    async def test_shipped_conditions_complete_mixed_matrix_and_fresh_attempt(self) -> None:
        citizens = load_benchmark_config(ROOT / "examples/civil-violence-citizens.yaml")
        police = load_benchmark_config(ROOT / "examples/civil-violence-police.yaml")
        self.assertEqual(citizens.civil_violence_citizens, police.civil_violence_police)
        self.assertEqual(citizens.civil_violence_citizens.max_transitions, 60)
        self.assertEqual(citizens.civil_violence_citizens.citizen_count, 101)
        self.assertEqual(citizens.civil_violence_citizens.police_count, 4)
        simulations = (*citizens.simulations, *police.simulations, "schelling-influence-pilot-v1")
        with tempfile.TemporaryDirectory() as directory:
            config = citizens.model_copy(
                update={
                    "simulations": simulations,
                    "civil_violence_police": police.civil_violence_police,
                    "schelling": SchellingSettings(
                        board_size=8,
                        vision_radius=1,
                        controlled_agent_count=2,
                        max_transitions=2,
                    ),
                    "output_directory": Path(directory),
                }
            )
            attempts: list[Path] = []
            first = await run_benchmark(
                config,
                runtime_builder=lambda _: runtime_for(scripted),
                on_output_directory=attempts.append,
            )
            self.assertEqual(tuple(entry.simulation_id for entry in first.entries), simulations)
            for entry in first.entries[:2]:
                output = attempts[0] / "runs" / entry.simulation_id / entry.model_id
                summary = json.loads((output / "result.json").read_text())
                expected_settings = (
                    config.civil_violence_citizens
                    if summary["role"] == "citizen"
                    else config.civil_violence_police
                )
                self.assertEqual(summary["config"], expected_settings.model_dump(mode="json"))
                self.assertEqual(summary["cycles_completed"], 60)
                scored_ids = summary["scored_agent_ids"]
                reference = mean_from_file(output / "ordinary.npz", scored_ids)
                controlled = mean_from_file(output / "model-controlled.npz", scored_ids)
                expected = controlled - reference
                if summary["role"] == "police":
                    expected = -expected
                self.assertAlmostEqual(entry.primary_score.value, expected)
                self.assertAlmostEqual(summary["ordinary"]["mean_activity"], reference)
                self.assertAlmostEqual(summary["model_controlled"]["mean_activity"], controlled)
                self.assertEqual(len(scored_ids), 99 if summary["role"] == "citizen" else 101)
                with np.load(output / "ordinary.npz") as ordinary:
                    self.assertEqual(len(ordinary["cycles"]), 61)
                with self.assertRaises(FileExistsError):
                    await CivilViolenceSim(expected_settings, summary["role"]).run(
                        runtime_for(scripted),
                        output,
                    )
            protected = {
                path: path.read_bytes() for path in attempts[0].rglob("*") if path.is_file()
            }
            second = await run_benchmark(
                config,
                runtime_builder=lambda _: runtime_for(scripted),
                on_output_directory=attempts.append,
            )
            self.assertNotEqual(attempts[0], attempts[1])
            self.assertEqual(len(second.entries), 3)
            for path, content in protected.items():
                self.assertEqual(path.read_bytes(), content)

    async def test_failed_citizen_pair_does_not_discard_police_or_schelling(self) -> None:
        async def fail_citizens(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            if any(tool.name == "participate" for tool in info.output_tools):
                raise ModelAPIError("offline", "secret=must-not-appear")
            return await scripted(messages, info)

        citizens = load_benchmark_config(ROOT / "examples/civil-violence-citizens.yaml")
        police = load_benchmark_config(ROOT / "examples/civil-violence-police.yaml")
        with tempfile.TemporaryDirectory() as directory:
            config = citizens.model_copy(
                update={
                    "simulations": (
                        *citizens.simulations,
                        *police.simulations,
                        "schelling-influence-pilot-v1",
                    ),
                    "civil_violence_citizens": citizens.civil_violence_citizens.model_copy(
                        update={"max_transitions": 2},
                    ),
                    "civil_violence_police": police.civil_violence_police.model_copy(
                        update={"max_transitions": 2},
                    ),
                    "schelling": SchellingSettings(
                        board_size=8,
                        vision_radius=1,
                        controlled_agent_count=2,
                        max_transitions=2,
                    ),
                    "output_directory": Path(directory),
                }
            )
            attempts: list[Path] = []
            with self.assertRaises(BenchmarkRunFailure) as raised:
                await run_benchmark(
                    config,
                    runtime_builder=lambda _: runtime_for(fail_citizens),
                    on_output_directory=attempts.append,
                )
            self.assertNotIn("must-not-appear", str(raised.exception))
            topline = json.loads((attempts[0] / "topline.json").read_text())
            self.assertEqual(
                [entry["simulation_id"] for entry in topline["entries"]],
                ["civil-violence-police-v1", "schelling-influence-pilot-v1"],
            )
            failed = attempts[0] / "runs" / "civil-violence-citizens-v1"
            self.assertEqual(list(failed.rglob("result.json")), [])
