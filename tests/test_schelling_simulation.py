import shutil
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path

from pydantic import JsonValue

from mam_bench.runner import BenchmarkSelection, preflight_benchmark, run_benchmark
from mam_bench.runtime import (
    ModelRequest,
    ModelResponse,
    ModelRuntime,
    ModelUsage,
    RuntimeCapability,
    RuntimeDescriptor,
    RuntimeMessage,
    SamplingControl,
    TextOutput,
    TextPart,
    ToolCallPart,
)
from mam_bench.simulations.schelling import (
    SchellingBenchmarkSimulation,
    SchellingEvidenceManifest,
    default_schelling_dataset_path,
)


class _AlwaysStayRuntime:
    descriptor = RuntimeDescriptor(
        model_id="always-stay",
        runtime="scripted",
        provider="test",
        model="test/always-stay",
        adapter_version="scripted-v1",
        capabilities=frozenset(RuntimeCapability),
        sampling_controls=frozenset(SamplingControl),
        max_concurrent_requests=16,
    )

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    async def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        output: JsonValue
        if request.request_id.endswith(":state"):
            arguments: dict[str, JsonValue] = {
                "board_round_offset": 0,
                "coordination_round_offset": 0,
                "include_reference": False,
            }
            output = arguments
            part = ToolCallPart(
                call_id=f"{request.request_id}:call",
                name="inspect_state",
                arguments=arguments,
            )
        elif isinstance(request.output, TextOutput):
            output = "stay coordinated"
            part = TextPart(text=output)
        else:
            arguments = {"stay": True, "row": None, "column": None}
            output = arguments
            part = ToolCallPart(
                call_id=f"{request.request_id}:call",
                name="submit_move",
                arguments=arguments,
            )
        return ModelResponse(
            request_id=request.request_id,
            messages=(RuntimeMessage(role="assistant", parts=(part,)),),
            output=output,
            usage=ModelUsage(requests=1, input_tokens=10, output_tokens=2),
            latency_seconds=0.0,
            provider_response_id=None,
        )


class _RuntimeFactory:
    descriptor = _AlwaysStayRuntime.descriptor

    def __init__(self) -> None:
        self.instances: list[_AlwaysStayRuntime] = []

    def create(self) -> ModelRuntime:
        runtime = _AlwaysStayRuntime()
        self.instances.append(runtime)
        return runtime


class SchellingBenchmarkSimulationTests(unittest.IsolatedAsyncioTestCase):
    async def test_fixed_pilot_runs_through_simulation_seam(self) -> None:
        simulation = SchellingBenchmarkSimulation()
        factory = _RuntimeFactory()
        with tempfile.TemporaryDirectory() as directory:
            output_directory = Path(directory) / "benchmark"

            prepared_simulation = simulation.prepare()

            self.assertFalse(output_directory.exists())
            self.assertEqual(factory.instances, [])
            self.assertEqual(prepared_simulation.config.cell.tolerance_index, 17)
            self.assertEqual(prepared_simulation.config.cell.vacancy_index, 3)
            self.assertEqual(prepared_simulation.config.seed_id, 22)
            self.assertEqual(prepared_simulation.config.objective, "integration")
            prepared = preflight_benchmark(
                BenchmarkSelection(
                    simulations=("schelling-influence-pilot-v1",),
                    runtimes=("always-stay",),
                    output_directory=output_directory,
                ),
                simulations={"schelling-influence-pilot-v1": simulation},
                runtimes={"always-stay": factory},
            )

            topline = await run_benchmark(prepared)

            self.assertEqual(len(factory.instances), 1)
            self.assertEqual(len(factory.instances[0].requests), 20 * 2 * 16 * 2)
            self.assertEqual(len(topline.entries), 1)
            entry = topline.entries[0]
            self.assertEqual(entry.simulation_id, "schelling-influence-pilot-v1")
            self.assertEqual(entry.model_id, "always-stay")
            self.assertEqual(entry.primary_score.name, "directional_lift")
            self.assertTrue(entry.primary_score.higher_is_better)
            run_directory = (
                output_directory / "runs" / "schelling-influence-pilot-v1" / "always-stay"
            )
            manifest_path = run_directory / "evidence-manifest.json"
            manifest = SchellingEvidenceManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
            self.assertEqual(manifest.schema_version, "mam-bench.evidence-manifest.v1")
            self.assertEqual(manifest.status, "complete")
            self.assertFalse(manifest.diagnostic_artifacts_retained)
            self.assertEqual(len(manifest.reference_dataset_manifest_sha256), 64)
            paths = {record.path for record in manifest.files}
            self.assertIn("run.json", paths)
            self.assertIn("trajectory.npz", paths)
            self.assertIn("events.jsonl", paths)
            self.assertIn("coordination-board.jsonl", paths)
            self.assertEqual(
                len([path for path in paths if path.startswith("actor-histories/")]),
                16,
            )
            self.assertEqual(
                len(
                    [
                        path
                        for path in paths
                        if path.startswith("checkpoints/") and path.endswith(".json")
                    ]
                ),
                42,
            )


class SchellingDatasetPreflightTests(unittest.TestCase):
    def test_preparation_validates_dataset_without_constructing_it(self) -> None:
        dataset_root = default_schelling_dataset_path()
        before = {
            path.relative_to(dataset_root): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in dataset_root.rglob("*")
            if path.is_file()
        }

        prepared = SchellingBenchmarkSimulation().prepare()

        after = {
            path.relative_to(dataset_root): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in dataset_root.rglob("*")
            if path.is_file()
        }
        self.assertEqual(before, after)
        self.assertEqual(len(prepared.reference_dataset_manifest_sha256), 64)

    def test_missing_changed_or_extra_cells_fail_before_runtime_or_output(self) -> None:
        mutations: dict[str, Callable[[Path], None]] = {
            "missing": _remove_cell,
            "changed": _change_cell,
            "extra": _add_cell,
        }
        for name, mutate in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                dataset_root = root / "dataset"
                shutil.copytree(default_schelling_dataset_path(), dataset_root)
                mutate(dataset_root)
                factory = _RuntimeFactory()
                output_directory = root / "output"

                with self.assertRaises((FileNotFoundError, ValueError)):
                    preflight_benchmark(
                        BenchmarkSelection(
                            simulations=("schelling-influence-pilot-v1",),
                            runtimes=("always-stay",),
                            output_directory=output_directory,
                        ),
                        simulations={
                            "schelling-influence-pilot-v1": SchellingBenchmarkSimulation(
                                dataset_root=dataset_root
                            )
                        },
                        runtimes={"always-stay": factory},
                    )

                self.assertEqual(factory.instances, [])
                self.assertFalse(output_directory.exists())


def _remove_cell(root: Path) -> None:
    (root / "cells" / "t00-v00.npz").unlink()


def _change_cell(root: Path) -> None:
    path = root / "cells" / "t00-v00.npz"
    path.write_bytes(path.read_bytes() + b"changed")


def _add_cell(root: Path) -> None:
    (root / "cells" / "unexpected.npz").write_bytes(b"unexpected")


if __name__ == "__main__":
    unittest.main()
