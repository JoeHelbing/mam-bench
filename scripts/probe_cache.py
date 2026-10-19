"""Measure provider-reported cache reuse through MAM-Bench's streamed sessions."""

import argparse
import asyncio
import json
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel
from pydantic_ai import Agent, ToolOutput, UsageLimits

from mam_bench.config import load_benchmark_config
from mam_bench.model_setup import resolve_model_setup
from mam_bench.runtime import create_model
from mam_bench.sessions import AgentSessions, Completed


class ProbeAnswer(BaseModel):
    value: int


async def probe(model_path: Path, output: Path) -> bool:
    config = load_benchmark_config(model_path, output_directory=output)
    setup = await resolve_model_setup(config.model)
    model = create_model(setup.selection)
    # A nonce makes the first long prefix cold even when the probe is rerun.
    prefix = f"Cache probe {uuid4().hex}. Return the requested value using record_probe.\n"
    reference = "\n".join(
        f"Reference row {i}: stable benchmark instructions preserve independent agent histories."
        for i in range(600)
    )
    rows: list[dict[str, object]] = []
    passed = True
    async with model:
        for group, instructions, turns in (
            ("short", prefix, (("short", 11),)),
            ("long", prefix + reference, (("cold", 21), ("warm", 22), ("growing", 23))),
        ):
            sessions = AgentSessions(
                Agent(
                    model,
                    instructions=instructions,
                    output_type=ToolOutput(ProbeAnswer, name="record_probe"),
                ),
                context_window_tokens=setup.context_window_tokens,
                settings=setup.selection.settings,
                artifact_directory=output / group,
            )
            for phase, value in turns:
                # Cold and warm are distinct agents; growing continues warm's history.
                identity = "warm" if phase == "growing" else phase
                before = sessions.usage
                result = await sessions.run(
                    identity,
                    f"Return value {value} using record_probe.",
                    deps=None,
                    usage_limits=UsageLimits(request_limit=3),
                )
                if not isinstance(result, Completed) or result.value.value != value:
                    raise RuntimeError(f"{phase}: model did not return the requested tool output")
                after = sessions.usage
                if phase in ("warm", "growing"):
                    passed = passed and after.cache_read_tokens > before.cache_read_tokens
                row: dict[str, object] = {
                    "phase": phase,
                    **{
                        field: getattr(after, field) - getattr(before, field)
                        for field in (
                            "requests",
                            "input_tokens",
                            "output_tokens",
                            "cache_read_tokens",
                            "cache_write_tokens",
                        )
                    },
                }
                rows.append(row)
                print(json.dumps(row), flush=True)
    report = {
        "model": setup.selection.model_dump(mode="json"),
        "setup": setup.metadata,
        "observations": rows,
        "warm_and_growing_cache_hits": passed,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "cache-report.json").write_text(json.dumps(report, indent=2) + "\n")
    return passed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    return 0 if asyncio.run(probe(args.model, args.output)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
