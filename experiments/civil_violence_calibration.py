"""Run 16 ordinary conditions and replay two provisional candidates; no model calls."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import TypedDict

import numpy as np

from mam_bench.simulations.civil_violence import CivilViolenceSettings, CivilViolenceSim
from mam_bench.simulations.civil_violence.models import Role


class Metrics(TypedDict):
    mean_activity: float
    mean_jailed: float
    first_half_activity: float
    second_half_activity: float
    peak_activity: float
    final_activity: float
    activity_series: list[float]
    jail_series: list[float]


class Record(Metrics):
    settings: dict[str, object]
    trajectory: str
    sha256: str


def summarize(path: Path) -> Metrics:
    """Calculate participation from saved arrays, independent of simulation metrics."""
    with np.load(path, allow_pickle=False) as data:
        citizens = data["roles"] == 0
        active = data["active"][1:, citizens] & ~data["jailed"][1:, citizens]
        jailed = data["jailed"][1:, citizens]
        activity_series = active.mean(axis=1)
        jail_series = jailed.mean(axis=1)
        return {
            "mean_activity": float(activity_series.mean()),
            "mean_jailed": float(jail_series.mean()),
            "first_half_activity": float(activity_series[: len(activity_series) // 2].mean()),
            "second_half_activity": float(activity_series[len(activity_series) // 2 :].mean()),
            "peak_activity": float(activity_series.max()),
            "final_activity": float(activity_series[-1]),
            "activity_series": activity_series.tolist(),
            "jail_series": jail_series.tolist(),
        }


def execute(settings: CivilViolenceSettings, path: Path) -> Record:
    trajectory = CivilViolenceSim(settings).run_reference()
    trajectory.write(path)
    metrics = summarize(path)
    if not np.isclose(metrics["mean_activity"], trajectory.mean_activity(), atol=1e-14):
        raise AssertionError("saved-array activity disagrees with public trajectory")
    return {
        "settings": settings.model_dump(),
        "trajectory": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/civil-calibration"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records: list[Record] = []
    for threshold in (1.0, 3.66356):
        for density in (0.03, 0.07):
            for seed in (50, 51):
                for duration in (30, 60):
                    settings = CivilViolenceSettings(
                        board_size=12,
                        citizen_density=0.7,
                        police_density=density,
                        citizen_vision=3,
                        police_vision=3,
                        threshold=threshold,
                        max_jail_term=10,
                        seed_id=seed,
                        max_transitions=duration,
                        controlled_agent_count=2,
                    )
                    record = execute(settings, args.output / f"ordinary-{len(records):02d}.npz")
                    records.append(record)
                    print(
                        json.dumps(
                            {
                                "trajectory": record["trajectory"],
                                "mean_activity": record["mean_activity"],
                                "mean_jailed": record["mean_jailed"],
                            }
                        ),
                        flush=True,
                    )
    # Transparent provisional ranking; these targets are study choices, not benchmark goals.
    candidates: dict[str, dict[str, object]] = {}
    targets: tuple[tuple[Role, float], ...] = (("citizen", 0.35), ("police", 0.5))
    for role, target in targets:
        selected = min(
            (r for r in records if r["settings"]["max_transitions"] == 60),
            key=lambda r: abs(r["mean_activity"] - target),
        )
        settings = CivilViolenceSettings.model_validate(selected["settings"])
        settings.validate_controlled_role(role)
        replay = execute(settings, args.output / f"{role}-replay.npz")
        with np.load(selected["trajectory"]) as original, np.load(replay["trajectory"]) as repeated:
            for field in original.files:
                np.testing.assert_array_equal(original[field], repeated[field])
        candidates[role] = {
            "selection_target": target,
            "original": selected["trajectory"],
            **replay,
        }
    report = {
        "profile": "binary epsilon-free Cascade; ordinary public lifecycle",
        "cascade_revision": "bd9598d0c813e1b72ee0d62abb7610d452b26313",
        "ordinary_runs": len(records) + len(candidates),
        "records": records,
        "candidates": candidates,
    }
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
