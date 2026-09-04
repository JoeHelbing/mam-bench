"""Built-in Schelling influence benchmark."""

from importlib.resources import as_file, files
from pathlib import Path

from mam_bench.benchmark import ModelRuntime, PrimaryScore

from .agent import RuntimeInfluenceTeam
from .fixture import load_evaluation_reference
from .models import InfluenceEvaluationConfig, SteeringObjective
from .profile import LandscapeCell, Rational
from .runtime import run_model_evaluation


class SchellingBenchmarkSimulation:
    """Run the fixed Schelling case and return its score."""

    simulation_id = "schelling-influence-pilot-v1"
    simulation_version = "schelling-influence-v1"

    async def run(
        self,
        runtime: ModelRuntime,
        output_directory: Path,
    ) -> PrimaryScore:
        resource = files("mam_bench.data").joinpath("schelling-reference-v2")
        with as_file(resource) as dataset_root:
            reference = load_evaluation_reference(dataset_root)
            result = await run_model_evaluation(
                InfluenceEvaluationConfig(
                    cell=LandscapeCell(20, Rational(3, 4), Rational(1, 4)),
                    seed_id=50,
                    objective=SteeringObjective.INTEGRATION,
                ),
                RuntimeInfluenceTeam(runtime),
                output_directory,
                reference=reference,
            )
        return PrimaryScore(
            name="directional_lift",
            value=result.final_directional_lift,
            unit="raw homophily fraction",
        )
