"""Built-in Schelling influence benchmark."""

from importlib.resources import as_file, files
from pathlib import Path
from typing import Never

from mam_bench.benchmark import (
    AgentInfrastructureFailure,
    ModelRuntime,
    PairFailure,
    PairFailureKind,
    PairInfrastructureFailure,
    PrimaryScore,
)

from .agent import RuntimeInfluenceTeam, SchellingTurnCoordinator
from .fixture import load_evaluation_reference
from .models import InfluenceEvaluationConfig, SteeringObjective
from .profile import LandscapeCell, Rational
from .runtime import (
    EvidencePublicationError,
    publish_v2_failure,
    publish_v2_success,
    run_v2_model_evaluation,
)


class SchellingBenchmarkSimulation:
    """Run the fixed Schelling case and return its score."""

    simulation_id = "schelling-influence-pilot-v1"
    simulation_version = "schelling-influence-v2"

    async def run(
        self,
        runtime: ModelRuntime,
        output_directory: Path,
    ) -> PrimaryScore:
        if output_directory.exists():
            raise FileExistsError(f"result directory already exists: {output_directory}")
        resource = files("mam_bench.data").joinpath("schelling-reference-v2")
        with as_file(resource) as dataset_root:
            reference = load_evaluation_reference(dataset_root)
            team = RuntimeInfluenceTeam(runtime)
            coordinator = SchellingTurnCoordinator(team.communication, team.evidence)
            try:
                result = await run_v2_model_evaluation(
                    InfluenceEvaluationConfig(
                        cell=LandscapeCell(20, Rational(3, 4), Rational(1, 4)),
                        seed_id=50,
                        objective=SteeringObjective.INTEGRATION,
                    ),
                    team,
                    coordinator,
                    reference=reference,
                    evidence=team.evidence,
                )
                await team.record_final_notebooks()
                publish_v2_success(output_directory, result, team.evidence.events)
            except AgentInfrastructureFailure as error:
                await self._fail_pair(runtime, output_directory, team, error.kind)
            except EvidencePublicationError:
                await self._fail_pair(
                    runtime,
                    output_directory,
                    team,
                    "evidence_publication",
                )
        return PrimaryScore(
            name="directional_lift",
            value=result.final_directional_lift,
            unit="raw homophily fraction",
        )

    async def _fail_pair(
        self,
        runtime: ModelRuntime,
        output_directory: Path,
        team: RuntimeInfluenceTeam,
        kind: PairFailureKind,
    ) -> Never:
        failure = PairFailure(
            simulation_id=self.simulation_id,
            simulation_version=self.simulation_version,
            model_id=runtime.info.model_id,
            provider=runtime.info.provider,
            model=runtime.info.model,
            kind=kind,
        )
        await team.evidence.record("pair_failed", failure_kind=kind)
        publish_v2_failure(output_directory, failure, team.evidence.events)
        raise PairInfrastructureFailure(failure)
