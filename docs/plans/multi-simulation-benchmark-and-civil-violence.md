# Multi-Simulation Benchmark and Civil Violence Implementation Plan

**Status:** Proposed; implementation has not started.

**Goal:** Make `mam-bench run CONFIG.yaml` execute the Cartesian product of
YAML-selected built-in Benchmark Simulations and models, while leaving each
Benchmark Simulation in full control of its scientific lifecycle. Prove the
architecture first with the existing Schelling pilot, then implement a separate,
source-backed Epstein Civil Violence Benchmark Simulation.

## Hard constraints

- Preserve the terminology in `CONTEXT.md`.
- Preserve the one-way dependency: `mam_bench_analysis` may import `mam_bench`;
  runtime code must not import `mam_bench_analysis` or Plotly.
- YAML selects built-in Benchmark Simulations and models. It does not override
  simulation mechanics, profiles, objectives, scoring, evidence, datasets, or
  Python import paths.
- Compatibility Preflight checks the complete simulation-model matrix before
  any model call.
- Each Benchmark Simulation owns its Model Interaction Protocol, ordinary
  dynamics, counterfactual or reference conditions, Primary Score, validation,
  Run Evidence, and Diagnostic Artifacts.
- Every Primary Score is higher-is-better and comparable only within its
  Benchmark Simulation. There is no cross-simulation aggregate.
- A scored result exists only after the Benchmark Simulation reloads and
  validates all required Run Evidence.
- Default Reference Datasets ship with the repository when practical. Runtime
  never builds or downloads them implicitly.
- Changing simulation behavior means changing code and explicitly rebuilding
  its Reference Dataset. There is no custom-profile mode.
- Civil Violence scientific choices must be frozen before its implementation.
  This plan must not invent its model-controlled role or Primary Score.

## Design-it-twice result

Four interface shapes were evaluated against the current Schelling code and a
future Civil Violence implementation.

### 1. Generic case mapping

A minimal interface accepted `simulation_id`, `runtime_id`, and an untyped
simulation-specific `case` mapping.

**Rejected:** it makes YAML a latent simulation-parameter interface and recreates
the custom-profile machinery that the architecture explicitly excludes.

### 2. Typed per-simulation YAML union

A flexible interface used a discriminated Pydantic branch for every simulation,
with fields such as Schelling tolerance, vacancy, seed, and objective.

**Rejected for ordinary benchmark execution:** it is safer than an untyped
mapping, but still exposes mechanics through YAML. It also forces the shared
configuration module to know each simulation's domain language.

### 3. Caller-first built-in selection

A caller-first interface made YAML select only built-in simulation IDs and model
bindings. The selected simulation supplied every scientific parameter.

**Chosen:** this gives the ordinary caller the smallest interface and preserves
locality inside each Benchmark Simulation.

### 4. Port-heavy Model Runtime

A ports-and-adapters interface separated provider-neutral model requests from
PydanticAI, OpenRouter, and OpenAI-compatible details.

**Partially chosen:** the Model Runtime seam is necessary because production and
deterministic fake adapters already exist. The first interface will include only
capabilities used by Schelling or demonstrated by the Civil Violence design. It
will not become a general agent framework.

## Recommended module shape

```text
src/mam_bench/
  benchmark.py                 # shared result and preflight value types
  config.py                    # strict YAML selection contract
  registry.py                  # explicit built-in allowlists
  runner.py                    # preflight, deterministic execution, topline
  runtime.py                   # provider-neutral Model Runtime interface
  runtimes/
    pydantic_ai.py             # PydanticAI/OpenRouter/OpenAI-compatible adapter
  simulations/
    schelling.py               # initial deep adapter over existing implementation
    civil_violence/            # added only after scientific gates pass
      profile.py
      reference.py
      interaction.py
      scoring.py
      evidence.py
      simulation.py
```

The initial Schelling adapter should wrap the current `profile.py`,
`reference.py`, `dataset.py`, `influence.py`, and `pydantic_team.py` rather than
move them in the same change. Physical relocation is a later expand-contract
refactor after behavior is protected at the new seam.

## Exact configuration interface

### YAML v1

```yaml
schema_version: mam-bench.run.v1

simulations:
  - schelling-influence-pilot-v1

models:
  - id: qwen-3-8-27b
    runtime: openrouter
    model: qwen/qwen3.8-27b
    provider: phala

  - id: local-model
    runtime: openai-compatible
    model: local/model
    base_url: http://127.0.0.1:8000/v1
    api_key_env: LOCAL_MODEL_API_KEY

output:
  directory: results/comparison-001
  retain_diagnostic_artifacts: false
```

### Configuration invariants

- The root is one mapping with exactly `schema_version`, `simulations`, `models`,
  and `output`.
- `schema_version` equals `mam-bench.run.v1`.
- `simulations` is a non-empty list of unique built-in simulation IDs.
- `models` is a non-empty list with unique, path-safe `id` values.
- Each model uses one closed, discriminated runtime configuration.
- Secrets never appear in YAML. `api_key_env` names an environment variable;
  Compatibility Preflight checks presence without reading or recording its value.
- OpenRouter fallback routing is disabled by the adapter and is not configurable
  in YAML.
- Sampling controls, context policy, prompts, tools, rounds, objectives,
  reference paths, and evidence switches are absent from YAML.
- `output.directory` resolves relative to the YAML file and must be new or empty.
- `retain_diagnostic_artifacts` never suppresses Run Evidence.
- YAML aliases, merge keys, custom tags, duplicate keys, and multiple documents
  are rejected.
- Unknown fields fail validation.
- Execution order is simulation order, then model order.
- Every selected simulation runs against every selected model.

### Pydantic configuration types

```python
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


class OpenRouterModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    runtime: Literal["openrouter"]
    model: str
    provider: str


class OpenAICompatibleModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    runtime: Literal["openai-compatible"]
    model: str
    base_url: str
    api_key_env: str


ModelSelection = Annotated[
    OpenRouterModel | OpenAICompatibleModel,
    Field(discriminator="runtime"),
]


class OutputSelection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    directory: Path
    retain_diagnostic_artifacts: bool = False


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["mam-bench.run.v1"]
    simulations: tuple[str, ...]
    models: tuple[ModelSelection, ...]
    output: OutputSelection
```

Public parsing interface:

```python
def load_benchmark_config(path: Path) -> BenchmarkConfig: ...
```

Implementation requires a real YAML parser. The proposed dependency is PyYAML
with a repository-owned strict `SafeLoader` that rejects duplicate keys,
aliases, merges, custom tags, and multi-document input. Adding the dependency
requires explicit approval at implementation time; this plan does not install it.

## Exact Model Runtime interface

The Model Runtime is the true-external seam. It owns provider execution and
translation. It does not own observations, tools, histories, coordination,
actions, or policy-failure meaning.

```python
from enum import StrEnum
from typing import Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

JsonValue: TypeAlias = (
    None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
)


class RuntimeCapability(StrEnum):
    TEXT_OUTPUT = "text-output"
    STRICT_STRUCTURED_OUTPUT = "strict-structured-output"
    TOOL_RESULT_CONTINUATION = "tool-result-continuation"
    REQUEST_SEED = "request-seed"
    AUDITABLE_MESSAGES = "auditable-messages"


class SamplingControl(StrEnum):
    TEMPERATURE = "temperature"
    TOP_P = "top-p"
    TOP_K = "top-k"
    REASONING_EFFORT = "reasoning-effort"
    MAX_OUTPUT_TOKENS = "max-output-tokens"


class RuntimeDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    runtime: str
    provider: str
    model: str
    adapter_version: str
    capabilities: frozenset[RuntimeCapability]
    sampling_controls: frozenset[SamplingControl]
    max_concurrent_requests: int = Field(ge=1)
    required_environment: tuple[str, ...]


class RuntimeRequirements(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    capabilities: frozenset[RuntimeCapability]
    sampling_controls: frozenset[SamplingControl]
    minimum_concurrent_requests: int = Field(ge=1)


class TextPart(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["text"] = "text"
    text: str


class ToolCallPart(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["tool-call"] = "tool-call"
    call_id: str
    name: str
    arguments: dict[str, JsonValue]


class ToolResultPart(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["tool-result"] = "tool-result"
    call_id: str
    name: str
    result: JsonValue


RuntimePart = TextPart | ToolCallPart | ToolResultPart


class RuntimeMessage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["system", "user", "assistant", "tool"]
    parts: tuple[RuntimePart, ...]


class ToolDefinition(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    description: str
    input_schema: dict[str, JsonValue]
    strict: bool = True


class TextOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["text"] = "text"


class StructuredOutput(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["structured"] = "structured"
    name: str
    description: str
    schema: dict[str, JsonValue]
    strict: bool = True


class SamplingSettings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    temperature: float
    top_p: float
    top_k: int
    reasoning_effort: str
    max_output_tokens: int = Field(gt=0)
    request_seed: int | None = Field(default=None, ge=0, le=2**32 - 1)


class RequestLimits(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_limit: int = Field(ge=1)
    tool_call_limit: int = Field(ge=0)
    timeout_seconds: float = Field(gt=0)


class ModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    messages: tuple[RuntimeMessage, ...]
    tools: tuple[ToolDefinition, ...] = ()
    output: TextOutput | StructuredOutput
    sampling: SamplingSettings
    limits: RequestLimits


class ModelUsage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    requests: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    request_id: str
    messages: tuple[RuntimeMessage, ...]
    output: str | JsonValue
    usage: ModelUsage
    latency_seconds: float = Field(ge=0)
    provider_response_id: str | None = None


class ModelRuntime(Protocol):
    @property
    def descriptor(self) -> RuntimeDescriptor: ...

    async def complete(self, request: ModelRequest) -> ModelResponse: ...
```

### Runtime invariants

- Repository-owned types cross the seam; PydanticAI and provider SDK types do not.
- The Benchmark Simulation supplies the Model Interaction Protocol and all
  sampling settings.
- The runtime translates every requested control or rejects it during preflight.
  It never drops or approximates a control silently.
- `REQUEST_SEED` means that the adapter accepts and transmits the seed. It does
  not claim that a remote provider is deterministic.
- Strict schemas are requested and validated locally; remote compliance cannot
  be guaranteed by preflight.
- Provider availability, credentials, rate limits, and remote model existence
  cannot be proven without a call and are therefore execution concerns.
- Infrastructure, timeout, and provider errors raise
  `RuntimeInfrastructureError`. Malformed model output is returned or raised as
  a protocol result for the Benchmark Simulation to classify under its frozen
  failure rules.
- Raw provider payloads may be retained as Diagnostic Artifacts. Required
  auditable messages use the provider-neutral schema above.

## Exact Benchmark Simulation interface

The simulation seam stays deliberately smaller than either simulation's
implementation.

```python
from pathlib import Path
from typing import Protocol


class SimulationDescriptor(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_id: str
    simulation_version: str
    title: str
    primary_score_name: str


class PreflightIssue(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_id: str
    model_id: str | None
    code: str
    message: str


class PrimaryScore(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    value: float
    higher_is_better: Literal[True] = True
    objective: str
    meaning: str
    unit: str
    semantics_version: str


class EvidenceReceipt(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    relative_directory: str
    manifest_path: str
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SimulationResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation: SimulationDescriptor
    model: RuntimeDescriptor
    primary_score: PrimaryScore
    evidence: EvidenceReceipt
    diagnostic_artifacts_retained: bool


class PreparedSimulation(Protocol):
    @property
    def descriptor(self) -> SimulationDescriptor: ...

    @property
    def runtime_requirements(self) -> RuntimeRequirements: ...

    async def execute(
        self,
        *,
        runtime: ModelRuntime,
        output_directory: Path,
        retain_diagnostic_artifacts: bool,
    ) -> None: ...


class BenchmarkSimulation(Protocol):
    @property
    def descriptor(self) -> SimulationDescriptor: ...

    def prepare(self) -> PreparedSimulation: ...

    def validate(
        self,
        *,
        output_directory: Path,
        runtime: RuntimeDescriptor,
    ) -> SimulationResult: ...
```

### Simulation invariants

- `prepare()` performs deterministic, model-independent preparation: validates
  the frozen profile and shipped Reference Dataset, constructs immutable
  scientific inputs, and declares runtime requirements. It performs no model
  calls and writes no run output.
- The runner prepares every selected simulation before evaluating any pairing.
- `PreparedSimulation.execute()` owns the complete simulation lifecycle and
  writes its Run Evidence. The runner never advances a round or interprets an
  action.
- `BenchmarkSimulation.validate()` reloads evidence, recomputes the Primary
  Score, and returns the only score eligible for topline publication.
- `validate()` must fail on absent, extra, path-escaping, hash-mismatched, or
  semantically invalid required evidence.
- Simulation-specific records do not enter `SimulationResult`.
- The same Benchmark Simulation implementation that executes a run validates it.
- Interrupted or infrastructure-failed attempts may retain checkpoints and
  failure evidence but cannot return `SimulationResult`.

## Explicit built-in registry

```python
from types import MappingProxyType


_SIMULATIONS = MappingProxyType({
    "schelling-influence-pilot-v1": SchellingInfluencePilotV1(),
})

_RUNTIME_FACTORIES = MappingProxyType({
    "openrouter": build_openrouter_runtime,
    "openai-compatible": build_openai_compatible_runtime,
})


def resolve_simulation(simulation_id: str) -> BenchmarkSimulation: ...

def build_runtime(model: ModelSelection) -> ModelRuntime: ...
```

There is no registration function, package entry point, plugin discovery, or
YAML import path. Civil Violence later adds one explicit import and one registry
entry after its scientific and evidence gates pass.

## Compatibility Preflight and runner interface

```python
class PreparedPairing(BaseModel):
    # Private construction; contains resolved implementations and paths.
    ...


class PreparedBenchmark:
    # Opaque immutable output of complete preflight.
    ...


def preflight_benchmark(config: BenchmarkConfig) -> PreparedBenchmark: ...


async def run_benchmark(prepared: PreparedBenchmark) -> BenchmarkTopline: ...
```

Preflight order:

1. Validate YAML syntax and the closed Pydantic model.
2. Resolve every simulation and runtime adapter from built-in registries.
3. Reject duplicate IDs, unsafe paths, nested run directories, and output
   conflicts.
4. Check required environment variable names without reading values.
5. Call `prepare()` once for every selected Benchmark Simulation.
6. Validate every shipped Reference Dataset named by each prepared simulation.
7. Build a fresh Model Runtime descriptor for every simulation-model pairing.
8. Compare required capabilities, sampling controls, and concurrency.
9. Let each runtime adapter prove that it can translate the exact requirements
   without dropping fields.
10. Aggregate all issues in simulation order then model order.
11. On any issue, raise `CompatibilityPreflightError` containing every issue.
    Make zero model calls and create no output directory.
12. Otherwise return an opaque `PreparedBenchmark` that fingerprints the config,
    simulation versions, runtime descriptors, and output plan.

Execution order:

1. Execute pairings sequentially in simulation order then model order.
2. Give each pairing a fresh runtime instance and a new run directory:
   `runs/<simulation-id>/<model-id>/`.
3. After execution, call the simulation validator against the retained evidence.
4. Stop on the first execution or validation failure. Preserve that pairing's
   unscored evidence and any earlier validated evidence, but do not publish a
   complete topline file.
5. After every pairing validates, atomically write `topline.json` and print each
   score grouped by Benchmark Simulation.

Sequential execution is an implementation policy hidden behind the runner
interface. Parallel pairings remain out of scope until rate limits, cost control,
and output isolation justify the added complexity.

## Topline interface

```python
class ToplineEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    simulation_id: str
    simulation_version: str
    model_id: str
    runtime: str
    provider: str
    model: str
    primary_score: PrimaryScore
    evidence: EvidenceReceipt


class BenchmarkTopline(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["mam-bench.topline.v1"]
    status: Literal["complete"]
    entries: tuple[ToplineEntry, ...]
```

`BenchmarkTopline` has no root score, model ranking, normalization, weighted
composite, or cross-simulation comparison.

Example CLI output:

```text
schelling-influence-pilot-v1
  qwen-3-8-27b  directional_lift = 0.245079
  local-model   directional_lift = -0.013420

topline: results/comparison-001/topline.json
```

## Run Evidence interface

Every scored run contains a simulation-owned `evidence-manifest.json` written
last. The shared receipt does not define which evidence a simulation requires.

```python
class EvidenceFile(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str
    schema_version: str
    relative_path: str
    byte_size: int = Field(gt=0)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class EvidenceManifest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: Literal["mam-bench.evidence-manifest.v1"]
    simulation_id: str
    simulation_version: str
    required: tuple[EvidenceFile, ...]
    diagnostics: tuple[EvidenceFile, ...]
```

For Schelling Influence Profile v1, all artifacts required by the frozen profile
remain Run Evidence: `run.json`, `trajectory.npz`, `events.jsonl`,
`coordination-board.jsonl`, complete actor histories, and complete checkpoints.
The diagnostic flag may suppress only future renderings or analysis outputs that
the profile does not require.

Before adapting Schelling, reconcile the implementation with the frozen evidence
contract. In particular, tool calls and phase boundaries named by the profile
must be represented and validated; wrapping the current incomplete evidence is
not sufficient.

## Test seams

These seams are already consistent with the confirmed architecture:

1. **Runner seam** - `preflight_benchmark()` and `run_benchmark()` with two
   unrelated fake Benchmark Simulations and fake Model Runtimes.
2. **Model Runtime seam** - provider-neutral requests/responses, a deterministic
   scripted adapter, a capability-limited adapter, and the PydanticAI adapter.
3. **Benchmark Simulation seam** - each simulation's `prepare`, `execute`, and
   `validate` behavior, tested without the real provider.
4. **CLI seam** - subprocess execution of `mam-bench run CONFIG.yaml`.
5. **Evidence seam** - load, tamper, delete, and recompute through the selected
   simulation's validator.

Tests assert external behavior, not registry internals or simulation-private
helpers.

## Implementation sequence

Each slice should use red-green TDD and remain green independently.

### Slice 1: Lock the simulation and runner seams with fakes

#### Slice 1 build

- Add shared value types and errors.
- Add `BenchmarkSimulation`, `PreparedSimulation`, and Model Runtime protocols.
- Add a runner that accepts injected test registries.

#### Slice 1 tests first

- Two unrelated fake simulations execute without shared domain fields.
- Scores remain separate and higher-is-better.
- One incompatible pairing prevents every runtime call.
- Preflight reports all incompatible pairings in deterministic order.
- Required evidence failure prevents topline publication.
- Output order follows simulation order then model order.

#### Slice 1 acceptance

- No Schelling vocabulary appears in runner or runtime interfaces.
- No filesystem output exists after failed preflight.

### Slice 2: Add strict YAML and built-in registries

#### Slice 2 build

- Add strict YAML loading and the closed v1 Pydantic configuration.
- Add explicit built-in simulation and runtime adapter registries.
- Add path planning for the full simulation-model matrix.

#### Slice 2 tests first

- Parse the exact v1 example.
- Reject aliases, merges, custom tags, duplicate keys, multiple documents,
  unknown fields, duplicate IDs, unsafe IDs, and mechanics fields.
- Reject arbitrary Python paths and generic parameter mappings.
- Prove models and simulations form the expected Cartesian product.

#### Slice 2 acceptance

- Adding a simulation requires source code and one explicit registry entry.
- YAML contains no simulation-specific field.

#### Slice 2 authorization gate

- Obtain explicit approval before adding PyYAML with `uv add`.

### Slice 3: Extract the Model Runtime adapter

#### Slice 3 build

- Add provider-neutral messages, output contracts, capabilities, sampling
  settings, usage, and errors.
- Move PydanticAI/OpenRouter/OpenAI-compatible construction and translation out
  of the Schelling Model Interaction Protocol.
- Keep `InfluenceTeam` as the Schelling-internal production/test seam.
- Make a runtime-backed Influence Team own Schelling prompts, Round Window,
  Coordination Board, actor histories, and policy-failure conversion.

#### Slice 3 tests first

- Scripted runtime records exact Schelling requests and deterministic seed values.
- Capability-limited runtimes fail preflight without calls.
- PydanticAI translation retains Phala routing, no fallback, top-k, reasoning,
  limits, and auditable messages.
- SDK types never cross the Model Runtime seam.
- Existing `FixedInfluenceTeam` mechanics tests remain unchanged.

#### Slice 3 acceptance

- Runtime modules contain no Schelling board, actor, movement, homophily, or
  Counterfactual Reference vocabulary.
- Schelling protocol modules contain no OpenRouter or OpenAI SDK construction.

### Slice 4: Adapt the fixed Schelling pilot

#### Slice 4 build

- Add built-in `schelling-influence-pilot-v1`.
- `prepare()` freezes tolerance `3/4`, vacancy `25%`, Evaluation Seed `22`,
  integration, 20 rounds, and the current model sampling contract.
- `execute()` delegates to the existing Schelling implementation through a
  runtime-backed Influence Team.
- `validate()` reloads evidence and maps recomputed final Directional Lift to the
  Primary Score.
- Add the evidence manifest and close gaps between emitted evidence and the
  frozen profile.

#### Slice 4 tests first

- Existing Counterfactual Reference values remain bit-exact.
- Existing fake-team twenty-round behavior remains bit-exact.
- Score equals independently validated final Directional Lift.
- Deleting or altering any required artifact makes the run unscored.
- Diagnostic suppression removes no current Schelling Run Evidence.

#### Slice 4 acceptance

- The adapter exposes none of the Schelling lifecycle to the runner.
- The existing `agent-pilot` command delegates to the same adapter during
  migration; no second implementation exists.

### Slice 5: Ship and validate the default Schelling Reference Dataset

#### Slice 5 build

- Track the validated v1 cell archives currently excluded by `.gitignore`, or
  place the same validated data in an equivalent repository-owned package-data
  location.
- Keep construction in `modal_reference_sweep.py` and runtime consumption
  separate.
- Make Schelling `prepare()` validate the shipped manifest and every required
  cell before any model call.

#### Slice 5 tests first

- A clean checkout contains the complete 23 by 7 dataset.
- Manifest hashes and scientific profile validate.
- Missing or changed cells fail preflight with zero model calls.
- Ordinary execution never calls reference construction or network download.

#### Slice 5 acceptance

- Clone-and-run works without rebuilding the Reference Landscape.
- The roughly 6.2 MiB dataset remains practical to ship directly.

### Slice 6: Add the YAML CLI and topline publication

#### Slice 6 build

- Add `mam-bench run CONFIG.yaml`.
- Run complete Compatibility Preflight, sequential execution, simulation
  validation, atomic topline publication, and grouped score printing.
- Preserve existing reference-construction and validation commands.

#### Slice 6 tests first

- CLI success writes a complete `topline.json` and prints grouped scores.
- Config/preflight errors exit `2`, make zero model calls, and write no output.
- Execution/validation errors exit `1`, preserve unscored evidence, and write no
  complete topline.
- Importing the runtime CLI does not import `mam_bench_analysis` or Plotly.

#### Slice 6 acceptance

- A cloned repository can run the shipped Schelling pilot with one YAML command.

### Slice 7: Localize Schelling physically

This is a separate, explicitly approved broad refactor after the new seam is
stable.

#### Slice 7 build

- Move Schelling profile, reference, dataset, influence, interaction, scoring,
  and evidence implementation under `simulations/schelling/`.
- Use temporary re-export shims only where repository consumers still require
  old imports.
- Remove shims after all consumers migrate.

#### Slice 7 acceptance

- Scientific expectations remain bit-exact.
- The runner, config, runtime, and registry contain no Schelling domain terms.
- Analysis still imports runtime artifacts one-way.

## Civil Violence plan

Civil Violence begins only after the multi-simulation foundation and Schelling
parity are complete.

### Scientific facts that constrain the plan

Epstein's 2002 paper defines two different models. Model I covers rebellion
against central authority; Model II adds inter-group violence, killing, birth,
death, and peacekeeping. They cannot share an unqualified implementation.

For Model I:

- Citizens have fixed hardship and risk aversion; legitimacy is global.
- Citizens are quiet, active, or jailed.
- Cops and non-jailed citizens move on a toroidal lattice.
- Activation is asynchronous: agents activate in random order, move, then apply
  their citizen or cop rule.
- Randomness affects initialization, traits, activation order, movement, arrest
  targets, and jail terms.
- The paper does not freeze a benchmark-ready RNG contract, implementation edge
  cases, or universal stopping rule.
- The paper demonstrates environmental interventions, not model-controlled local
  agents, and does not provide a MAM-Bench Primary Score.

NetLogo and Mesa are not interchangeable oracles. NetLogo floors the visible
cop/active ratio; Mesa rounds it; the paper specifies neither adjustment. Other
movement, occupancy, jail, and scheduling details also diverge.

Sources:

- Joshua M. Epstein, ["Modeling Civil Violence: An Agent-Based Computational
  Approach"](https://pmc.ncbi.nlm.nih.gov/articles/PMC128592/), PNAS
  99(Suppl. 3), 2002.
- Epstein, Steinbruner, and Parker,
  [Brookings Working Paper No. 20](https://www.brookings.edu/wp-content/uploads/2016/06/cviolence.pdf),
  2001.
- [NetLogo Rebellion model](https://ccl.northwestern.edu/netlogo/models/Rebellion).
- [Mesa Epstein Civil Violence example](https://mesa.readthedocs.io/latest/examples/advanced/epstein_civil_violence.html).

### Civil Slice 1: Freeze the ordinary scientific profile

#### Civil Slice 1 decision gate

Create and accept a Civil Violence Reference Profile ADR and profile document.
Resolve before implementation:

- Model I versus Model II and scientific lineage;
- neighborhood and distance rules;
- fixed counts versus probabilistic density initialization;
- arrest-probability numeric semantics, including floor/round/neither;
- exact asynchronous activation, mutation visibility, movement, arrest, jail,
  and release ordering;
- occupancy of jailed citizens;
- RNG implementation and named streams;
- horizon and checkpoint cadence;
- invariants and failure conditions.

**Recommendation to evaluate, not a frozen decision:** start with a named
Model-I-derived modernized profile because it is narrower and has NetLogo/Mesa
implementations available for comparison.

### Civil Slice 2: Build conformance fixtures before the engine

#### Civil Slice 2 tests first

- Hand-calculated grievance and arrest-probability examples.
- Exact threshold-boundary cases.
- Tiny lattice movement and visibility cases.
- Asynchronous order cases where a later activation observes an earlier change.
- Arrest target and jail-term cases.
- Jail decrement and release timing cases.
- Impossible occupancy and illegal activation failures.

Expected values must come from the frozen profile or hand calculation, not from
the implementation under test.

### Civil Slice 3: Implement the ordinary engine

#### Civil Slice 3 build

- Host-neutral initialization, one-period transition, checkpoint, resume, and
  validation.
- Stable citizen/cop identities and complete state.
- Named independent RNG streams for traits, placement, activation order,
  movement, arrest target, and jail term.
- Replayable ordered event stream plus sufficient checkpoints.

#### Civil Slice 3 acceptance

- Fixed seeds replay bit-exactly.
- Event replay reconstructs every retained state.
- The engine makes no model calls and has no provider dependency.

### Civil Slice 4: Characterize ordinary behavior

#### Civil Slice 4 build

- Run a small pilot matrix before freezing a large sweep.
- Measure active and jailed counts, outburst size/frequency/duration, waiting
  times, and spatial diagnostics.
- Use multiple Landscape Seeds and held-out Evaluation Seeds.
- Measure event volume and select a practical checkpoint cadence.

#### Civil Slice 4 acceptance

- The profile exhibits non-degenerate, stochastic regimes.
- Reference storage remains replayable and practical.
- Selected evaluation conditions come from ordinary behavior, not desired model
  performance.

### Civil Slice 5: Freeze and ship the Reference Dataset

#### Civil Slice 5 build

- Freeze the parameter landscape, seed derivation, held-out panel, manifest,
  hashes, provenance, and validation.
- Ship the default validated data when practical.
- Keep any cloud execution as an adapter over host-neutral generation.

#### Civil Slice 5 acceptance

- Clean-checkout validation succeeds without construction or download.
- Changed mechanics require explicit code changes and explicit reconstruction.

### Civil Slice 6: Design the Model Interaction Protocol

#### Civil Slice 6 decision gate

Freeze a separate intervention profile only after ordinary behavior is accepted.
Resolve:

- which local identities the model controls: citizens, cops, or another explicit
  role;
- controlled count and selection;
- public versus private observations;
- local visibility and any communication;
- exact asynchronous microstep at which model inference occurs;
- valid actions and collision/priority semantics;
- treatment of controlled identities in metrics;
- matched counterfactual and RNG stream pairing;
- policy versus infrastructure failures;
- required runtime capabilities.

Do not replace the asynchronous engine with Schelling-style synchronized rounds.
Do not use global legitimacy schedules merely because the paper studies them;
that may conflict with MAM-Bench's locally acting-agent objective.

### Civil Slice 7: Establish a Primary Score and Run Evidence

#### Civil Slice 7 build

- Evaluate candidate metrics against ordinary, random, scripted, and
  counterfactual controls.
- Measure variance, discrimination, directionality, and gameability.
- Designate one higher-is-better Primary Score only after the evidence supports
  it.
- Freeze mandatory Run Evidence sufficient to replay every intervention and
  recompute the score.

#### Civil Slice 7 required evidence candidates

- profile and schema versions;
- complete parameters, identities, traits, initial positions, and RNG derivation;
- ordered activation, movement, state-decision, arrest, and jail events;
- retained checkpoints or trajectories sufficient for replay;
- every model observation, response, validated action, and failure;
- counterfactual identity and paired streams;
- software/runtime provenance and artifact hashes;
- independently recomputable metrics.

### Civil Slice 8: Implement and register the Benchmark Simulation

#### Civil Slice 8 build

- Implement `civil_violence/interaction.py`, `scoring.py`, `evidence.py`, and
  `simulation.py` behind the existing Benchmark Simulation and Model Runtime
  seams.
- Test with deterministic fake runtimes before any live call.
- Add one explicit registry entry only after profile, score, and evidence
  validation pass.

#### Civil Slice 8 acceptance

- Adding Civil Violence changes its own module and one registry entry.
- Runner, YAML parser, topline schema, and Schelling implementation require no
  domain-specific edits.
- A two-simulation YAML run produces separate Primary Scores with no aggregate.
- One incompatible pairing fails the complete matrix before any model call.

## Global verification

Run after every implementation slice:

```bash
PYTHONPATH=src uv run python -m unittest discover -s tests -v
uv run ruff check src tests modal_reference_sweep.py
uv run pyright
```

Critical-path gates:

```bash
./bin/mam-bench run examples/schelling-pilot.yaml
./bin/mam-bench validate-evaluation \
  results/schelling-pilot/runs/schelling-influence-pilot-v1/qwen-3-8-27b
```

Before the Civil Violence treatment panel, require:

- ordinary-engine conformance tests;
- Reference Dataset validation from a clean checkout;
- deterministic fake-runtime full lifecycle;
- evidence tamper and replay tests;
- one complete live pilot with no infrastructure failure.

## Explicitly out of scope

- Implementing any code in this planning step.
- Third-party simulation plugins or YAML import paths.
- Custom simulation profiles or mechanics overrides in YAML.
- Cross-simulation score normalization or ranking.
- Parallel pairing execution.
- Automatic reference construction or download during benchmark execution.
- Choosing a Civil Violence model-controlled role or Primary Score before the
  ordinary model and reference behavior are accepted.
- Treating NetLogo or Mesa as a silent paper-exact oracle.
