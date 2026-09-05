# MAM-Bench

> [!WARNING]
> **Work in progress:** MAM-Bench is an early research benchmark. Its interfaces,
> evaluation profiles, datasets, and scoring may change before the first stable
> release. Do not treat current results as a mature or standardized benchmark.

MAM-Bench first characterizes one explicit Schelling segregation model, then tests whether separate model-controlled Influence Actors can steer its emergent outcome. Schelling Reference Dataset v2 expands the ordinary-agent landscape across three board sizes while preserving the first influence benchmark.

Read [Benchmarking Self-Organized AI Swarms in Steering Complex Systems](https://joehelbing.net/post/mam-bench) for an illustrated overview of the current pilot.

## Schelling Reference Dataset v2

The frozen profile uses:

- 20x20, 60x60, and 100x100 toroidal boards;
- two equally sized groups;
- a radius-1 Moore neighborhood;
- exact fractional tolerance over occupied neighbors;
- stationary satisfied agents;
- sequential reservation of nearest satisfactory vacancies against a frozen state;
- staged application of all reserved moves; and
- equilibrium, blocked, or 30-transition termination.

The landscape crosses all 23 behaviorally distinct radius-1 tolerance fractions with vacancy fractions `1/100`, `1/40`, `1/20`, `1/10`, `1/5`, `1/4`, `3/10`, `2/5`, and `1/2`. Each of the 621 cells receives 50 paired Landscape Seeds, producing 31,050 runs.

This is a modernized profile, not a claim to reproduce Schelling's original checkerboard procedure exactly. See [`docs/schelling-reference-profile-v2.md`](docs/schelling-reference-profile-v2.md) for the complete scientific, randomness, and artifact contract. The v1 profile remains as historical documentation.

## Install

The project uses Python 3.14, NumPy, Pydantic 2, PydanticAI, PydanticAI Harness, PyYAML, and uv. The checked-in mise configuration pins the development toolchain and provides common tasks:

```bash
mise install
mise run setup
```

If the toolchain is already available, `uv sync` is sufficient.

Modal is used only as a cloud execution adapter. The engine itself runs and tests locally.

Compact validated reference data ships under `src/mam_bench/data/schelling-reference-v2/`: a 621-cell aggregate landscape and the fixed Seed 50 evaluation fixture. The simulation loads the fixture with `importlib.resources`; it never downloads or reconstructs reference data implicitly. Full trajectories are developer-only material under ignored `.scratch` and are not included in clones or wheels.

## Reconstruct the complete sweep on Modal

This is an explicit developer workflow, not part of ordinary benchmark execution. Use it after an intentional Reference Profile mechanics change to reconstruct and revalidate the shipped Reference Dataset.

The Modal CLI must already be authenticated. The adapter runs one CPU task per Landscape Cell and writes each returned artifact locally as it completes. Rerunning the command skips artifacts whose resumable receipts and SHA-256 hashes remain valid.

```bash
modal run src/mam_bench/simulations/schelling/utils/modal_reference_sweep.py \
  --output-dir .scratch/schelling-reference-v2-full \
  --receipt-path .scratch/schelling-reference-v2-receipt.json
```

After all 621 full-trajectory artifacts arrive, validate them and publish only the compact products used by ordinary evaluation and landscape inspection:

```bash
uv run src/mam_bench/simulations/schelling/utils/publish_reference_data.py
```

## Artifact layout

```text
src/mam_bench/data/schelling-reference-v2/
|-- landscape.jsonl
|-- evaluation-reference.json
`-- evaluation-reference.npz
```

`landscape.jsonl` contains one aggregate outcome record for each of the 621 Landscape Cells. The evaluation fixture contains only the fixed Seed 50 initial grid, stable locations and types, terminal Counterfactual Reference state, status, rounds, and comparison metrics. Its JSON metadata records the NPZ hash. The selected test set holds vacancy at 25% and uses preference regimes `1/2`, `3/4`, and `6/7`; the current benchmark uses `3/4`.

The complete 31-state trajectories and their artifact inventory remain under `.scratch/schelling-reference-v2-full/` for local scientific analysis. Because `.scratch` is ignored, preserve them separately if they need to outlive the working copy.

## Read the Schelling implementation

The runtime path is easiest to read in this order:

1. `benchmark.py` constructs the selected model and its shared request settings.
2. `simulation.py` selects the fixed case and packaged reference.
3. `runtime.py` runs rounds, applies moves, scores outcomes, and writes evidence.
4. `agent.py` under `simulations/schelling/` defines the Schelling tools,
   terminal outputs, and turn policy.
5. `prompt.py` contains the Influence Actor instructions.
6. `models.py` defines the shared Schelling runtime data contracts.
7. `reference.py` contains the ordinary-agent Schelling mechanics.
8. `profile.py` fixes the parameter grid and deterministic seed contract.

The supporting modules at the package root have separate responsibilities:
`agent.py` owns persistent sessions, private Harness notebooks, and compaction;
`communication.py` owns ordered shared records and rolling admission;
`evidence.py` records normalized messages and events; and `usage.py` aggregates
native PydanticAI usage before serializing the benchmark's evidence fields.

`fixture.py` loads the compact comparator used at runtime. Everything involved in regenerating or analyzing reference data lives under `utils/`; see `utils/README.md`.

## Run a benchmark matrix from YAML

The ordinary workflow reads a selection-only YAML file, attempts every selected
simulation-model pairing, and writes `topline.json` after all pairings finish.
It prints scores when the full matrix succeeds; after a partial failure it
prints the aggregate failure while the successful scores remain in the topline:

```bash
uv run main.py examples/schelling-pilot.yaml
```

`main.py` accepts exactly one `.yaml` path. The YAML contract selects built-in
Benchmark Simulations and Model Runtimes. It cannot change mechanics, objectives,
prompts, tools, datasets, rounds, or scoring. The configured output directory
must not already exist; MAM-Bench never overwrites an earlier run.
Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` before running the
OpenRouter example. `OPENROUTER_BASE_URL` defaults to the standard endpoint and
can be overridden from the environment or `.env`. The ignored `.env` file must
never be committed. OpenAI-compatible YAML selections name their required API-key
environment variable directly; benchmark YAML never contains credentials.

Schelling Influence v2 uses the same 20x20 parameter cell with held-out Seed 50
from the Reference Dataset v2 randomness contract. Each selected model receives
one stochastic 30-round trial that retains 31 states. A successful pair writes
`run.json`, `trajectory.npz`, and canonical `events.jsonl`; a failed pair writes
`failure.json` and redacted `failed-events.jsonl` and remains unscored.

Independent pairs continue after a failure. `topline.json` contains successful
pairs only, and the command exits nonzero after the matrix if any pair failed.
A failed pair cancels and awaits its remaining actors, retains failure evidence,
and discards its runtime. It cannot resume; rerunning starts a fresh trial in a
new output directory. A partial topline therefore does not mean the complete
matrix passed. One trial is one sample, not a reliability estimate or confidence
interval. See the Influence Profile for the complete protocol, scoring, and
evidence contract.

## Verify the implementation

```bash
mise run check
```

The individual `test`, `lint`, and `typecheck` tasks are also available through
`mise run`. Tests use fake, scripted, or function-backed models and make no live
provider calls.

## Documentation

- [`CONTEXT.md`](CONTEXT.md): canonical project language.
- [`docs/schelling-reference-profile-v2.md`](docs/schelling-reference-profile-v2.md): current reference scientific and data contract.
- [`docs/schelling-influence-profile-v2.md`](docs/schelling-influence-profile-v2.md): current Influence Actor evaluation contract.
- [`docs/schelling-reference-profile-v1.md`](docs/schelling-reference-profile-v1.md): historical reference contract.
- [`docs/adr/0001-use-a-modernized-schelling-reference-profile.md`](docs/adr/0001-use-a-modernized-schelling-reference-profile.md)
- [`docs/adr/0002-map-behaviorally-distinct-tolerance-vacancy-cells.md`](docs/adr/0002-map-behaviorally-distinct-tolerance-vacancy-cells.md)
- [`docs/adr/0003-generate-the-reference-landscape-on-modal.md`](docs/adr/0003-generate-the-reference-landscape-on-modal.md)
- [`docs/adr/0004-select-three-final-satisfaction-test-spots.md`](docs/adr/0004-select-three-final-satisfaction-test-spots.md)
- [`docs/adr/0005-use-globally-informed-influence-actors.md`](docs/adr/0005-use-globally-informed-influence-actors.md)

## Method source

Thomas C. Schelling, "Dynamic Models of Segregation," *Journal of Mathematical Sociology* 1 (1971), 143-186.
