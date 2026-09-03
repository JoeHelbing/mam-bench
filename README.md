# MAM-Bench

> [!WARNING]
> **Work in progress:** MAM-Bench is an early research benchmark. Its interfaces,
> evaluation profiles, datasets, and scoring may change before the first stable
> release. Do not treat current results as a mature or standardized benchmark.

MAM-Bench first characterizes one explicit Schelling segregation model, then tests whether separate model-controlled Influence Actors can steer its emergent outcome. Reference Landscape v1 and the first live model-evaluation pilot are complete.

Read [Benchmarking Self-Organized AI Swarms in Steering Complex Systems](https://joehelbing.net/post/mam-bench) for an illustrated overview of the current pilot.

## Reference Landscape v1

The frozen profile uses:

- a 20x20 toroidal grid;
- two equally sized groups;
- a radius-1 Moore neighborhood;
- exact fractional tolerance over occupied neighbors;
- stationary satisfied agents;
- sequential reservation of nearest satisfactory vacancies against a frozen state;
- staged application of all reserved moves; and
- equilibrium, blocked, or 500-transition termination.

The landscape crosses all 23 behaviorally distinct radius-1 tolerance fractions with seven vacancy levels from 0.10 through 0.40. Each of the 161 cells receives 20 paired Landscape Seeds, producing 3,220 runs.

This is a modernized profile, not a claim to reproduce Schelling's original checkerboard procedure exactly. See [`docs/schelling-reference-profile-v1.md`](docs/schelling-reference-profile-v1.md) for the complete scientific, randomness, and artifact contract.

## Install

The project uses Python 3.14, NumPy, Pydantic 2, PydanticAI, PyYAML, and uv. The checked-in mise configuration pins the development toolchain and provides common tasks:

```bash
mise install
mise run setup
```

If the toolchain is already available, `uv sync` is sufficient.

Modal is used only as a cloud execution adapter. The engine itself runs and tests locally.

The complete validated Reference Dataset ships under
`data/reference-landscape/v1/`. Ordinary benchmark preparation validates those
repository-owned files before any Model Runtime call. It never downloads or
reconstructs the dataset implicitly.

## Reconstruct the complete sweep on Modal

This is an explicit developer workflow, not part of ordinary benchmark execution. Use it after an intentional Reference Profile mechanics change to reconstruct and revalidate the shipped Reference Dataset.

The Modal CLI must already be authenticated. The adapter runs one CPU task per Landscape Cell and writes each returned artifact locally as it completes. Rerunning the command skips artifacts whose resumable receipts and SHA-256 hashes remain valid.

```bash
modal run modal_reference_sweep.py \
  --output-dir data/reference-landscape/v1
```

After all 161 artifacts arrive, use the dataset APIs in
`mam_bench.simulations.schelling.dataset` to validate every archive, verify paired
initial grids, and publish the complete manifest. The dataset is not complete
until `manifest.json` exists and `validate_dataset()` succeeds.

## Artifact layout

```text
data/reference-landscape/v1/
|-- manifest.json
`-- cells/
    |-- t00-v00.npz
    |-- ...
    `-- t22-v06.npz
```

Each compressed NPZ archive contains 20 complete cell-type trajectories, stable per-agent location traces, static agent types, trajectory lengths, terminal statuses, seed IDs, and exact parameter metadata. NPZ files are loaded with `allow_pickle=False`. The Pydantic JSON manifest records the complete profile, array schema, code mappings, software provenance, byte sizes, and SHA-256 hashes.

Optional scientific diagnostics live in the separate, downstream
`mam_bench_analysis` package. Its analysis modules expose Python APIs for writing
`final-satisfaction-manifold.json`, `final-satisfaction-manifold.csv`, and model
evaluation diagnostics. The selected test set holds vacancy at 25% and uses three
preference regimes: `1/2`, `3/4`, and `6/7`. See ADR 0004 for the rationale.

Analysis artifacts are derived outputs, not Run Evidence or Primary Scores. The runtime package never imports the analysis package.

## Run a benchmark matrix from YAML

The ordinary workflow reads a selection-only YAML file, validates the complete Benchmark Simulation-Model Runtime matrix, executes each pairing, prints scores grouped by Benchmark Simulation, and writes `topline.json` only after all Run Evidence validates:

```bash
uv run main.py examples/schelling-pilot.yaml
```

`main.py` accepts exactly one `.yaml` path. The YAML contract selects built-in
Benchmark Simulations and Model Runtimes. It cannot change mechanics, objectives,
prompts, tools, datasets, rounds, or scoring. `examples/schelling-pilot.yaml`
requires `OPENROUTER_API_KEY` through a secret manager; the configuration names
required environment variables but never contains credentials.

The first completed `(3/4, 25%), Seed 22, integration` pilot achieved final
Directional Lift `0.2450793` and passed the technical artifact gate. Each case
writes the complete shared transcript to `coordination-board.jsonl`, one stable-ID
synchronized round per line, in addition to canonical events, private actor
histories, retained wave checkpoints, and a hashed `evidence-manifest.json`. A
single case does not establish reliability across seeds, objectives, or baselines.
See the Influence Profile for the complete mechanics and evidence contract.

## Verify the implementation

```bash
mise run check
```

The individual `test`, `lint`, and `typecheck` tasks are also available through `mise run`.

## Documentation

- [`CONTEXT.md`](CONTEXT.md): canonical project language.
- [`docs/schelling-reference-profile-v1.md`](docs/schelling-reference-profile-v1.md): frozen reference scientific and data contract.
- [`docs/schelling-influence-profile-v1.md`](docs/schelling-influence-profile-v1.md): frozen Influence Actor evaluation contract.
- [`docs/adr/0001-use-a-modernized-schelling-reference-profile.md`](docs/adr/0001-use-a-modernized-schelling-reference-profile.md)
- [`docs/adr/0002-map-behaviorally-distinct-tolerance-vacancy-cells.md`](docs/adr/0002-map-behaviorally-distinct-tolerance-vacancy-cells.md)
- [`docs/adr/0003-generate-the-reference-landscape-on-modal.md`](docs/adr/0003-generate-the-reference-landscape-on-modal.md)
- [`docs/adr/0004-select-three-final-satisfaction-test-spots.md`](docs/adr/0004-select-three-final-satisfaction-test-spots.md)
- [`docs/adr/0005-use-globally-informed-influence-actors.md`](docs/adr/0005-use-globally-informed-influence-actors.md)

## Method source

Thomas C. Schelling, "Dynamic Models of Segregation," *Journal of Mathematical Sociology* 1 (1971), 143-186.
