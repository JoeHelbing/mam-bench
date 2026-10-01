# MAM-Bench

> [!WARNING]
> **Work in progress:** MAM-Bench is an early research benchmark. Its
interfaces,
> evaluation profiles, datasets, and scoring may change before the first stable
> release. Do not treat current results as a mature or standardized benchmark.

MAM-Bench measures how model-controlled agents change a simulation relative to
ordinary agents initialized with the same seed and parameters. One invocation
evaluates one explicit language model against an ordered suite of test cases.
Every completed case contributes a signed score; the final score is their sum.

The [project overview](https://joehelbing.net/post/mam-bench) illustrates the
pilot.

## Run a custom suite

Python 3.14 and the dependencies declared in `pyproject.toml` are required.
With the project environment already prepared:

```fish
uv run --no-sync main.py --model examples/model-muse-spark.yaml \
  --suite examples/development-suite.yaml --output results/development
```

This command makes real provider requests and can incur charges. Automated
tests use scripted responses instead. OpenRouter reads `OPENROUTER_API_KEY`
and optional `OPENROUTER_BASE_URL` from the environment or `.env`.
An OpenAI-compatible model file instead supplies `runtime: openai-compatible`,
`model`, `base_url`, and `api_key_env`; the last field names the environment
variable holding its credential. YAML must not contain credentials.

The required `--model` file selects exactly one model. Model files also accept
`settings` for sampling, request/turn timeouts, concurrency, and compaction;
[config.py](src/mam_bench/config.py) defines those validated settings.
The Muse and Qwen files preserve the existing provider examples; they are not
recommendations or completed evaluations.

`--suite` selects a YAML mapping containing `cases`. Every simulation parameter
must be explicit, including objective and seed. The entire case list replaces
the shipped list; cases run sequentially in listed order, including repetitions.
Missing fields identify the case and field in a Pydantic validation error before
any provider is constructed or output is created. There is no matrix expansion,
parameter inheritance, or default model.

The package's [default-suite.yaml](src/mam_bench/default-suite.yaml) deliberately
has no cases until suite selection is completed. Omitting `--suite` therefore
reports a validation error and the deferral. See
[development-suite.yaml](examples/development-suite.yaml) for complete, explicitly
labelled examples of both Schelling goals and all four Civil Violence role/goal
combinations.

The CLI prints each completed case's effective parameters and signed score, then
the Combined Benchmark Score. On failure it identifies the failed case, retains
completed rows, exits nonzero, and withholds the total. `--output` is relative to
the working directory and defaults to `results`; each invocation creates a fresh
timestamp/UUID child and never overwrites a previous attempt.

## Rules and scores

Both worlds start from the same seed and use the same staged turn mechanism:
selected replacement identities decide first, then the remaining ordinary agents.
Model decisions can overlap up to the configured concurrency limit, but claims
are accepted in seeded priority order, not response order. Ordinary decisions
use that same priority and see earlier reservations. Within each phase, physical
actions settle only after both groups finish. Invalid claims retain their turn for bounded
retries or fallback; a slow earlier turn can delay later claims.

Turn priorities and action randomness have separate, stable addresses: case seed,
step, role phase, identity, and purpose. Replacing an ordinary decision leaves its
random slot unused instead of consuming a proposal in a different world state.
Jail, release, or early termination in one world cannot shift another agent's
random stream. Different available actions can still map the same randomness to
different choices. Model outputs and message timing are not made deterministic.

This changes the ordinary reference dynamics. Historical calibration archives
remain historical evidence, not interchangeable baselines for this scheduler;
recompute calibration before selecting defaults or evaluation seeds.

Schelling preserves strict local improvement for ordinary agents: satisfaction
uses occupied radius-one neighbors and a floating-point tolerance; unhappy
agents choose the nearest predicted improvement within their vision, with random
ties. Destination predictions use only cells visible from the current origin,
excluding the origin itself. Unknown cells are excluded; zero known occupied
neighbors predicts quality one. Model agents may request any starting vacancy.
Selected identities reserve first in both worlds, followed by the remaining
ordinary identities; both groups use seeded priorities and all moves settle
together. Ordinary execution stops at equilibrium, blockage, or the horizon;
controlled execution runs to the configured horizon.

Set `tolerance` and `vacancy_fraction` as numbers between zero and one (for example,
`0.75` and `0.25`). Vacancy counts round `board_size ** 2 * vacancy_fraction` to the
nearest integer, with ties to even; population validation still requires equal
type counts. Satisfaction compares the same-type neighbor share directly with
the tolerance, without an epsilon.

Schelling scores final ordinary-to-ordinary edge homophily, excluding the selected
controlled identities in both worlds:

- Integration: `400 * (reference_homophily - controlled_homophily)`.
- Segregation: `400 * (controlled_homophily - reference_homophily)`.

Fractions are from zero to one. Scores are signed and unclamped: an improvement
from 100% to 40% homophily earns 240 integration points.

Civil Violence retains binary, epsilon-free Cascade activation, one-cell movement
on a single-occupancy torus, citizen settlement before police observation, atomic
movement/arrest reservations, off-grid custody, and cached-activity release.
Every case replaces exactly 16 existing citizens or police. Either role can seek
increased or decreased participation.

Participation is the active-or-jailed fraction of scored ordinary citizens.
Selected controlled identities are excluded from both worlds. Each world stops
independently at its first completed step with at least 95% participation, or
after step 30; revolution is checked on step 30 before the horizon.

```text
direction = +1 for increase, -1 for decrease
score = direction * (100 * (controlled_participation - reference_participation)
                   + 100 * (controlled_revolution - reference_revolution))
```

Revolution indicators are zero or one. Time to revolution is saved without a
timing reward. Arrest alone does not reduce participation. A jailed model citizen
retains its identity, session, memory, and shared-board access, receives an
explicit jailed observation, and ends its communication-only turn with `defer`.
Physical actions retry; ordinary jailed citizens defer automatically.

The Combined Benchmark Score sums every case score, including negative values,
without averaging or additional weights. Compare totals only for the same suite
and scoring version.

## Artifacts and failures

```text
<output>/<timestamp>-<uuid>/
  config.json
  completed-cases.jsonl
  benchmark.json                 # complete runs only; calculated total_score
  failure.json                   # failed runs, when storage permits
  cases/001/
    config.json
    ordinary.jsonl
    controlled.jsonl
    ordinary-outcome.json         # retained even if the controlled world fails
    controlled-outcome.json
    result.json                  # one simulation-owned result, including score
    agent-messages/              # native Pydantic AI Harness archives
    message-board.jsonl
    turns.jsonl                  # exhausted-turn reason and fallback, when needed
    failure.json                 # failed case, when storage permits
```

Each trajectory starts with initialization, then appends every completed step
immediately after settlement. Civil Violence writes one state per full cycle,
not separate phase snapshots. States contain stable identities, locations,
simulation-specific state, and intermediate measurements. Case results contain
the exact settings, scored/controlled identities, both outcomes and termination,
and calculated scores. Those same result objects feed saved and displayed totals.

Native Harness `StepPersistence` retains messages, model reasoning when returned,
tool calls/results, and interrupted runs. Shared-board posts persist as accepted.
Native snapshots overlap; inspect them rather than concatenating histories.
Compaction does not erase earlier archived messages. Private notebook operations
are in tool histories; standalone notebook files and summarizer conversations
are not separate artifacts.

Invalid actions receive bounded retry feedback. Exhausted request, tool, output,
or retry budgets select a simulation fallback and retain a diagnostic reason.
They do not cause an infrastructure failure or an extra scoring penalty.
Provider/network/timeouts, persistence failures, and simulation defects stop
the invocation and cancel outstanding turns. Completed records remain available.
Failure records are best effort; failure to write one is logged without masking
the original error. There is no resume or selective rerun support.

INFO logging reports case progress; DEBUG adds session/request/tool timing and
usage. Set `MAM_BENCH_LOG_LEVEL` to override the YAML `log_level`. Logs and
native failure events omit sensitive exception bodies. Conversation artifacts
intentionally retain model content.

## Source walkthrough

For either simulation, start at [main.py](main.py): parse an explicit model file
and suite, then [config.py](src/mam_bench/config.py) validates the complete nested
configuration. [BenchmarkRunner](src/mam_bench/runner.py) creates one provider,
one fresh `CaseRuntime` and writer per case, constructs the selected simulation,
and awaits `evaluate(runtime)`. It never steps a world.

For Schelling, [simulation.py](src/mam_bench/simulations/schelling/simulation.py)
constructs two distinct worlds through the same constructor, executes its single
step loop for each, and writes initial/completed states. Its
[board](src/mam_bench/simulations/schelling/board.py) owns spatial state and
measurements. Persistent [agents](src/mam_bench/simulations/schelling/agents.py)
choose actions; the simulation validates/reserves and settles them. Its
[result](src/mam_bench/simulations/schelling/results.py) calculates signed lift.

For Civil Violence,
[simulation.py](src/mam_bench/simulations/civil_violence/simulation.py) follows the
same evaluate/run/step path. One citizen-then-police sequence serves both worlds.
Persistent [agents](src/mam_bench/simulations/civil_violence/agents.py) select
role-specific proposals or model actions; the simulation applies custody and
atomic phase settlement before measuring participation and checking revolution.
Its [result](src/mam_bench/simulations/civil_violence/results.py) calculates the
participation and revolution components through the same `score` property.

Both use [sessions.py](src/mam_bench/sessions.py) for individual model turns,
private history/memory, and communication. Instructions carry identity, goal,
and standing rules; user inputs contain changing JSON observations. Each
simulation owns legality and settlement. [scheduling.py](src/mam_bench/scheduling.py)
shares cohort ordering, concurrent admission, claim gates, failure cleanup, and
addressed random streams. [artifacts.py](src/mam_bench/artifacts.py) writes supplied
records and preserves native archives. The runner retains case
results, and `BenchmarkResult.total_score` sums their calculated scores.

There are no reset/reinitialization paths, alternate constructors, old YAML
adapters, simulation-model matrices, base-simulation framework, or reference
replay dependency. The obsolete 60-step/two-agent calibration selector and tests for
retired interfaces are removed. Historical result files are untouched.

## Verify without model calls

```fish
uv run --no-sync python -m unittest discover -s tests -v
ruff check src tests main.py
ruff format --check src tests main.py
uv run --no-sync pyright --pythonpath .venv/bin/python
```

Checks cover upfront validation, paired worlds, an independent Schelling movement
oracle, score replay, signed formulas, reservations and settlement, persistent
sessions and jailed communication, matched ordinary-policy replacements, isolated
random slots, stopping, incremental artifacts, fallback diagnostics, cancellation,
and failure redaction. They verify implementation behavior, not scientific
calibration or real-model effectiveness.
