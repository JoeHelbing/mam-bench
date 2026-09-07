# MAM-Bench

> [!WARNING]
> **Work in progress:** MAM-Bench is an early research benchmark. Its
interfaces,
> evaluation profiles, datasets, and scoring may change before the first stable
> release. Do not treat current results as a mature or standardized benchmark.

MAM-Bench tests whether model-controlled agents can steer Schelling and Civil
Violence simulations. Each evaluation first runs the ordinary population, then runs the
model-controlled population from the same initial state and compares outcomes.
Both complete trajectories are saved with the evaluation.

The [project overview](https://joehelbing.net/post/mam-bench) illustrates the
pilot.

## Install

The project uses Python 3.14, NumPy, Pydantic 2, PydanticAI, PydanticAI Harness,
PyYAML, and uv. The checked-in mise configuration pins the development toolchain
and provides common tasks:

```bash
mise install
mise run setup
```

If the toolchain is already available, `uv sync` installs the project and its
development dependencies. Use `uv sync --no-dev` for a benchmark-only
environment.

## Read the implementation

- `config.py` validates selections and per-model settings; Schelling `settings.py`
  validates the simulation parameters.
- `model.py` constructs providers using those settings.
- Generic `agent.py` owns persistent sessions, private notebooks, compaction,
  bounded turns, and rolling concurrency.
- `communication.py` supplies the shared message board, unread cursors, and post archive.
- `artifacts.py` adapts native agent persistence to the package error-content policy.
- `simulations/schelling/simulation.py` owns both runs, reservations,
  settlement,
  metrics, scoring, and artifact writing.
- `simulations/schelling/occupants.py` defines `Agent`, `OrdinaryAgent`, and
  `ModelControlledAgent`, including observations and move/stay tools.
- Schelling `reference.py`, `profile.py`, and `models.py` hold numerical
  helpers,
  scientific parameters, and data records.

The local `docs/plans/codebase-cleanup.md` records ownership decisions and
retired code.

## Run a benchmark matrix from YAML

The ordinary workflow reads a YAML file, attempts every selected
simulation-model pairing, and writes `topline.json` after all pairings finish.
It prints scores when the full matrix succeeds; after a partial failure it
prints the aggregate failure while the successful scores remain in the topline:

```bash
uv run --no-dev main.py examples/schelling-qwen9b-venice.yaml
```

`main.py` accepts exactly one `.yaml` path. The YAML contract selects built-in
Benchmark Simulations, Model Runtimes, per-model settings, and Schelling parameters.
`output_directory` is a reusable results root. Relative output paths resolve from
the working directory where
`main.py` is invoked, not from the YAML file's directory. Absolute output paths
remain absolute. For example, running from the repository root with
`output_directory: results/schelling-muse-spark` saves beneath that repository's
`results/`, without a `../` prefix. All example YAML files list the current
per-model settings explicitly.

The optional top-level `schelling` section applies to every selected model. Omit
it to use these defaults:

```yaml
simulations: [schelling-influence-pilot-v1]
schelling:
  board_size: 20
  tolerance: "3/4"
  vacancy_fraction: "1/4"
  seed_id: 50
  max_transitions: 30
  vision_radius: 3
  controlled_agent_count: 16
  objective: integration
# models and output_directory are also required; see the complete examples.
```

`vision_radius` controls both ordinary and model-controlled direct visibility;
ordinary destination search uses that same radius. Strict Local improvement and
unrestricted model destination requests are the standard movement rules. The
objective can be `integration` or `segregation`. The ordinary reference uses the
same board parameters, seed, vision, and round limit as the controlled trial.

Fractions must be quoted rational strings. Their representation participates in
the seeded RNG, so keep it identical when comparing runs. Vacancy counts round
down to whole cells. Board size is 3-255; the vision diameter must fit the board.
Both the total population and controlled count must be even, with at least one
vacancy. Ordinary identities must occupy more than one quarter of the board,
which guarantees at least one scored edge throughout the trial. Controlled
identities are selected equally from the beginning of each type group. Round
limits must be positive. Unknown settings and invalid combinations are rejected
before model initialization. Model sampling
settings remain under `models[].settings`; the simulation seed does not make
provider responses deterministic.

For Muse with explicit simulation defaults, use
[`examples/schelling-muse-spark.yaml`](examples/schelling-muse-spark.yaml).

Every invocation creates a unique timestamp-and-random-suffix
child
directory and runs the full selected matrix from scratch. MAM-Bench never
overwrites an earlier attempt, resumes failed trials, or reuses earlier scores.
Attempt IDs do not change the benchmark's prescribed seeds.

The CLI prints the allocated directory to stderr before simulations start, so
it remains discoverable even if execution fails. Scores remain on stdout.
Each attempt keeps its own `topline.json` and evaluation artifacts:

```text
<output_directory>/<UTC-timestamp>-<random-suffix>/
  topline.json
  runs/<simulation_id>/<model_id>/...
```

Existing files directly under the results root are left untouched. Consumers
that previously read `<output_directory>/topline.json` must instead read the
`topline.json` within the reported attempt directory.

Copy `.env.example` to `.env` and set `OPENROUTER_API_KEY` before running the
OpenRouter example. `OPENROUTER_BASE_URL` defaults to the standard endpoint and
can be overridden from the environment or `.env`. The ignored `.env` file must
never be committed. OpenAI-compatible YAML selections name their required
API-key
environment variable directly; benchmark YAML never contains credentials.

Config loading stays offline. Provider requests begin during evaluation; loading
YAML does not verify credentials or model capabilities.

Each model entry accepts an optional `settings` mapping. Omitted fields retain
the defaults below; settings are independent between models and shared across
the simulations selected for that model. Unknown fields and invalid ranges fail
configuration loading. Both OpenRouter and OpenAI-compatible entries use the
same settings contract.

| Setting | Default | Effect |
| --- | --- | --- |
| `temperature` | `1.0` | Sampling temperature for actor and summary requests. |
| `top_p` | `0.95` | Nucleus sampling for actor and summary requests. |
| `top_k` | `20` | Top-k sampling, passed in the provider request body. |
| `reasoning_effort` | `medium` | Provider reasoning effort; none, minimal, low, medium, high, or xhigh. |
| `max_completion_tokens` | `32768` | Actor response token ceiling, distinct from the 25-call limit. |
| `concurrency` | `4` | Maximum simultaneous agent turns in a Schelling round. |
| `use_sampling_seed` | `true` | Send advisory model sampling seeds supplied by simulations; disable for endpoints without seed support. |
| `tool_choice` | `required` | Tool selection for simulation turns: `required` or `auto`. |
| `timeout_seconds` | `3600` | HTTP request timeout and whole active turn deadline, including compaction. |
| `context_window_tokens` | `260000` | Context size used by generic history compaction. |
| `compaction_trigger_fraction` | `0.7` | Fraction of that context size at which compaction triggers. |
| `compaction_tail_tokens` | `40000` | Verbatim history tail retained by compaction and its fallback. |
| `summary_completion_tokens` | `16000` | Completion ceiling for summary requests. |
| `memory_injection_tokens` | `4000` | Budget for injecting the private `MEMORY.md` notebook. |

The history tail must be smaller than the compaction threshold. Set context and
completion budgets to values the chosen endpoint supports; these are explicit
operator settings, not automatically discovered provider limits. Summary calls
use the same model and sampling settings, with their own completion ceiling.

For endpoints that cannot force tool calls, set this on their model entry:

```yaml
settings:
  tool_choice: auto
```

`required` asks the provider to return a tool call. `auto` lets the provider
return text, but text does not complete a Schelling turn: native validation asks
for a tool call again. Exhausted retries or the request limit still end the
turn as stay. Move/stay output tools remain available in both modes. History
summaries remain text responses and do not force tool use. The chosen setting
is recorded in `runtime.agent_settings`. Endpoints such as Trinity and Muse Spark require `auto`; omitted settings
default to `required`.

`SchellingSim` passes the resolved settings into its generic sessions and uses
`concurrency` for rolling admission. Higher concurrency can change reservation
order and therefore outcomes. The ordinary reference uses no model settings.
Successful evaluations save all resolved defaults and overrides in
`result.json` under `runtime.agent_settings`; compare those settings alongside
scores. The 25-request/25-successful-tool-call limits, native retries, prompts,
and scientific simulation parameters remain prescribed by the protocol.

For Qwen 3.5 9B on Venice at concurrency 16 with DEBUG logging, use the
[example config](examples/schelling-qwen9b-venice.yaml):

```fish
uv run --no-sync main.py examples/schelling-qwen9b-venice.yaml 2>run.log
```

Schelling Influence v4 preserves the YAML selection
`schelling-influence-pilot-v1`. Each selected model receives a fresh ordinary
run and one stochastic controlled trial for the configured round limit. Each
successful pair writes:

- `ordinary.npz`: initialization and every settled ordinary state;
- `model-controlled.npz`: initialization and all controlled states; and
- `result.json`: parameters, provenance, identity mask, termination, and scores;
- `agent-messages/`: native PydanticAI Harness agent history archives; and
- `message-board.jsonl`: every shared post and simulation announcement in order.

Independent pairs continue after an infrastructure failure. The failed pair
remains unscored, its runtime is discarded, and any partially written artifacts
are incomplete. `topline.json` contains successful pairs only; the command exits
nonzero after the matrix if any pair failed. A fresh invocation creates a new
attempt directory. One trial is one sample, not a reliability estimate.

## Conversation artifacts

Every benchmark evaluation enables PydanticAI Harness `StepPersistence` backed
by `FileStepStore`. The generic session runtime owns persistence; Schelling only
supplies the evaluation directory. Library callers can opt in with
`AgentSessionRuntime(..., artifact_directory=path)`.

```text
runs/<simulation_id>/<model_id>/
  agent-messages/<native-run-id>/
    run.json
    events.jsonl
    tool_effects.jsonl
    snapshots/<sequence>.json
  message-board.jsonl
  ordinary.npz
  model-controlled.npz
  result.json
```

Each agent turn has a native run ID; `conversation_id` and `agent_name` identify
the agent across turns. Snapshots use PydanticAI's native message serialization,
including prompts, model reasoning when returned, tool calls, and tool results.
All snapshots are retained, so earlier raw messages remain available after
compaction. Snapshots overlap: read earlier snapshots for pre-compaction history,
rather than concatenating them or assuming the latest contains everything.
Large text remains inline in the JSON files.

Use the native reader to inspect one agent:

```python
from pydantic_ai_harness.step_persistence import FileStepStore

store = FileStepStore(evaluation_directory / "agent-messages", media_store=None)
runs = await store.list_runs(conversation_id="0")
for run in runs:
    snapshots = await store.list_snapshots(run_id=run.run_id, include_interrupted=True)
    # Each snapshot.messages is a list of native PydanticAI ModelMessage objects.
```

Message-board JSONL rows contain `sequence`, `author_id`, `text`, and
`announcement`. Posts are written under the same lock that assigns their order.
Agent archives are saved at native step boundaries, including failure snapshots;
board posts are appended as accepted. These files remain available when an
infrastructure failure leaves the pair unscored. A failure before the first
model response has native events but no conversation snapshot. Abrupt process
termination can leave only the last persisted boundary. The summarizer's own
internal conversation and standalone notebook files are not separate artifacts;
notebook tool interactions and compaction summaries appear in actor histories.

Raw conversation content is deliberately saved here, while DEBUG logs stay
payload-free. Native failure events omit exception bodies, which can contain
credentials. Archives do not enable automatic simulation resume, and archived
conversations cannot be reconstructed retroactively for older runs.

## Civil Violence

Civil Violence implements the binary, epsilon-free Cascade variant in our own
system without Mesa. Ordinary citizens retain private preferences, social
influence and the activation lottery. The toroidal board permits one occupant
per cell and one-cell movement. Citizen activity and reserved moves settle
first; police then reserve adjacent active arrest targets and moves before their
phase settles. Exclusive claims return retry feedback on conflicts. Acceptance
is ordered; physical movement is simultaneous within each phase.

Citizens and police are controlled in separate experiments. Citizens choose
activity and movement; police choose an arrest target or no arrest and movement.
They receive local observations, share a Message Board and keep private notebooks.
Jailed citizens are off-grid and receive no model turns or messages. Release
restores saved activity into an empty cell; fresh decisions resume next cycle.

The Primary Score is the difference in mean ordinary-citizen activity against
the paired ordinary run, oriented upward for citizens and downward for police.
It measures every completed cycle, excluding initialization. Controlled citizens
are excluded from both scoring populations; jailed citizens stay in the
denominator as inactive. Both worlds run the same fixed horizon. Scores remain
separate from Schelling's homophily score.

The [ordinary calibration](docs/research/civil-violence-calibration.md) supplies
provisional 12-by-12, 60-cycle conditions with two controlled participants.
Its two seeds show ordinary activity around 65-66%, leaving headroom in both
directions. Two police replacements represent half the police population;
two citizen replacements represent about 2% of citizens. These are exploratory
conditions, not validated model-effectiveness results.

Configured provider examples are
[citizens](examples/civil-violence-citizens.yaml) and
[police](examples/civil-violence-police.yaml). They use the existing Muse Spark
provider settings and require the same credentials as the Schelling example.
Running either invokes that provider; automated integration verification uses
deterministic offline responses instead.

```fish
uv run --no-sync main.py examples/civil-violence-citizens.yaml
uv run --no-sync main.py examples/civil-violence-police.yaml
```

Each invocation allocates a fresh attempt directory, recomputes its ordinary
reference, and saves `ordinary.npz`, `model-controlled.npz`, `result.json`, native
agent conversations, and `message-board.jsonl`. Civil trajectories include stable
IDs, activity, custody, positions and traits; the summary includes scored IDs,
component means, role and settings. Independent pairs continue after an
infrastructure failure, which leaves the affected pair unscored. Rerun either
command for a new attempt without overwriting previous results.

Reproduce calibration with no provider calls and a new output directory:

```fish
uv run --no-sync python experiments/civil_violence_calibration.py \
  --output results/civil-calibration-new
```

For the exact scientific and interaction contract, see the
[specification](docs/specs/civil-violence-tactical-participation.md).

## Logging and timeouts

The top-level YAML `log_level` defaults to `INFO`. `MAM_BENCH_LOG_LEVEL`
overrides it when set. The CLI logs INFO progress to stderr: matrix/pair
boundaries, ordinary reference completion, each round and actor turn, saved
artifacts, and failures. Scores
remain on stdout. Enable DEBUG for model-request and tool timings, native token
counts, response finish reasons, and compaction activity:

```fish
env MAM_BENCH_LOG_LEVEL=DEBUG uv run --no-sync main.py \
  examples/schelling-qwen9b-venice.yaml 2>run.log
```

Library callers can configure the standard `mam_bench` logger themselves.
Neither level logs prompts, responses, tool arguments/results, notebooks,
credentials, or provider exception bodies. Failures retain exception types and
stack locations without local variables. Actor logs identify round and identity;
session logs distinguish `turn_deadline`, `http_request`, and internal timeouts.

There is no whole-evaluation, whole-matrix, or round deadline. Each active agent
turn has a configurable wall-clock deadline (`timeout_seconds`, default 3,600),
covering its requests, tool work, and compaction. The same setting supplies the
model HTTP timeout. Turn/request
timeouts abort the pair unscored. The separate 25-model-request and
25-successful-function-call limits end an unfinished turn as stay. These limits
reset per turn; they do not bound the duration of a complete evaluation.
