<p align="center">
  <img src="docs/assets/mam-bench-logo.webp" alt="MAM-Bench logo: orange and blue cells linked across a dark grid" width="200">
</p>

# MAM-Bench

**Compare language models by how well their agent groups steer multi-agent simulations.**

Each simulation creates 16 separate model controlled agents challenged to understand the simulation's rules,
cooperatively plan toward a goal, coordinate through a shared message board, and execute
their strategy. MAM-Bench measures the resulting change in ordinary-agent outcomes
against a paired, rule-based world. Run the same suite with different models to
compare their ability to steer a complex system toward a predefined goal.

[![CI](https://github.com/JoeHelbing/mam-bench/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/JoeHelbing/mam-bench/actions/workflows/ci.yml)
[![Latest release](https://img.shields.io/github/v/release/JoeHelbing/mam-bench?label=release)](https://github.com/JoeHelbing/mam-bench/releases/latest)
[![Apache-2.0 license](https://img.shields.io/github/license/JoeHelbing/mam-bench)](LICENSE)

[How it works](#how-it-works) | [Quickstart](#quickstart) |
[Custom suites](docs/custom-test-suite.md) | [Reference](#reference)

> [!IMPORTANT]
> **Research-stage benchmark.** The shipped conditions were selected from
> exploratory scripted interventions, not independent held-out qualification or
> language-model evaluations. The suite, interfaces, and scoring may change;
> this is not a standardized model ranking.

The [project overview](https://joehelbing.net/post/mam-bench) introduces the idea.

## How it works

Each case runs one selected model in a group of agents with a specific goal.
The two worlds start with the same seed and settings:

```text
                    one case + one seed
                   /                   \
    ordinary world                     controlled world
    all agents follow rules             selected agents use the model
                   \                   /
           compare ordinary-agent outcomes
                         v
                signed case score
```

The model-controlled agents can coordinate, but the score comes from their
*effect on the ordinary agents*, not from judging their conversations. The
shipped suite contains six cases in three matched-seed goal pairs:

| Simulation | Goal pair | Measured outcome |
| --- | --- | --- |
| Schelling (2 cases) | Integration / segregation | Final ordinary-agent neighbor homophily |
| Civil Violence, citizen control (2 cases) | Increase / decrease participation | Ordinary-citizen participation and revolution |
| Civil Violence, police control (2 cases) | Increase / decrease participation | The same ordinary-citizen outcomes |

Each case produces a signed score. The **Combined Benchmark Score** sums all
case scores (six in the shipped suite), including negative ones. One invocation
evaluates **one model**; compare totals only when the suite and scoring version
are identical.
Model responses and resulting trajectories can still vary between runs. See
[rules and scoring](docs/rules-and-scoring.md) for the exact formulas and world
mechanics.

## Quickstart

You need [uv](https://docs.astral.sh/uv/), Python 3.14, and access to a model
service that supports tool calls. For direct Anthropic API access, follow the
[Anthropic benchmark guide](docs/native-anthropic.md). The quickstart below uses
vLLM; an OpenRouter example follows it. Bring your own model and endpoint.

```bash
git clone https://github.com/JoeHelbing/mam-bench.git
cd mam-bench
uv python install 3.14.7
uv sync --frozen
```

To verify the checkout **without model calls or provider charges**:

```bash
uv run --no-sync python -m unittest discover -s tests -v
```

For an evaluation, create `.scratch/model.yaml` (run `mkdir -p .scratch` first).
The directory is Git-ignored. Replace the placeholders with a real model ID and
endpoint; do not put credentials in YAML:

```yaml
runtime: vllm
model: YOUR_MODEL_ID
base_url: https://YOUR-ENDPOINT.example/v1
api_key_env: MAM_MODEL_API_KEY
```

Use the actual served model ID so PydanticAI's `VLLMProvider` can select its
model-family profile. It handles reasoning, tool schemas, and system-message
formatting. Generic OpenAI-compatible and SGLang runtimes are not supported.
`settings.reasoning_effort` uses native unified thinking (`none` disables it).
Omitted sampling and reasoning settings use endpoint defaults, including for
history summaries. Action and summary output caps default to one quarter of the
discovered serving context, bounded by any reported provider output limit. For
a 262,144-token context, the default is 65,536 output tokens per request. Override
these with `settings.max_completion_tokens` and `settings.summary_completion_tokens`
in the model YAML. MAM-Bench does not add a model-request seed; case seeds still
control simulation randomness. Put desired request overrides under `settings`.
`tool_choice: required` retains native fallback when a profile cannot force
tools; `tool_choice: auto` opts out of forcing while keeping terminal actions.

For local vLLM builds supporting repeated-pattern detection, add this at the
top level of the model YAML for future runs:

```yaml
repetition_detection:
  min_pattern_size: 16
  max_pattern_size: 128
  min_count: 8
```

This stops consecutive repeated token patterns early; it does not change token
probabilities. The policy applies to agent and summary requests and is retained
in the run configuration. Omit it, or use `repetition_detection: null`, for a
baseline or a server without this vLLM extension. The adapter retains a native
repetition stop in provider details and uses bounded incomplete-output handling.
These thresholds are an initial policy, not a measured optimum.

Export `MAM_MODEL_API_KEY` in the environment, including for a local server
that ignores its value: this adapter still requires the variable. Keep secrets
out of the repo. Then run the shipped suite:

```bash
uv run --no-sync main.py --model .scratch/model.yaml --output results/default
```

**This makes real model requests and can incur substantial charges.** Check
endpoint support, data-use terms, and pricing before running all six cases.
Each invocation creates a new output directory and does not overwrite a
previous run. Results
include model conversations; handle them as sensitive data. See
[artifacts and failures](docs/artifacts-and-failures.md) for the saved records
and failure behavior.

Each CLI run also saves native OpenTelemetry traces, logs and metrics locally.
Prompts, replies, reasoning and tool arguments/results are captured by default.
Set `LOGFIRE_TOKEN` to send this telemetry to Logfire in the background. Complete
agent conversations also remain in local Harness archives.

You can replace the six cases with a YAML suite of your own:

```bash
uv run --no-sync main.py --model .scratch/model.yaml \
  --suite path/to/suite.yaml --output results/custom
```

The [custom suite guide](docs/custom-test-suite.md) lists every allowed field.
The shipped cases select `max_steps: 30`; custom cases can choose another
positive integer in either simulation. For OpenRouter, select
`runtime: openrouter` with `model` and `provider` in the model YAML, and set
`OPENROUTER_API_KEY` in the environment or `.env`:

```yaml
runtime: openrouter
model: meta/muse-spark-1.3-contributor
provider: meta
settings:
  reasoning_effort: medium
  tool_choice: auto
```

Meta's live Muse endpoint rejected `required` even though its metadata
advertised support, so this example selects `auto` explicitly. Before the first
case, setup discovers the selected provider's limits from OpenRouter endpoint
metadata or the served vLLM model's `/v1/models` entry. Missing context metadata
stops setup; context limits cannot be supplied in YAML. All selected OpenRouter
routes must report context; the smallest context/prompt ceiling applies when
routes differ. Larger served context remains available to the benchmark.

Setup rejects request overrides that violate known constraints. Unknown parameter
capabilities are left to the native provider and endpoint. Omitted generation
settings remain unset; setup does not invent output reservations or silently
reduce requested limits. Compaction, retained history, memory, concurrency, and
timeouts remain benchmark settings. Compaction uses the discovered context and
the configured trigger fraction (default 70%); token estimates and unknown
endpoint output defaults cannot guarantee that every request fits.
Direct library callers supply discovered context separately to `CaseRuntime`
and `AgentSessions`.

PydanticAI owns model-specific reasoning, tool safeguards, and prompt-cache
breakpoints. MAM-Bench enables its native OpenRouter instruction, message, and
tool caching controls; native profiles handle supported hints. Native telemetry
records reported cache usage; hints never guarantee a hit. The run saves requested
settings and discovered serving facts separately; unreported endpoint defaults
remain unknown. See [config.py](src/mam_bench/config.py) and
[model_setup.py](src/mam_bench/model_setup.py). If you use mise,
`mise install` and `mise run setup` prepare the pinned project tools and
dependencies.

## Reference

- [Anthropic benchmark guide](docs/native-anthropic.md) - configure a model and run the full suite.
- [Rules and scoring](docs/rules-and-scoring.md) - world mechanics, goals, formulas,
  and comparability limits.
- [Artifacts and failures](docs/artifacts-and-failures.md) - trajectories,
  messages, partial results, retries, and sensitive output.
- [Architecture and source walkthrough](docs/architecture.md) - where the
  runner, simulations, sessions, and persistence live.
- [Custom test suite YAML](docs/custom-test-suite.md) - fields and validation
  for explicit test cases.

See [CONTRIBUTING.md](CONTRIBUTING.md) for changes and bug reports,
[CHANGELOG.md](CHANGELOG.md) for releases, and [LICENSE](LICENSE) for Apache-2.0.
