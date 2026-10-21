# Run a benchmark with OpenAI

Use the OpenAI Responses API to evaluate one selected model against the full
six-case benchmark suite.

## Configure the model

Put `OPENAI_API_KEY`, `MODAL_TOKEN_ID`, and `MODAL_TOKEN_SECRET` in your Git-ignored
`.env` file. Modal hosts each controlled agent's private workspace. Select the model in
`models/openai.yaml`:

```yaml
runtime: openai
model: gpt-4.1-mini
```

Replace `model` with the exact model ID you want to evaluate. The example uses
provider defaults for generation and the standard benchmark session settings.

## Run the full suite

Run these Bash commands from the repository checkout:

```bash
uv sync --frozen
uv run --no-sync --env-file .env python main.py \
  --model models/openai.yaml --output results/openai
```

This runs all six shipped cases with real API calls: Schelling integration and
segregation, plus increasing and decreasing Civil Violence participation under
citizen and police control. A completed run prints the Combined Benchmark Score.

Each invocation creates a new directory under `results/openai`. It retains
the requested and effective model configuration, per-case results, trajectories,
agent conversations, and the final `benchmark.json`. Failed runs retain partial
artifacts without a final score.

## Choose evaluation settings

Optional overrides belong under `settings`, using the existing model YAML
format. Use `reasoning_effort` to choose the model's thinking mode,
`max_completion_tokens` and `summary_completion_tokens` for explicit generation
limits, and `concurrency` or `timeout_seconds` for session execution controls.
Choose limits appropriate for the selected model and intended evaluation.
The example leaves these overrides unset. Omitted output settings use native provider
defaults. PydanticAI supplies model context metadata; unknown context stops setup.

To replace the shipped cases, add `--suite path/to/suite.yaml`. See the
[custom suite guide](custom-test-suite.md), [scoring rules](rules-and-scoring.md),
and [artifact guide](artifacts-and-failures.md).
