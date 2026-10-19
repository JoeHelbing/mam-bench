# Native OpenAI

Select `models/openai.yaml` to use PydanticAI's `OpenAIResponsesModel` directly.
The example uses `gpt-4.1-mini` for inexpensive setup checks. Change
`model` to evaluate another supported model.

Keep `OPENAI_API_KEY` in a Git-ignored env file. Load it explicitly;
the adapter uses the native provider's standard environment configuration.
Run these Fish-compatible commands from this branch's checkout, replacing the
env-file path with your existing file:

```fish
uv sync --frozen
uv run --frozen --env-file /path/to/mam-bench/.env python scripts/probe_cache.py --model models/openai.yaml --output .scratch/cache-probe
uv run --frozen --env-file /path/to/mam-bench/.env python main.py --model models/openai.yaml --output results
```

The probe performs a short control request, a cold long-prefix request, a warm
request from a separate agent, and a continuation of that warm agent's history.
Each request must return the requested structured tool output. It records native
input/output and cache read/write token counts in `cache-report.json`, retains
session artifacts, and exits nonzero unless both warm phases report cache reads.
Its synthetic long instructions exceed cache minimums; real benchmark prompts
may be too short to cache. These are small setup checks, not benchmark scores.

OpenAI caches eligible repeated prefixes automatically. A stable per-model cache key
helps route related benchmark requests. Responses are not stored server-side by
this adapter; each Agent Session retains its own local history.

Context comes from the native PydanticAI model profile (including its bundled
model metadata), not live serving-limit discovery. Unknown context fails setup.
Omitted output caps retain native provider defaults; the example explicitly
caps both actions and summaries at 1,024 tokens. The endpoint validates its
model-specific output ceiling and request capabilities.

No keys or personal env-file paths are saved in model YAML or run metadata.
Do not publish env files or session artifacts containing private prompts.

Sources: [PydanticAI OpenAI integration](https://pydantic.dev/docs/ai/models/openai/),
[provider caching documentation](https://developers.openai.com/api/docs/guides/prompt-caching).

## Verification

On October 19, 2026, PydanticAI 2.38.0 completed the streamed structured-tool
probe with this branch's example model. One live run reported:

| Phase | Input tokens | Cache reads | Cache writes |
| --- | ---: | ---: | ---: |
| short | 608 | 0 | 0 |
| cold | 8408 | 0 | 0 |
| warm | 8408 | 8320 | 0 |
| growing | 8444 | 8320 | 0 |

Counts can vary between runs. Both warm requests showed prefix reuse, while the
short control remained uncached. All 134 offline tests, Ruff, formatting, and
strict Pyright checks passed. A live one-step smoke suite also completed Schelling
and both Civil Violence roles using their actual benchmark tools and valid
controlled-agent populations. These shortened cases do not form a benchmark score.
