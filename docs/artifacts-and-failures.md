# Artifacts and failures

```text
<output>/<timestamp>-<uuid>/
  config.json                   # original requested model and suite
  effective-model.json          # requested settings and discovered serving facts
  completed-cases.jsonl
  benchmark.json                 # complete runs only; calculated total_score
  failure.json                   # failed runs, when storage permits
  telemetry/
    traces.jsonl                 # native agent/model/tool spans, including content
    logs.jsonl                   # correlated package logs
    metrics.jsonl                # native GenAI metrics, delta temporality
  cases/001/
    config.json
    ordinary.jsonl
    null.jsonl                   # null-policy world: every selected turn falls back
    controlled.jsonl
    ordinary-outcome.json         # retained even if the controlled world fails
    null-outcome.json
    controlled-outcome.json
    result.json                  # one simulation-owned result, including scores
    turns.jsonl                  # exhausted-turn reason and fallback, when needed
    failure.json                 # failed case, when storage permits
```

Each trajectory starts with initialization, then appends every completed step
immediately after settlement. Civil Violence writes one state per full cycle,
not separate phase snapshots. States contain stable identities, locations,
simulation-specific state, and intermediate measurements. Case results contain
the exact settings, scored/controlled identities, all three outcomes (reference,
null policy, controlled) and termination, and the raw, null-policy, and adjusted
scores. Those same result objects feed saved and displayed totals.
The effective setup records discovered context/capabilities separately from
requested settings. It does not claim to know unreported endpoint generation
defaults. Native CLI telemetry records request/token/cache
usage, including summary calls and requests completed before a later failure.
An absent or zero cache-token count does not establish a measured cache miss.
Native model spans retain per-response usage. Direct library users can
enable telemetry with `collect_telemetry`; simulations do not write a separate
usage artifact.

`Agent.instrument_all()` is the single source of saved agent data: prompts,
replies, returned reasoning, tool arguments/results, and usage. Model-controlled
simulation turns use native streaming; completed responses are assembled into
model spans. There are no separate Harness message snapshots, board archives,
stream-event files, or raw vLLM chunks. Instrumentation does not record each
chunk as it arrives, and interrupted streams may lack partial response content.
Compaction does not erase earlier exported spans; nested summary requests are
instrumented too. Session histories and board messages remain in memory for
execution. Coder commands and file operations appear as native tool spans.
Package logs record session and sandbox IDs for cleanup diagnostics. Workspace
files are not copied into local artifacts and are lost when the sandbox is
terminated; notebook memory tools have been removed.

Every controlled identity owns a separate Modal sandbox for one case, retained
across turns. The custom image is defined in `sandboxes.py`. Project Modal
credentials come from the host environment or `.env` and are not forwarded into
the sandbox. `observation.json` is refreshed before each model turn, and
`observe.py` lets Python code read this same local snapshot. An agent can also
call the ordinary `observe()` tool. Neither grants additional simulation visibility.
Sandbox preparation is outside the model-turn deadline. Cases explicitly
terminate all owned sandboxes on completion, error, or cancellation; the default
12-hour sandbox lifetime is a backstop. Cleanup failures stop a successful case,
or are logged while preserving an existing failure.

Invalid actions receive bounded retry feedback. Exhausted request, tool, output,
or retry budgets select a simulation fallback and retain a diagnostic reason.
Transient API errors, including streamed provider errors, connection failures,
request timeouts, HTTP 408/409/429, and server errors, retry within the same turn.
Failed API requests count toward the request allowance. API errors and failed
tool calls share a cumulative three-failure limit; successful calls do not reset
it. At the limit, `call_error_limit` ends only that turn with the simulation's
normal fallback. Each new turn starts with a fresh failure budget. Accepted tool
effects remain; incomplete streamed tool calls receive an explicit not-executed
reply before retrying. Failed calls log their exception type and status code.
Tool failures include argument validation, rejected actions, unknown tools, and
reported failures across actions, messaging, and workspace tools. Tool batches settle
before the limit is checked, preserving matching replies and accepted effects.
The limit prevents the next request and does not override an accepted terminal action.
They do not cause an infrastructure failure or an extra scoring penalty.
Permanent HTTP errors (including invalid credentials or request parameters),
turn deadlines, persistence failures, and simulation defects still stop the
invocation and cancel outstanding turns. Completed records remain available.
Native compaction runs before requests; a server-rejected context overflow
follows normal provider-failure handling.
Failure records are best effort; failure to write one is logged without masking
the original error. Both run and case `failure.json` retain the error message
and complete traceback, including chained causes. The raised `BenchmarkRunFailure`
also retains the original exception as its cause. Configuration errors print the
original file/YAML/validation details, including invalid values.
There is no resume or selective rerun support.

INFO logging reports case progress; DEBUG adds session timing and usage.
Native OpenTelemetry spans supply model request and tool timing, including
nested summarizer requests. Set `MAM_BENCH_LOG_LEVEL` to override the YAML
`log_level`. Logs retain complete exception chains and provider error bodies,
including errors received inside successful HTTP streaming responses. Native
spans retain their original error details.

The CLI collects all three telemetry signals locally using standard SDK JSON
exporters. Native instrumentation includes prompts, replies, reasoning, and tool
arguments/results by default, both locally and when sending to Logfire.
Request/tool spans export when they finish; metrics export every 30 seconds and
at orderly shutdown. Read each JSONL record independently;
the console-export JSON is a diagnostic format, not an OTLP protobuf dump.
Native `gen_ai.agent.call.id` identifies an agent call and
`gen_ai.conversation.id` identifies the session. Use these attributes to group
related request and tool spans.

The CLI reads `LOGFIRE_TOKEN` from the environment or the working directory's
`.env` file to enable background OTLP HTTP delivery to Logfire. Environment
values override `.env`, including an empty token to disable cloud delivery.
`LOGFIRE_BASE_URL` follows the same precedence and defaults to
`https://logfire-us.pydantic.dev`; use the base URL for the region of the token.
Keep `.env` private and Git-ignored; put no credentials in YAML or Git.
The startup log records `telemetry.export destination=logfire` or `local` and
the run ID. This reports configuration; exporter failures are logged separately.
Library collection remains local unless `send_to_logfire=True` is requested.
Local collection works without a Logfire account or token. Remote delivery
errors do not fail agent turns; HTTP exports have a three-second timeout.
An abrupt process kill can lose unfinished spans and buffered cloud records.

Each signal already carries the generated `mam_bench.run_id` resource attribute.
Dashboards should select that attribute through a query-backed Logfire variable,
so a new run requires no query edits. For a batch of runs, its launcher can set
one shared `mam_bench.campaign` label in the standard `OTEL_RESOURCE_ATTRIBUTES`
environment variable. The native SDK includes it on traces, logs and metrics;
the benchmark needs no campaign registry, path discovery or new YAML fields.
Use explicit selectors when campaigns overlap. Server/GPU measurements belong
to their host or serving process; a separately observed benchmark status must
not label those measurements as belonging to an old campaign.

For example, a Logfire query-backed list variable named `run_id` can use:

```sql
SELECT otel_resource_attributes->>'mam_bench.run_id' AS run_id
FROM records
WHERE service_name = 'mam-bench'
  AND otel_resource_attributes->>'mam_bench.run_id' IS NOT NULL
GROUP BY 1
ORDER BY max(start_timestamp) DESC
```

Panels filter with `otel_resource_attributes->>'mam_bench.run_id' = $run_id`.
Use the same pattern for an optional campaign label. The dropdown is populated
from recorded data, rather than a maintained list of dated campaign names.
See [Logfire dashboard variables](https://pydantic.dev/docs/logfire/observe/dashboards/#variables)
and [native OTel resource labels](https://opentelemetry.io/docs/languages/sdk-configuration/general/#otel_resource_attributes).

Native telemetry retains conversation content, exception messages, stacktraces
and span status descriptions. Model request parameters and binary content are
not captured. The same records go to Logfire when export is explicitly enabled.

Library imports do not initialize telemetry. `collect_telemetry(directory)`
owns one invocation's process-wide agent instrumentation and scoped package
logging; run independent collection contexts in separate processes. Scientific
results and trajectories retain their existing atomic/checkpoint writes.
