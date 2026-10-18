# Artifacts and failures

```text
<output>/<timestamp>-<uuid>/
  config.json                   # original requested model and suite
  effective-model.json          # requested settings and discovered serving facts
  completed-cases.jsonl
  benchmark.json                 # complete runs only; calculated total_score
  failure.json                   # failed runs, when storage permits
  telemetry/
    traces.jsonl                 # native OpenTelemetry records, content disabled
    logs.jsonl                   # correlated package logs
    metrics.jsonl                # native GenAI metrics, delta temporality
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
The effective setup records discovered context/capabilities separately from
requested settings. It does not claim to know unreported endpoint generation
defaults. Native CLI telemetry records request/token/cache
usage, including summary calls and requests completed before a later failure.
An absent or zero cache-token count does not establish a measured cache miss.
Native message snapshots retain per-response usage. Direct library users can
enable telemetry with `collect_telemetry`; simulations do not write a separate
usage artifact.

Native Harness `StepPersistence` retains messages, model reasoning when returned,
tool calls/results, and interrupted runs. Shared-board posts persist as accepted.
Native snapshots overlap; inspect them rather than concatenating histories.
Compaction does not erase earlier archived messages. Private notebook operations
are in tool histories; standalone notebook files and summarizer conversations
are not separate artifacts.

Invalid actions receive bounded retry feedback. Exhausted request, tool, output,
or retry budgets select a simulation fallback and retain a diagnostic reason.
Three failed tool calls in one Agent Turn stop it with `tool_error_limit` and
select the same simulation fallback. Argument validation failures, rejected
actions, unknown tools, and reported tool failures count across all tools;
successful calls do not reset the count. Each new turn has a fresh budget.
Tool batches settle before the limit is checked, preserving their results and
accepted effects. The limit prevents the next model request; it does not
interrupt a response still being generated or override an accepted terminal action.
They do not cause an infrastructure failure or an extra scoring penalty.
Provider/network/timeouts, persistence failures, and simulation defects stop
the invocation and cancel outstanding turns. Completed records remain available.
Native compaction runs before requests; a server-rejected context overflow
follows normal provider-failure handling.
Failure records are best effort; failure to write one is logged without masking
the original error. There is no resume or selective rerun support.

INFO logging reports case progress; DEBUG adds session timing and usage.
Native OpenTelemetry spans supply model request and tool timing, including
nested summarizer requests. Set `MAM_BENCH_LOG_LEVEL` to override the YAML
`log_level`. Logs and native failure events omit sensitive exception bodies.
Conversation artifacts intentionally retain model content.

The CLI collects all three telemetry signals locally using standard SDK JSON
exporters. Native instrumentation includes prompts, replies, reasoning, and tool
arguments/results by default, both locally and when sending to Logfire.
Request/tool spans export when they finish; metrics export every 30 seconds and
at orderly shutdown. Harness owns the complete conversation archives, including
the history retained before a failure. Read each JSONL record independently;
the console-export JSON is a diagnostic format, not an OTLP protobuf dump.
Native `gen_ai.agent.call.id` matches the archived messages' `run_id`;
`gen_ai.conversation.id` identifies the session. Use `FileStepStore.list_runs`
and `list_snapshots` to read matching histories instead of decoding archive
directory names or creating a second identity system.

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

Prompt, completion, tool content and model request parameters are disabled in
telemetry. The export adapter also removes exception-event bodies and span
status descriptions, which native instrumentation can populate despite disabled
content capture. Safe exception types and stack locations remain in package
failure logs. Full local conversations remain sensitive artifacts.

Library imports do not initialize telemetry. `collect_telemetry(directory)`
owns one invocation's process-wide agent instrumentation and scoped package
logging; run independent collection contexts in separate processes. Scientific
results and trajectories retain their existing atomic/checkpoint writes.
