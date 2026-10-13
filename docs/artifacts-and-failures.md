# Artifacts and failures

```text
<output>/<timestamp>-<uuid>/
  config.json
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

INFO logging reports case progress; DEBUG adds session timing and usage.
Native OpenTelemetry spans supply model request and tool timing, including
nested summarizer requests. Set `MAM_BENCH_LOG_LEVEL` to override the YAML
`log_level`. Logs and native failure events omit sensitive exception bodies.
Conversation artifacts intentionally retain model content.

The CLI collects all three telemetry signals locally using standard SDK JSON
exporters. Request/tool spans export when they finish; metrics export every 30
seconds and at orderly shutdown. These files do not contain complete histories:
Harness owns those lossless archives. Read each JSONL record independently;
the console-export JSON is a diagnostic format, not an OTLP protobuf dump.
Native `gen_ai.agent.call.id` matches the archived messages' `run_id`;
`gen_ai.conversation.id` identifies the session. Use `FileStepStore.list_runs`
and `list_snapshots` to read matching histories instead of decoding archive
directory names or creating a second identity system.

Setting `LOGFIRE_TOKEN` enables additional background OTLP HTTP delivery to
Logfire. `LOGFIRE_BASE_URL` defaults to `https://logfire-us.pydantic.dev`; use
the base URL for the region of the token. Keep credentials outside YAML and Git.
Local collection works without a Logfire account or token. Remote delivery
errors do not fail agent turns; HTTP exports have a three-second timeout.
An abrupt process kill can lose unfinished spans and buffered cloud records.

Prompt, completion, tool content and model request parameters are disabled in
telemetry. The export adapter also removes exception-event bodies and span
status descriptions, which native instrumentation can populate despite disabled
content capture. Safe exception types and stack locations remain in package
failure logs. Full local conversations remain sensitive artifacts.

Library imports do not initialize telemetry. `collect_telemetry(directory)`
owns one invocation's process-wide agent instrumentation and scoped package
logging; run independent collection contexts in separate processes. Scientific
results and trajectories retain their existing atomic/checkpoint writes.
