# Artifacts and failures

```text
<output>/<timestamp>-<uuid>/
  config.json                   # original requested model and suite
  effective-model.json          # resolved model settings and endpoint metadata
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
    usage.json                   # native request/token/cache totals for the case
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
The effective setup records the metadata used and which model choices came from
YAML versus derived defaults. Per-case usage includes summarization calls. A
zero cache-token total alone does not establish that the provider reported a
cache miss. `cache_observation` counts action responses with cache telemetry and
uses `null` for read/write totals when any action response did not report them;
summary calls still appear in native usage totals. Successful provider usage
is tallied even when a later request fails, and partial `usage.json` is saved
on case failure when storage permits. Per-response reporting is also retained
in native message snapshots.

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
