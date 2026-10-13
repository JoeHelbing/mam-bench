# Observability options for MAM-Bench

Status: initial options proposal, 2026-10-12. The selected Logfire integration
and current implementation are in [logfire-design.md](logfire-design.md).
Prepared on `codex/observability`, based on upstream `dev` commit
`04d59f9fba2fd9b825bfcd51804bef43faedd35c`.

Direction update: Joe prefers OpenTelemetry metrics, traces and logs and excludes
Grafana and Jaeger. Compare hosted and self-hosted viewers before developing
the full design. The JSONL-first recommendation below is the earlier proposal,
not the selected implementation direction. See `telemetry-viewers.md` for the
current comparison. Existing scientific archives remain independent evidence.

## Recommendation

Start with structured local events and passive server/hardware measurements.
Reuse the existing session hooks and artifact directory. Add a tracing backend
only when a concrete debugging task benefits from its timeline or query UI.

The goal is to answer these questions without reconstructing an experiment by
hand:

- Is the run making progress? Which case, world, step and phase is active?
- Are agents waiting for admission, calling the model, using tools, or compacting?
- Does a model do more work, or does scheduling leave the GPU idle?
- Which request caused a timeout, repeated output, retry, or fallback?
- Which source revision, suite, model, serving recipe and request policy ran?
- How much of the run has valid measurements, and what remains incomplete?

Observability observes. It must not recreate the rejected external guard or stop
a benchmark because a metrics poll, exporter or dashboard fails. Existing
scientific artifacts and their persistence requirements remain separate.

## What exists today

Verified in the branch's source, rather than inferred from old run status:

| Existing facility | What it provides | Main gap |
| --- | --- | --- |
| `diagnostics.py` | Package-only stderr logging; configurable level; exception types and stack locations without bodies | Text records, no common run/case/turn/request context |
| `runner.py` | Case start/end, classified failures, completed case results | Little visibility during a long case |
| `sessions.py` | DEBUG request/tool/turn durations, tokens and outcomes; compaction timing | No joined scheduler timeline; compaction logs lack session context |
| `scheduling.py` | Concurrent admission and cohort/phase ordering | No measurements of queue delay, occupancy or phase waiting |
| `artifacts.py` and simulations | Completed-step trajectories, messages, reasoning, tool histories, fallbacks and results | Rich forensic evidence but expensive to interpret as live progress |
| Scratch campaign launcher | Sampled vLLM running/waiting requests and KV occupancy | Outside package; poll failures are silently skipped; no coverage accounting |

The existing request hook wraps logical model calls. It should not be assumed
to count every HTTP attempt hidden inside the provider SDK. The history
summarizer creates a nested agent using the shared model; the outer decision
agent's request hook is not automatically attached to that nested agent.
Native global agent tracing can cover this path, with context propagation and
distinct summary attribution verified before use.

## Options

These options build on one another; they are not mutually exclusive.

| Option | Best use | Cost and limitations |
| --- | --- | --- |
| Local JSONL events and a live status view | Reproducible per-run evidence, progress, timing breakdowns, simple queries | Smallest change; no always-on backend. A timeline/query view still needs to consume the events. |
| Native Pydantic AI OpenTelemetry instrumentation | Follow one turn through model calls and tools; inspect nested timelines | Reuse framework instrumentation, add MAM context. Requires an SDK/exporter and a backend for useful collection. Provider HTTP retries and server internals need separate coverage. |
| vLLM metrics with Prometheus/Grafana and GPU telemetry | Historical queue, latency, token throughput, cache and hardware trends | Separate infrastructure to operate; complements application events. Aggregate server metrics cannot explain an individual agent's behavior. |

For tracing, Logfire is the framework's documented integration; a self-hosted
OpenTelemetry backend such as Jaeger is an alternative. Cloud export should be
an explicit choice. Local event recording does not depend on either backend.

The current lockfile has `opentelemetry-api` and `logfire-api` through dependencies,
but not an OpenTelemetry SDK/exporter or full Logfire package. An installed API
does not mean telemetry is being collected.

## Minimum useful instrumentation

Use stable correlation fields: `run_id`, `case_id`, `world`, `step`, `phase`,
`agent_id`, `turn_id`, and a locally generated `request_id`. Preserve a separate
provider response ID when available; failed calls may never receive one.

Emit one enriched completion record for each logical model request and turn.
Include outcome, duration, token usage, terminal tool, retry/fallback category,
and an archive reference. Preserve the raw provider finish/stop reason separately
from a library-normalized reason so repetition stops are distinguishable from
output-cap exhaustion.

Also record starts and completed-step checkpoints. A run can spend minutes in
one request or be killed before a completion event; completion-only logging
cannot explain its live state or prove a started operation finished. A bounded
status snapshot can show active operations and the age of the last observation.
It should distinguish observed failure, completion, stale data and unknown state.

Add timings at the seams already present:

- Scheduler: ready/enqueued, admitted, finished; selected/ordinary cohort and
  citizen/police phase; active and pending turns.
- Model: call start/end, tokens, provider ID, finish reason; distinguish agent
  decisions from history-summary calls.
- Tools and memory: calls, retries, duration and content sizes; keep full content
  in the existing archives. Separate private-memory operations, shared messages
  and compaction.
- World: settled-step checkpoint and phase duration; do not call an unsettled
  partial step complete.
- Run: source/suite/model/request-policy/serving-recipe fingerprints and lifecycle
  state; failure scope and partial-artifact references.

Use UTC for cross-host alignment and a monotonic clock for durations. Record
timestamp source and sampling interval for hardware/server samples. Never
subtract clocks on different hosts to manufacture a precise queue duration.

## Measuring performance honestly

Collect server running/waiting requests, KV occupancy, prefill/output token
counters and available latency histograms from the actual installed vLLM
endpoint. Record exposed metric names/version rather than assuming today's docs
exactly match the pinned campaign image.

Collect supported GPU activity/utilization, power, clocks and memory measurements
on PGX. Validate availability on its hardware before choosing an exporter. A
GPU activity percentage is a sampled device signal, not a percentage of peak
FLOPs or proof of useful model work.

Report:

- Wall-clock run duration, settled steps and completed turns.
- Aggregate output tokens per wall-clock second; prefill/input work separately.
- Scheduler admission delay and active-slot occupancy over time.
- Model-request elapsed time, tool/memory time and compaction time.
- Repetition stops, cap stops, retries, fallbacks and token counts by purpose.
- Server queue/cache measurements and time-weighted GPU activity, including
  valid idle samples and the fraction of the run covered by measurements.

Overlapping request durations must not be added and presented as wall-clock
time. Client request time includes transport, server waiting, prefill and decode;
it is not pure generation time. The existing non-streaming client cannot infer
time-to-first-token from total elapsed time. Server histograms can describe
population latency; individual correlation requires explicit support.

Keep missing samples missing and report coverage. Scrape outages must not become
zero utilization. If a model generates more tokens, makes more calls or spends
longer compacting, that is model behavior to report separately from scheduler
efficiency. Controlled comparisons still need matching workloads and policies.

## How to use the guide

[Logging Sucks](https://loggingsucks.com/) argues for enriched request records,
consistent context and queryable fields. Those ideas fit this benchmark well.
Its high-volume web-traffic sampling advice needs adaptation: retain all small
request/turn records for research denominators. Keep bulky model content in the
native archives. Combine completion records with lifecycle/checkpoint events
for long-running and interrupted experiments.

High-cardinality run/agent/request identifiers belong in events and traces.
Keep Prometheus labels bounded; do not create a time series for every request.
Prefer one local event representation before adding multiple independently
maintained logging paths.

## Reviewable first increment

Proposed, not yet implemented:

1. Add joined structured metadata to existing request/turn/compaction hooks and
   scheduler admission points, with checkpoint progress at world settlement.
2. Retain events with each run and expose a readable live status summary.
3. Save passive vLLM and supported PGX hardware samples with explicit poll errors
   and freshness/coverage fields. Telemetry failure cannot abort inference.
4. Verify that instrumentation leaves actions, scores and scheduling semantics
   unchanged; exercise failure/cancellation correlation, concurrent context
   isolation and summary-request coverage. Measure instrumentation overhead with
   a bounded deterministic workload.

This increment requires no new monitoring backend. Optional backend integration
can follow once the event fields and live view answer real debugging questions.

## Source map

Repository facts above come from `src/mam_bench/{diagnostics,runner,sessions,
scheduling,artifacts,runtime}.py`, both simulation implementations,
`docs/artifacts-and-failures.md`, and `uv.lock` at the stated branch base.
Scratch metrics behavior was inspected in the retained October 12 campaign;
that code is evidence, not part of this proposed package design.

Primary-source research and backend-specific caveats accompany this proposal in
`observability-primary-sources.md`.

The unmodified branch baseline passed all 90 native tests using the existing
project environment and this checkout's source path. No dependencies were
installed. The first sandboxed test process stalled and was interrupted; the
successful verification ran with host access. This verifies the branch baseline,
not the proposed instrumentation.
