# Logfire trial and instrumentation design

Status: draft for review, 2026-10-12. Viewer selected: Logfire, new free Personal
project `joe-helbing/mam-bench` in the US region. Branch: `codex/observability`,
based on upstream `dev`. The initial SDK trial is verified. The first integration
slice now uses the native OpenTelemetry SDK for local collection and optional
Logfire delivery; the broader scheduler and PGX collection design remains proposed.

## Outcome

One Logfire project should let us follow an agent turn, inspect its retry or
fallback, see current benchmark progress, and compare scheduler occupancy with
server throughput and GPU activity. Keep native scientific archives locally.
Logfire's default 30-day history is a debugging window, not the paper archive.

Use OpenTelemetry conventions and native Pydantic AI instrumentation. Domain
attributes supply the benchmark context. A receiver/export failure must not
abort the benchmark or change agent actions. No external guard is added.

## Trial before integration

The prepared scratch program uses the existing `AgentSessions` with a
`FunctionModel`. One agent makes three synthetic model requests, rejects one
invalid tool action, then accepts the corrected action. It produces:

- A manual trial span with a stable synthetic run ID.
- Native agent, model-call and tool spans using `Agent.instrument_all()`.
- Existing package DEBUG logs through `LogfireLoggingHandler`.
- A custom completed-turn counter and duration histogram.

The response token counts are deliberately synthetic and must not be used as
model measurements. No OpenRouter, local inference endpoint or PGX is called.
Prompts, completions, tool content and full request parameters are not exported
in the native trace. Tool exception text is deliberately harmless synthetic text.

Trial SDK: `logfire==4.41.0` was installed in an isolated UV overlay. The project
uses `opentelemetry-sdk==1.44.0`, its HTTP exporter at the same version, and
`opentelemetry-instrumentation-logging==0.65b0`. Installation of each was approved.
The project does not need the Logfire SDK: standard OTLP delivers all three signals.

Before accepting the trial, verify cloud ingestion, not just an SDK flush:

1. Find the synthetic run and expand the native model and tool spans.
2. Find `trial.start`, retry context and `trial.end` on the same trace.
3. Find `mam_bench.trial.turns` and `mam_bench.trial.turn.duration` in Metrics;
   a single run produces a point, not a meaningful time-series trend.
4. Confirm the native request count and synthetic usage are not double-counted
   with parent agent totals.
5. Check usage, credentials location and the absence of model-provider calls.

### Verified trial evidence

On 2026-10-12, the in-memory exporter verified one correlated trace, three
native model requests, 14 package log records, the counter value 1 and histogram
count 1. Request-level synthetic token totals were 30 input and 6 output.
The successful tool-call usage count was 1; native tool spans showed two
attempts, including the rejected action. Keep attempts, retries and successes
distinct in production metrics.

Cloud trial run ID: `synthetic-0bb0a9425e1340ad8e014a9f8d44c302`.
The SDK flush succeeded. The authenticated Live view independently showed
`mam_bench.synthetic_trial`, its native agent child and correlated start/end
logs. The Metrics explorer independently listed `mam_bench.trial.turns`,
`mam_bench.trial.turn.duration` and native `gen_ai.client.token.usage`.
The Personal plan page confirmed 10M records/month included free.

The user renamed `starter-project` to `mam-bench`; existing trial data remained
visible. Local SDK project metadata was refreshed using the new name.
Global SDK CLI authentication is in the user's native `~/.logfire` directory;
project write credentials are inside the ignored trial `.logfire` directory.
The newer `logfire-cli` wizard authenticated separately via the OS credential
store. Its agent launcher was cancelled because this chat owns the setup.
No credential values are included in this document.

Logfire 4.41.0's standard logging handler sends zero-duration OTel spans marked
`logfire.span_type=log`. Thus this trial verifies traces, correlated logs in
Logfire, and metrics; it does not establish a separate OTLP Logs wire stream.
If that separate signal is a requirement, configure a native OTel LoggingHandler
and LogRecord exporter instead of duplicating delivery through both handlers.

The SDK CLI's project-status check requires an additional read token. None was
created; verification used the authenticated UI. Evidence and executable trial
programs remain in `.scratch/logfire-trial/`; the viewer is
[MAM-Bench in Logfire](https://logfire-us.pydantic.dev/joe-helbing/mam-bench).

## Trace ownership and context

At run-directory creation, configure telemetry once, before creating agents. Attach the
native OpenTelemetry logging handler only to the `mam_bench` logger; preserve stderr output
and leave unrelated/root logging alone. Library imports do not initialize cloud
access. Test processes use a local collecting exporter or disable sending.

The implementation uses the existing run directory identity as `run_id` on the
benchmark span and `mam_bench.run_id` on all three signals' resources. Further
domain instrumentation belongs in the owner that already performs the work:

- Runner: benchmark start/end/failure and case evaluation.
- Simulation: ordinary/controlled worlds, settled steps and citizen/police phases.
- Scheduler: cohort/phase waiting, per-agent ready/admitted/finished timestamps,
  active/pending counts and cancellation.
- Session: turn context and completion outcome around native agent spans.
- Compaction: parent turn context, strategy, before/after context size, summary
  request purpose, tokens and duration.

Carry `run_id`, case index, simulation, world, step, phase, cohort, agent ID,
turn ID and request purpose through task-local context. Concurrent tasks must
not share a mutable global context dictionary. SDK response IDs stay distinct
from locally assigned logical request IDs and optional transport attempt IDs.

Native Pydantic AI instrumentation owns agent/model/tool spans. Avoid a second
custom span for each model request. Add domain attributes and a canonical turn
summary instead of retaining duplicate request.start/end debug narratives in
both systems. Keep a separate HTTP attempt layer only when it supplies retry
visibility hidden inside the provider SDK.

The standard Harness summarizer creates a nested agent. Global instrumentation
can observe it, but MAM context and `purpose=compaction` must carry through.
Verify coverage and avoid double-counting its usage in parent-turn aggregates.

Parent spans naturally provide a run/case hierarchy. Long-lived run spans are
not sufficient live progress: child completions, lifecycle logs and periodic
progress observations must expose ongoing work before the root span ends.

## Logs and error content

Use enriched completion logs for turns and settled-step checkpoints with the
same correlation IDs as traces. Include outcome, elapsed time, request/tool
counts, token totals and fallback/stop category. Capture raw provider stop
reasons separately from library-normalized reasons, including repetition stops.
The repetition adapter work remains separate from this dev-based branch.

Record start/cancellation/failure events so interrupted runs are intelligible.
Classify timeout scope: whole-turn deadline, HTTP request or internal operation.
Retain safe exception types and stack locations as the existing policy does.

Disabling native content capture alone does not prove provider exception bodies
are protected: native SDK spans may record exception messages/stack traces.
Production export must sanitize those fields before cloud delivery. Choose the
SDK-supported export/scrubbing seam after inspecting the installed trial SDK;
test with an exception containing a sentinel secret before enabling production
instrumentation. No raw HTTP headers/bodies or `capture_all=True` baseline.

Native instrumentation captures prompts, replies, reasoning, and tool
arguments/results by default. Logfire receives that content when cloud delivery
is configured; complete conversation histories also remain in Harness archives.
Provider exception messages and stack traces are still omitted from exports.

## Metrics

Keep IDs in traces/logs; metric dimensions stay bounded: simulation, world,
phase, purpose, outcome and model. Record:

| Group | Measurements |
| --- | --- |
| Progress | Completed cases/steps/turns; last settled-step time; active turns |
| Scheduling | Admission wait histogram; pending and active turns; phase duration |
| Agent work | Requests/tool calls; turn duration; retries/fallbacks; board and private-memory operation counts/durations |
| Compaction | Count, duration, context size and summary token use |
| Inference server | Running/waiting requests, KV occupancy, token counter deltas, exposed queue/prefill/decode histograms |
| Hardware | Supported GPU activity, power/clocks, system memory and swap |
| Collection | Last-success time, success/error counts and sample coverage |

Use native GenAI token metrics as the authoritative request-level aggregation
when supplied. Parent agent/turn usage totals serve as summaries, not additional
tokens to sum. Store unsupported or missing provider fields as unknown.

Use delta metric export where needed for Logfire dashboards. Prometheus-format
vLLM metrics can be collected by an OTel Collector receiver without operating a
Prometheus server or Grafana. Keep that collector configuration outside the
simulation code and retain an allowlist of metrics from each pinned image.
OTLP Summary metrics are unsupported by Logfire; preserve counters/histograms
and compute quantiles at query time.

Hardware collection runs near the device; the viewer/collector's placement must
avoid consuming PGX inference resources unnecessarily. Verify supported driver
counters on PGX before choosing a GPU receiver. Device activity is kernel-active
time, not percent of peak FLOPs. Collect unrelated-workload context if needed.

Collector ownership is passive: no Docker health checks or workload stop action.
Failures create a gap, not a zero reading. Timings use monotonic clocks locally
and UTC across hosts. Report time-weighted utilization and sample coverage.

## Persistence and export behavior

Native results, trajectories and messages retain existing persistence rules.
Retain telemetry snapshots locally with runs for paper provenance. Decide the
smallest supported archive route after the trial: SDK exporter to append-only
OTel records or a collector file export. Do not build an additional application
event system merely to mirror native spans. Verify the selected archive format
can be read and correlated without Logfire access.

SDK export is bounded and asynchronous; flush with a bounded timeout at orderly
shutdown. Remote outages cannot make model calls fail or block shutdown
indefinitely. A hard kill may lose buffered telemetry; local scientific
checkpoints and cloud lifecycle observations must keep that limit explicit.
Collector buffering/backpressure and diagnostic status are reviewed separately
from simulation failure handling.

## Implemented collection and simplification

`telemetry.py` initializes and closes explicit SDK providers. Native console
exporters write traces/logs/metrics JSONL under the run directory. Native HTTP
exporters deliver the same records to Logfire when `LOGFIRE_TOKEN` is present.
Local traces and logs flush on completion; metrics export every 30 seconds and
at shutdown. Cloud traces/logs use standard batch processors; HTTP timeout is
three seconds. The CLI owns the context in its async task; library imports do
not initialize exporters. Run separate invocations in separate processes.

`Agent.instrument_all()` supplies model/tool traces and native GenAI metrics,
including nested summarizer agents. Parent usage has `gen_ai.aggregated_usage.*`
names, distinct from request usage. The custom `_RequestLogging` capability and
its imports were removed (80 fewer lines in sessions). Turn summaries and safe
failure classification remain. No simulation logic, scoring, RNG, admission
order or provider parameters change.

`artifacts.py` already delegates history persistence to Harness `FileStepStore`.
Its small archive adapter protects provider-error bodies and classifies write
failures. Its JSON writer retains atomic results and scientific checkpoints.
`diagnostics.py` retains fixed failure categories, safe stack locations and
stderr setup. Deleting these responsibilities would lose protections or move
equivalent code into callers. No replacement history serializer was written.
The telemetry setup adds code overall while removing duplicate request/tool
collection; this is a new observability capability, not a net line-count reduction.

### Privacy issue found during integration

An installed Logfire exception callback sanitized exception events but native
OpenTelemetry subsequently repopulated span status descriptions with a synthetic
provider-error sentinel. Disabling content capture also leaves exception events
enabled. Logfire's default scrubbing excludes exception-message/stacktrace keys.
The implementation therefore sanitizes public `ReadableSpan` records at the
export seam, preserving exception types and status codes while removing bodies
and status descriptions for both local and cloud delivery. Package logs retain
safe stack locations. Native Harness failure events need their existing sanitizer.
These are narrow policy adapters around maintained exporters, not custom transports.

## Validation and staged rollout

1. The initial synthetic SDK cloud trial is verified. The implemented pipeline
   uses separate OTLP traces, logs and metrics exporters; cloud verification is
   recorded below when complete.
2. Public-interface tests verify complete reasoning/tool histories, partial
   histories on provider failure, native request counts/tokens and sentinel
   exclusion, including trace-to-archive native call IDs. Nested compaction adds
   one request span without duplicating usage. A loopback receiver rejects all
   three signals without preventing
   agent completion or local evidence preservation.
3. Paired deterministic benchmark runs produce identical scores and saved
   ordinary/controlled trajectories with telemetry on/off. Existing archive,
   compaction, cancellation and scheduler tests retain their coverage.
4. A bounded workload measures instrumentation overhead and emitted record count.
5. Passive collection validates cumulative counters, restarts, missing samples,
   metric temporality and PGX counter availability.
6. One selected benchmark can then test live views and locally retained evidence;
   restarting the Gemma/Qwen campaign is a separate explicit step.

The implemented pipeline's synthetic cloud trial on 2026-10-12 used run ID
`native-otlp-48eee8b34d2a489eb754c939d9702c51`. Logfire Live independently showed
the benchmark span, successful and failed agent children, and correlated
`native-otlp-trial.start/end` records with `kind=log`. Metrics independently
showed `gen_ai.client.token.usage` for service `mam-bench` in the trial window.
Synthetic usage was 20 input and 4 output tokens; its histogram average of 6
is not a total. Local records exclude the synthetic provider-error sentinel.
Screenshots and executable verification remain in ignored `.scratch/logfire-trial/`.
No real model call or historical conversation was sent to Logfire.

Fresh local verification: all 95 unittest cases passed in 92.911 seconds.
Ruff lint and format checks passed; Pyright reported zero errors or warnings.

Do not merge until Joe agrees with the setup. Initial account/project work stays
on Personal; no paid upgrade or billing changes are part of this trial.

## Sources

- [Pydantic AI native OTel](https://pydantic.dev/docs/ai/integrations/logfire/)
- [Logfire first trace and account setup](https://pydantic.dev/docs/logfire/get-started/first-trace/)
- [Standard-library logging](https://pydantic.dev/docs/logfire/integrations/logging/logging/)
- [Metric instruments and polling](https://pydantic.dev/docs/logfire/instrument/python/add-metrics/)
- [Write tokens](https://pydantic.dev/docs/logfire/manage/create-write-tokens/)
- [Pricing](https://pydantic.dev/pricing)

GPU/server collection, domain scheduler metrics, a measured overhead study and
real benchmark rollout remain separate steps. The native archive and export
sanitization are implemented in this slice, using installed Pydantic AI 2.38.0,
Harness 0.29.0 and OpenTelemetry 1.44.0.
