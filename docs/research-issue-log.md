# Research issue log

This log records development and inference-serving problems that can affect
MAM-Bench results, runtime, or reproducibility. It separates observations,
interventions, verified outcomes, and questions still requiring experiments.
Keep it with the paper's methods and limitations material.

Initial evidence cutoff: 2026-10-12, 16:30 UTC. Entries are dated snapshots,
not a live campaign dashboard. No sampling or serving changes were made to
the active campaign while compiling these notes.

## Issue index

| ID | Problem | Status at cutoff |
| --- | --- | --- |
| R01 | Claim ordering and shared-step stragglers may reduce utilization | Admission changed; efficiency comparison remains open |
| R02 | Gemma thinking repeats until the output cap | Measured; mitigation untested |
| R03 | Served context and output reservation exceed the window | 256k context validated; compaction estimator remains open |
| R04 | Gemma parser converts integer tuple coordinates to strings | Scoped parser workaround verified |
| R05 | Malformed tool names still cause retries | Measured; cause and fix open |
| R06 | Model startup causes unified-memory allocation pressure | Explicit KV cap and guarded startup validated for tested setup |
| R07 | Long runs depend on interactive process lifetime | Independently supervised campaign implemented |
| R08 | Observability can misstate concurrency or hide generated tokens | Queue sampler corrected; raw-output visibility remains open |
| R09 | Short MTP trials do not isolate throughput or quality | Recorded as exploratory trials only |
| R10 | Partial runs, fallbacks, and inherited provenance fields can mislead | Verification and reporting rules documented |
| R11 | Shared sampling defaults differ from Gemma guidance | Recorded; effect unmeasured |

## Evidence and update rules

- Add an entry when a problem is observed, before trying a fix. Record UTC
  timestamps, exact symptoms, model/runtime versions, source and suite hashes,
  request settings, and an artifact path. Preserve failed attempts.
- Append dated follow-ups rather than rewriting an unresolved cause as fact.
  Use **observed**, **reproduced**, **mitigated**, or **verified fix** explicitly.
- Record what changed and what stayed fixed in every comparison. Separate
  application admission, inference-server queueing, and device utilization.
- Record failed requests and fallback actions alongside scores and timing.
  A mitigation that changes decoding or scheduling is a new experimental
  condition, even if it improves reliability.
- Paths below are relative to the repository root. `.scratch/` artifacts are
  Git-ignored local evidence, not a permanent or published research archive.
  Before paper submission, retain a reviewed, redacted evidence bundle with
  checksums and stable storage. This document preserves the findings in Git;
  it does not by itself archive every underlying trace.

> 🧠 **From Hindsight memory (Rerun Qwen thinking and Gemma thinking modes on arrival-order scheduling)** — Changing scheduling changes score semantics, so
> older results are not a controlled thinking-mode comparison. Retained source
> and campaign files were checked for this log; the page's older campaign
> status is superseded by the dated receipts below.

## R01 - Scheduling, claim order, and shared-step tail latency

**Observed, 2026-10-10 to 2026-10-12.** The older scheduler allowed concurrent
model decisions but gated claims by seeded priority. The replacement accepts
ready actions without that claim gate and admits another queued turn when a
slot becomes free. It retains one turn per eligible agent per shared step,
selected/ordinary cohort boundaries, unchanged step observations, legality
checks, and settlement barriers.

**Intervention and validation.** Arrival-order admission shipped in commit
`14a97f91649912ff376892211652980b5467e363`, [PR #36](https://github.com/JoeHelbing/mam-bench/pull/36).
The recorded validation passed 90 tests, Ruff, formatting, and Pyright.
Native events measured 16 overlapping client requests and equal turn counts.
Current code is [scheduling.py](../src/mam_bench/scheduling.py).

**Remaining problem.** A shared step waits for its slowest eligible turns.
Gemma thinking case 1 averaged 3.43 running server requests against an eight-slot
ceiling and filled all eight slots in about 24.3% of retained samples. This is
batch occupancy, not GPU utilization. The older Qwen case also reached 16
overlapping client requests; describing it as wholly sequential is incorrect.

**Causal limit and paper implication.** Cross-model duration cannot establish
that a scheduler is faster or slower. Model-generated work, retry rates,
server slots, MTP, parser behavior, and sampling differ. Historical timing
improvements are preliminary, not a controlled scheduler result. Arrival-order
claims are timing-sensitive and may change simulation outcomes.

**Required follow-up.** Compare schedulers on the same model, serving recipe,
suite, sampling, and client/server concurrency. Sample device utilization,
power, running/waiting queues, completed tokens, and step/phase boundaries on
one clock. Report idle time and barrier tails. Use fixed request replays to
isolate scheduling overhead, then separate end-to-end runs to measure changes
in agent behavior and scores. Repeat runs rather than relying on one timing.

Evidence: `.scratch/model-runs/gemma-qwen-256k-20261011/runtime-comparison.json`,
`runtime-comparison-cases-2-3.json` in that directory; current campaign queue
samples at `gemma-thinking/queue-metrics.jsonl`; old Qwen artifacts at
`.scratch/model-runs/pgx-current-20261008/results/20261009T043635Z-75d7eb892a16`.

## R02 - Repetitive thinking traces and output-cap exhaustion

**Measured, 2026-10-12.** In Gemma thinking cases 1 and 2, 75 of 1,465 retained
responses (5.12%) met the conservative exact-repeat screen. All 75 consumed
32,768 reported output tokens. These responses accounted for 2,457,600 of
4,960,281 reported output tokens (49.55%). This is their entire output-token
share, not a count of tokens proven wasted. Their dominant phrases repeated
278-2,551 times. Client request latency was 21.1-30.9 minutes, including
queueing and generation. Case 3 had no flagged responses among 663.

The historical Qwen thinking run had one flagged response among 3,415 across
six cases: a sentence repeated 616 times. It also had 112 capped responses.
Many capped Qwen responses retain far less parsed content than their reported
output usage suggests; we cannot conclude the remaining raw generation was
free of loops. A token cap alone is not evidence of repetition.

**Method.** Deduplicate overlapping native snapshots by provider response ID;
check counts against request-completion events. Normalize whitespace and
screen lines or sentences of at least 12 characters repeated at least 20
times, whose repeated units cover at least 20% of retained thinking text.
The detector misses paraphrased and weaker loops. The compared models use
different serving setups and the Gemma campaign is incomplete.

**Mitigation status.** The current cap bounds one response's cost; it does not
prevent loops or guarantee an action. Repetition/frequency/presence penalties,
N-gram early stopping, and reasoning-budget controls are candidates for a
separately labeled ablation. No penalty or detector efficacy has been tested
in the current campaign. See the controls appendix below.

**Required follow-up.** Replay problematic prompts with baseline decoding and
one intervention at a time. Measure flagged-loop frequency, generated tokens,
valid tool-call rate, retries, fallbacks, latency, and task outcomes. Early
stopping may interrupt before a valid tool call; account for the resulting
retry or fallback. Evaluate false positives on legitimate repeated reasoning
and tool syntax. Any selected policy needs independent benchmark validation.

Evidence: `.scratch/model-runs/gemma-qwen-256k-20261011/repetition-audit.py`,
`repetition-audit.json`, and the native snapshots identified in that JSON.

## R03 - Context overflow and insufficient compaction

**Reproduced, 2026-10-11.** A failed non-thinking campaign served only 65,536
tokens while reserving 32,768 output tokens. The preserved HTTP 400 says the
prompt contained at least 32,769 tokens, exceeding the combined window by at
least one token. Raising a client-side context declaration cannot enlarge the
server window.

**Intervention and validation.** Both Gemma modes now declare and serve
262,144 tokens. A replay of the failed request returned HTTP 200 with 33,392
input and 379 output tokens. A near-full-window request with 228,575 input
tokens plus a 32,768 output reservation (261,343 total) completed with HTTP 200
and a tool call. That trial generated seven output tokens; it verifies the
reservation was accepted, not simultaneous near-full prompts and maximum
outputs for eight clients.

Eight server slots, 16 clients, and a 24 GiB FP8 KV allocation allow excess
requests to queue. The corrected queue trial completed 16/16 HTTP requests,
with 14 parsed tool-call responses and a peak of 15 waiting. Its observed peak
was four running; eight is the configured ceiling, not the trial's measured
peak. Larger context alone does not prove arbitrary concurrency fits.

**Still open.** Earlier inspection found provider input usage about 31,742
tokens against a character-based history estimate about 16,255. A 16,000-token
retained-tail estimate could therefore preserve most history despite
compaction. The larger window mitigates overflow but does not repair that
estimation mismatch. Current trigger fraction is 0.7, retained-tail estimate
40,000, and summary output cap 16,000.

**Required follow-up.** Tokenize the full serialized request, including tools
and template overhead, reserve completion space, and verify that compaction
actually reduces the measured prompt. Add a context-boundary regression when
implementing the fix. Keep this separate from evaluating 256k serving.

Evidence: `.scratch/model-runs/gemma-qwen-ordered-20261011/gemma-non-thinking/provider-errors.jsonl`;
`.scratch/model-runs/gemma-qwen-256k-20261011/pilot/probe-receipt.json`,
`pilot-queue8/probe-receipt.json`, `gemma-thinking/model.yaml`, and
`gemma-thinking/recipe.yaml` in the latter directory.

## R04 - Nullable integer tuples were parsed as string arrays

**Reproduced, 2026-10-10.** In the pinned vLLM Gemma parser, native tool syntax
such as `destination:[10,13]` became `{"destination":["10","13"]}`. Schema
repair handled homogeneous array `items`, but not nullable tuple `prefixItems`.
Strict integer validation rejected these coordinates. An initial two-case,
one-step thinking-off smoke exhausted retries for 31/32 agents; thinking-on
also failed on numeric-string coordinates.

**Verified workaround.** A read-only parser plugin repairs types according
to the existing schema, preserving strict validation and the frozen suite.
An exact-wire replay with identical native token IDs changed string coordinates
`["6","15"]` to integer coordinates `[6,15]` and returned HTTP 200. Separate
thinking-on/off one-step integration smokes each completed all 32 agents with
zero retry exhaustion. The plugin SHA-256 is
`b0278d87ac9a225f8513bac006a572206bd6245bdef55ed07beaf6eefaf8c004`.

**Limit.** This fixes one demonstrated schema conversion defect, not all tool
generation. It is a serving intervention that must accompany model results.
Original failed responses did not all retain raw tokens, so the offline
reproducer does not prove every historical error began with numeric tokens.

Evidence under `.scratch/model-runs/pgx-three-model-20261009/vllm/`:
`parser-diagnosis.md`, `gemma4-tuplefix-plugin.py`,
`tuplefix-second-exact-wire-replay-receipt.json`,
`tuplefix-second-off-smoke-receipt.json`, and
`tuplefix-second-on-smoke-receipt.json`.

## R05 - Malformed native tool names remain after tuple repair

**Measured, 2026-10-12.** Gemma thinking case 2 retained 171 tool calls whose
names contain malformed syntax, for example
`move(destination=null)<tool_call|>` instead of `move`. Native retries report
unknown tool names. The case recorded seven retry-exhaustion fallbacks,
57 output-token-limit fallbacks, and one usage-limit fallback.

**Status.** Tuple repair does not fix this category. Checkpoint behavior,
chat template, parser boundaries, and quantization are competing hypotheses;
the retained parsed response alone cannot identify the cause. Repetition and
tool-format errors may interact, but correlation does not establish direction.

**Required follow-up.** Capture raw token output, parse it offline through the
pinned parser, and distinguish incorrect generation from incorrect extraction.
Keep valid schema/tool names fixed while testing a parser or decoding change.

Evidence: `.scratch/model-runs/gemma-qwen-256k-20261011/runtime-comparison-cases-2-3.json`
and corresponding case-2 native response and retry snapshots.

## R06 - Startup allocation pressure on GB10 unified memory

**Observed, 2026-10-10.** One Gemma MTP-off startup recorded 15 NVIDIA
`NV_ERR_NO_MEMORY` allocation failures; sampled available host memory fell to
12,012 MiB. A second guarded startup, with eager execution and autotuning
disabled, still recorded 11 GPU allocation errors. Its worker log reported
55.78 GiB available for KV allocation before the safety stop. Neither attempt
became an inference endpoint or produced a model score.

**Mitigation and validation.** An explicit 24 GiB FP8 KV cap, eager execution,
disabled FlashInfer autotuning, scoped health/memory guards, and later eight
server slots were used. The explicit KV allocation overrides the nominal
`gpu_memory_utilization=0.75` sizing rule. Guarded small trials recorded no new
GPU OOM; the completed 256k non-thinking run retained at least 42.64 GiB
available RAM and passed co-tenant health/restart checks.

**Limits.** Available host RAM, per-process GPU accounting, KV occupancy, and
device utilization are different quantities. This demonstrates a feasible
tested setup, not a maximum memory capacity or a single isolated cause. The
first failed container's complete runtime log was unavailable after stopping.
Current health checks do not establish uninterrupted service during an old
incident. Preserve startup logs before future cleanup.

Evidence under `.scratch/model-runs/pgx-three-model-20261009/vllm/`:
`startup-incident-audit.md`, `mtp0-safety-stop-receipt.json`,
`guarded2560-safety-stop-receipt.json`, `mtp-short-comparison.json`;
current `.scratch/model-runs/gemma-qwen-256k-20261011/gemma-non-thinking/verification.json`.

## R07 - Campaign process lifetime and long-running supervision

**Operational incident, 2026-10-11.** Session history records a previous
thinking attempt interrupted during desktop-run cleanup. Its benchmark archive
does not independently identify the terminating process or signal. Preserve
it as an interrupted attempt, rather than attributing its lack of a score to
model behavior.

**Intervention.** The replacement campaign is owned by Pitchfork, independently
of the interactive agent turn. It persists status, launches jobs in the
authorized order, checks readiness, renews the remote health guard, retains
partial artifacts, verifies completed results, and cleans up the scoped
inference workload. Guard expiry and campaign execution failure remain distinct.

**Validation and limit.** It completed the non-thinking job and advanced to
thinking, with guard renewal retained. This validates the observed transition,
not a destructive desktop-shutdown or host-reboot recovery test. There is no
partial-case resume mechanism; a rerun is a separate attempt.

Evidence: `.scratch/model-runs/gemma-thinking-on-arrival-20261010/`;
replacement `.scratch/model-runs/gemma-qwen-256k-20261011/campaign.py`,
`pitchfork.toml`, `campaign.json`, and `gemma-thinking/guard-renewal-001.jsonl`.

## R08 - Metrics and raw-output visibility

**Corrected measurement defect, 2026-10-11.** The first queue pilot matched
metric-name prefixes and could include waiting-reason series in the primary
waiting count. Its queue counts must not support a concurrency claim. The
replacement matches the exact metric name before labels and retains dated
samples. The corrected pilot receipt reports four running and 15 waiting.

**Unresolved visibility, 2026-10-12.** Native snapshots retain parsed thinking
and tools, not necessarily every generated token. Some Qwen responses report
32,768 output tokens but only short thinking content and a small tool call.
An example has about 600 retained thinking characters and 161 reported
reasoning tokens. This discrepancy needs raw-stream capture before diagnosis.

**Reporting rule.** Distinguish decode throughput, request latency including
queueing, whole-case tokens/wall time, batch occupancy, and actual GPU
utilization. They answer different questions. Do not sum overlapping snapshots;
deduplicate responses and validate against completion events.

Evidence: `.scratch/model-runs/gemma-qwen-256k-20261011/probe.py`,
`queue-probe.py`, `campaign.py`, `repetition-audit.py`, and
`repetition-audit.json` in that directory. The Qwen example response ID is
`chatcmpl-acb2fe0ba150b8da`.

## R09 - MTP timing changes confounded with generated work

**Exploratory trials, 2026-10-10.** Matched two-case, one-step smokes compared
Gemma MTP0/2/4, thinking on/off. Thinking-off times were 37.0/28.5/30.4 seconds.
Thinking-on times were 897.5/1,152.5/1,098.3 seconds, with respectively
127,743/166,343/161,728 output tokens. MTP2 and MTP4 thinking each had one
output-cap fallback; MTP0 thinking completed all 32 agents. All guarded
variants maintained sampled co-tenant health and had no new kernel GPU OOM.

**Interpretation.** The produced tokens and actions differed despite fixed
seeds. These times do not isolate throughput gains, nor do one-step scores
establish six-case quality. The current Gemma campaign uses MTP0; queued Qwen
uses MTP3. Historical Qwen settings must be verified separately rather than
inferred from its replacement recipe. FP8 uncalibrated-scale warnings were
recorded; their impact on quality was not measured.

Evidence: `.scratch/model-runs/pgx-three-model-20261009/vllm/mtp-short-comparison.json`.

## R10 - Score completeness, fallbacks, and provenance

**Reporting control, 2026-10-12.** A complete six-case score requires all cases
and trajectory/score replay verification. Provider failure and model-turn
fallback are different outcomes. A fallback action remains part of the
simulation result and must be reported alongside the score.

At cutoff, Gemma non-thinking completed six cases with a verified score of
21.478460424376575, 197 fallback turns out of 2,880 (6.84%), and zero provider
HTTP errors. Thinking remained active and Qwen was queued. Earlier failed
attempts and one-step smokes have no comparable six-case score.

**Provenance defect to avoid.** Per-job request policies contain inherited
`HEAD` fields alongside later `source_code_commit` and source-file hashes.
Do not quote the inherited field alone as the executed source revision. Use
the frozen source files and campaign manifest, and validate their hashes.

Current campaign source commit: `14a97f91649912ff376892211652980b5467e363`.
Frozen suite SHA-256:
`d3308f1f5abca314dc9f8c4a386e3bcc08533ac46eaf9bfd2af852ea9a1137c8`.
Gemma checkpoint revision: `a19cfe00be84568a6867111c9a68c9c44fdcffe6`.
Serving image: `eugr/spark-vllm@sha256:761a1542d89cccdf770580563057c0bc686c0b495fa309d7eaa5c9676328ff7f`.
Recorded vLLM build: `0.23.1rc1.dev790+g85d955ed5.d20260705`.

Evidence: `.scratch/model-runs/gemma-qwen-256k-20261011/campaign.json`,
`source/`, `suite.yaml`, `gemma-thinking/request-policy.json`, and
`gemma-non-thinking/verification.json`. General artifact contracts:
[Artifacts and failures](artifacts-and-failures.md).

## R11 - Shared sampling defaults differ from model guidance

**Configuration observation, 2026-10-12.** The frozen benchmark defaults send
temperature 1.0, top-p 0.95, and top-k 20. Google's official Gemma 4 guidance
recommends temperature 1.0, top-p 0.95, and top-k 64. This discrepancy does not
establish the cause of repetition; no top-k ablation has been run. Shared
decoding settings and model-recommended settings are distinct evaluation
policies. State which policy the paper uses and keep its baseline fixed.

Evidence: [config.py](../src/mam_bench/config.py),
[runtime.py](../src/mam_bench/runtime.py), and frozen equivalents in the
campaign's `source/`; [Google model guidance](https://huggingface.co/google/gemma-4-26B-A4B-it#best-practices).

## Repetition controls: verified interface, untested efficacy

Documentation and installed-source check: 2026-10-12. vLLM exposes the
following controls. Defaults in this table are library defaults; effective
request settings may inherit model generation configuration.

| Setting | Neutral default | Behavior |
| --- | --- | --- |
| `repetition_penalty` | 1.0 | Values above 1 discourage tokens already present in the prompt or generated output |
| `frequency_penalty` | 0.0 | Positive values penalize output tokens in proportion to their previous count |
| `presence_penalty` | 0.0 | Positive values penalize output tokens that have appeared at least once |
| `repetition_detection` | Unset | Stops generation when repeated token N-grams meet configured thresholds |
| `thinking_token_budget` | Unset | Additional reasoning limit; requires compatible reasoning boundaries |

Presence and frequency values must be in [-2, 2]; repetition penalty must be
finite and greater than zero. Presence/frequency are ordinary Chat Completions
fields; repetition penalty and repetition detection are vLLM extensions, sent
in HTTP request JSON or an OpenAI client's `extra_body`.
Sources: [SamplingParams](https://docs.vllm.ai/en/latest/api/vllm/sampling_params/)
and [OpenAI-compatible serving](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).

**Implementation inference.** These penalties operate on token logits before
selection, without a reasoning/tool-field exemption in the documented
implementation. They can affect both looping prose and legitimate repeated
tool syntax or vocabulary. They do not detect whether a repeated sentence is
useful reasoning. See the official
[sampler](https://docs.vllm.ai/en/latest/api/vllm/v1/sample/sampler/)
and [penalty implementation](https://docs.vllm.ai/en/latest/api/vllm/model_executor/layers/utils/).

**Installed-build evidence.** Read-only inspection of the active container
confirmed penalty fields/defaults, the N-gram detector schema, and the chat
API forwarding `repetition_detection` to sampling parameters. Detector fields
are `min_pattern_size`, `max_pattern_size`, and `min_count`. This establishes
interface availability, not end-to-end success. A detector stops output; it
does not force completion of a valid tool call. The thinking-budget interface
also exists, but Gemma-specific boundary integration remains unverified.
See [thinking budget requirements](https://docs.vllm.ai/en/latest/features/reasoning_outputs/#thinking-budget-control).

The current recipe/client has no explicit penalty or repetition-detector
override. Its effective checkpoint defaults have not been fully audited;
absence of a client setting alone is insufficient to prove neutral decoding.
Google's model card does not prescribe a nonneutral penalty. No control value
or threshold is recommended as a proven Gemma fix by this log.

Local references under `.scratch/model-runs/gemma-qwen-256k-20261011/`:
`repetition-penalty-references.md`, `installed-sampling-params-20261012.py`,
`installed-chat-controls-20261012.txt`, and
`installed-reasoning-controls-20261012.txt`. The references note links captured
official docs. Those docs use `latest`; installed-build observations are
separately identified. A new control should be tested as a separate condition
rather than silently changing a running benchmark.

## Follow-up - Deferred detector rollout and Tailscale interruption

**2026-10-12.** The user authorized repetition detection for subsequent local
vLLM runs, then explicitly requested that the current Gemma attempt finish
before enabling it. No detector, client-code, serving-command, or active-job
configuration change was applied. Threshold selection and live validation
remain pending.

The verified chat interface exposes `repetition_detection` per request and
passes it directly into sampling parameters. The documented server-default
merge covers a separate list of ordinary sampling fields; it does not include
this detector. Do not assume `--override-generation-config` enables detection
for every client. A shared benchmark client can apply the same detector
configuration to every future supported vLLM request, including summaries,
while recording it in run provenance. Other clients must also send it.
Sources: [chat request conversion](https://docs.vllm.ai/en/latest/api/vllm/entrypoints/openai/chat_completion/protocol/)
and [startup generation defaults](https://docs.vllm.ai/en/latest/configuration/engine_args/#--override-generation-config).

**New operational incident.** The Linux Tailscale node key expired at
2026-10-12 16:33:59 UTC (11:33:59 CDT). The remote guard's last retained sample
was 16:33:58 UTC, followed by a broken-pipe disconnect. The benchmark was
cancelled at 16:36:51 UTC; the subsequent remote model-stop command failed
with SSH exit 255. Live, unsandboxed checks showed the local Tailscale service
and Pitchfork supervisor running, with Tailscale in `NeedsLogin`. An initial
sandboxed supervisor/service check could not access the sockets and should
not be interpreted as a stopped service.

The retained thinking attempt completed cases 1-3 and 27 of 30 cycles in
case 4; cases 5-6 had not started. It has no final six-case score. The remote
inference server's cleanup is unconfirmed while access is unavailable.
Restoring authenticated access and checking remote ownership are prerequisites
to recovery; the current artifact contract has no partial-run resume support.
This incident is independent of repetition detection, which was never enabled.

Evidence: current campaign `campaign.json`; thinking `benchmark.log`,
`exit-code`, `model-stop.log`, `guard-renewal-001.err`, and
`guard-renewal-001.jsonl`; retained case-4 `controlled.jsonl`. Paths share
`.scratch/model-runs/gemma-qwen-256k-20261011/`. Authentication status was
checked read-only with `tailscale status --json` outside the execution sandbox.

## Follow-up: repetition-detected rerun, 12 October 2026

The user reauthenticated Tailscale and authorized a fresh sequence: Gemma
thinking, Qwen thinking, then Qwen non-thinking. Authenticated PGX access was
verified live. The idle Gemma server from the interrupted attempt was stopped
after retaining its server log; ASR and TTS remained healthy. The previous
attempt and its artifacts remain separate. Gemma non-thinking's completed
result is a historical baseline without repetition detection; it is not a
matched detector-enabled result.

New campaign: `.scratch/model-runs/gemma-qwen-repetition-20261012/`.
Pitchfork daemon: `mam-gemma-qwen-repetition-20261012/campaign`.
Launch observed at 2026-10-12 17:14 UTC. The worker runs the three modes
sequentially, verifies each complete six-case run, and regenerates
`.scratch/playground/pages/gemma-qwen-repetition-analysis.html` every minute.
Fresh Gemma preflight passed and case 1 started at 17:16 UTC. At 17:17 UTC,
native server metrics showed eight requests running and eight waiting, while
Qwen's two modes remained queued. This confirms active admission and queuing;
it does not establish average GPU utilization or final throughput.
Its health guard requires at least 20 GiB available RAM and healthy ASR/TTS,
records restart counts and OOM events, and stops the scoped inference workload
on a failed guard. This is supervised local orchestration over Tailscale;
connectivity loss remains a possible interruption rather than a solved issue.

All three modes send the same native vLLM `repetition_detection` request
parameter: minimum pattern size 16 tokens, maximum 128, minimum count 8.
It detects consecutive repeated token patterns and stops the completion.
These are conservative initial thresholds, not an empirically optimal policy.
Gemma serves 262,144 tokens with eight active server slots; Qwen serves
262,144 with 16 slots. Both use 16 benchmark clients and a 32,768 output cap.
Gemma has MTP disabled; Qwen retains MTP3. Qwen thinking sends
`enable_thinking: true` and reasoning effort `medium`; Qwen non-thinking sends
`enable_thinking: false` and omits reasoning effort. Detector settings apply
to both agent requests and memory-summary requests from the common model
factory. Other runs must explicitly opt in through their model YAML.

The six-case suite remains frozen at SHA-256
`d3308f1f5abca314dc9f8c4a386e3bcc08533ac46eaf9bfd2af852ea9a1137c8`.
Of the 23 frozen source files, only config and runtime changed to configure
the detector and accept its vendor-specific finish reason. The campaign
manifest records every source hash, model revision, image digest, recipe,
client policy and validation receipt. No simulation or scheduling change is
part of this intervention.

### R12: native repetition stop rejected by the strict OpenAI adapter

Status: reproduced and corrected at the client boundary; benchmark effect
remains to be measured.

The synthetic engine probe forced one allowed token with EOS ignored. Without
the detector, Gemma emitted all 256 requested tokens in 9.053 seconds. With
the detector it stopped at 128 tokens in 4.553 seconds and returned
`finish_reason: repetition`, `stop_reason: repetition_detected`. This establishes
that the native engine implements early stopping; it says nothing about
natural reasoning quality or benchmark-level speedup.

The strict OpenAI-compatible client rejected that vendor finish reason before
the benchmark could apply its normal incomplete-output handling. The adapter
now normalizes this known reason to `length` for library validation and retains
the original `repetition` reason and stop cause in provider details. Existing
bounded retries and fallbacks remain in force. Unknown finish reasons still
undergo strict validation. A detector stop can therefore enter the existing
incomplete-output path and may receive its legacy `output_token_limit` fallback
label; raw provider causes and `repetition-stops.jsonl` distinguish actual
detector stops from exhaustion of the output budget.

Validation: the real Gemma model client accepted the forced early stop and
then produced a valid `move(destination: null)` terminal tool call. All three
frozen model factories sent the intended detector and thinking parameters.
The 92 native tests passed; type checking reported zero errors and warnings;
Ruff, formatting and diff whitespace checks passed. The pinned Qwen image
contains the detector API and engine implementation. Each newly started
engine must pass the same live stop/tool check before its benchmark starts;
Qwen's live validation is pending until its turn in the sequence.

Evidence under the new campaign directory: `detector-probe-receipt.json`,
`probe-baseline-response.json`, `probe-detected-response.json`,
`installed-detector-engine.py`, `qwen-image-detector-support.txt`,
`gemma-thinking/engine-validation.json`, `validation.json`, `native-tests.log`,
`pyright.log`, and `old-model-cleanup.log`. Per-job `repetition-stops.jsonl`
records response IDs, native stop causes, usage and phase. Entries marked
`preflight` are synthetic validation requests and must be excluded from
benchmark metrics. The earlier unmarked prelaunch synthetic response
`chatcmpl-8c4bcd6af5b4a1e1` must also be excluded.

Paper implication: report this as an explicit decoding-policy intervention.
Compare native stop counts, token volume, incomplete-output fallbacks, wall
time and utilization, alongside verified scores. A successful synthetic
stop and a valid tool call do not establish reduced natural looping, improved
decisions, or preserved scores. The older Gemma non-thinking baseline differs
in detector policy and cannot isolate thinking alone.

## Follow-up: retrospective detector coverage, 12 October 2026

The user asked whether the configured 16-128-token, eight-repeat detector
would catch the Gemma thinking loops already reported. A read-only replay
tested all 75 previously flagged Gemma traces from cases 1-3 using the pinned
Gemma serving tokenizer and the installed engine's detector functions. It
caught 73/75 (97.3%). First matching prefixes contained 272-20,470 retained
thinking tokens, with median 2,216; all original responses reported 32,768
output tokens. The response repeating `Actually, I'll use
move(destination=null)` 2,112 times matched at retained token 1,355.

The two misses contain longer exact cycles, with periods 154 and 327 tokens,
outside the configured maximum of 128. An offline maximum-512 diagnostic
caught them at retained tokens 4,314 and 24,379 respectively; each hit and
the immediately preceding non-hit were checked with the installed engine
function. Merely lowering minimum size to 1 or count to 4 while retaining
maximum 128 did not catch either miss. A maximum of 512 would cover all 75
in this retained-text replay; it was not applied to the running campaign.

Twenty caught traces have a primitive period shorter than 16 tokens. They
still match allowed larger blocks comprising multiple primitive repetitions.
Consequently, a minimum of 16 does not exclude every loop built from a shorter
phrase; it requires enough consecutive repetitions to form eight allowed
blocks. This corrects an overly broad interpretation of the minimum size.

Across these 75 traces, 2,078,933 of 2,457,360 retokenized thinking tokens occur
after the replay's first detection point (84.6%). This is a hypothetical
retained-token reduction, excluding subsequent retries/fallbacks; it is not
measured wall-time savings or whole-benchmark token savings. The original
generated token IDs were not retained. Retokenizing parsed thinking omits
parser-hidden/control tokens and may use different segmentation, so reported
prefix positions are approximate original-generation stop locations. The
screen selected severe exact repeats and does not cover every semantic loop
or establish false-positive rates on other responses.

At the live observation cutoff (last stop 2026-10-12 19:14:10 UTC), the new
Gemma thinking run had 15 native benchmark repetition stops, with reported
output counts 1,043-6,609 and median 2,776. Preflight probes were excluded.
This independently confirms that the detector catches natural benchmark
generation, without proving score improvements or total runtime savings.

Evidence: `.scratch/model-runs/gemma-qwen-repetition-20261012/` contains
`replay-historical-loops.py`, `historical-loop-replay/report.json`, cached
retokenizations under `historical-loop-replay/tokens/`,
`historical-loop-replay/missed-diagnostics.json`, and the native per-job
`repetition-stops.jsonl`. The optimized prefix search was checked against the
original engine functions on deterministic synthetic examples and every
detected real hit boundary. Report checksums record the input audit and
installed engine source. No inference was generated for the retrospective
replay and no live run settings changed.

## R13 - Local Tailscale restart aborts a healthy PGX run

Observed 2026-10-12 19:57:48-58 UTC (14:57 CDT). The detector-enabled Gemma
thinking campaign stopped in case 1 after 18/30 controlled steps, with no
completed case or six-case score. Its two Qwen modes never started. Eighteen
native benchmark repetition stops were retained. This is an interrupted
attempt, distinct from the earlier node-key-expiry incident.

The final remote guard sample at 19:57:48 UTC reported 42.88 GiB available,
ASR/TTS HTTP 200, unchanged restart counts [1, 3], and no OOM marker. Local
systemd then stopped and started `tailscaled.service`; its new main process
started at 19:57:49 UTC. The guard's SSH stderr recorded
`client_loop: send disconnect: Broken pipe`. The campaign asserted on the
nonzero guard-process exit and sent SIGINT to the benchmark, explaining its
terminal `KeyboardInterrupt`. Scoped model cleanup succeeded.

The recovered Gemma server log shows it continued generating at 133.7 tokens/s
with seven active requests at 19:57:51 UTC. Thus the retained evidence shows
a transport/supervision interruption, without a PGX memory or model-engine
failure preceding it. The service journal shows an orderly local restart.
A subsequent package-log check established the trigger: an APT transaction
invoked through `aptdaemon role='role-commit-packages'` ran from 19:57:06 to
19:58:09 UTC. It upgraded Tailscale from 1.102.4 to 1.104.1, unpacking at
19:57:45-46, configuring at 19:57:47 and recording the new version installed
at 19:57:49. The upgrade's service restart aligns with the SSH disconnection.
The same transaction upgraded Docker/containerd on the local machine. The
logs establish the package-update mechanism. The user subsequently confirmed
that they initiated the Ubuntu update; that attribution comes from their
statement, rather than the package logs.
Local node authentication remained valid, with key expiry 2027-04-10, and
PGX was reachable afterward with 16 days uptime and the inference container
removed. ASR/TTS containers remained up without restart.

Cause: remote health-guard lifetime was coupled to one foreground SSH
connection. A local transport restart killed that connection, and the worker
treated its exit as a generic PGX guard failure. Failing closed preserves
service safety, but this design makes long runs fragile and obscures the
difference between an actual remote health violation and a lost monitor.
No automatic retry or resumed run was performed during diagnosis.

Required recovery design: keep the safety guard supervised on PGX, retaining
its stop action and durable health receipt independently of the watching
SSH session. Reconnecting observers must read guard state and freshness;
missing or stale health data must not silently disable the guard. Record
transport failure separately from resource/service failure. Validate a
controlled observer disconnect against a disposable job before another full
campaign. Partial-case resume remains unsupported by the current runner.

Evidence under `.scratch/model-runs/gemma-qwen-repetition-20261012/`:
`tailscale-restart-lifecycle.tsv`, `gemma-thinking/guard.jsonl`, `guard.err`,
`tailscale-update-history.txt`, `tailscale-update-dpkg.txt`,
`server.log`, `benchmark.log`, `model-stop.log`, `exit-code`, `campaign.json`,
and the retained case-1 `controlled.jsonl`. The paper must not interpret this
interruption as model latency, repetition-detector failure, or a completed
score.

**Subsequent user decision:** the user rejected the external guard as
unnecessary. Future launches will omit it, rather than implement the
PGX-supervised replacement proposed above. A direct source check found no
`startup-watch`, guard-process polling or PGX health-stop condition in
`src/`, `main.py`, tests or package configuration. The guard belongs only to
the scratch campaign launcher and remote scratch monitor. Historical scripts
and receipts remain preserved for this interrupted attempt. This decision
does not remove the separate repetition-detection request settings.

## Template for subsequent entries

```text
ID / title:
First observed UTC / last updated UTC:
Status: observed | reproduced | mitigated | verified fix | open
Experimental condition and provenance:
Symptom and quantitative impact:
Evidence paths, response IDs, and checksums:
Cause: verified mechanism or competing hypotheses
Intervention: exact change, date, and authorization
Validation: checks passed; checks absent or failed
Remaining uncertainty:
Paper implication and next experiment:
```

## Evidence checksums at this cutoff

These files are under `.scratch/model-runs/gemma-qwen-256k-20261011/`.
Checksums identify the analyzed snapshots if later audits regenerate them.

```text
repetition-audit.json
bd0e21dc7adf4648c744263500ca02460bc1493bce838bc6259ea693ce17c5f9

runtime-comparison.json
3f16017bac25a1f11ba896df90a3cb11f3b6581903ed89a26187cee3eb8039f3

runtime-comparison-cases-2-3.json
f07643e471deec8b768cb022825c975f9b58d4f213e3cdde8d759c84299c5a98

gemma-non-thinking/verification.json
bc243f9bbb8c2cb9c8336aa4dd282491d510a7f4260e8682ec0fb4be3883f783

installed-sampling-params-20261012.py
41a8f63edbac1002f3c318fdf6e295b088ec389c310baafea15f0684720808cb
```
