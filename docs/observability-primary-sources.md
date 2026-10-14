# MAM-Bench observability options

Initial research on 2026-10-12, before package installation or runtime changes. Public documentation checked with `ketch`; installed package sources checked for version-specific boundaries. See [logfire-design.md](logfire-design.md) for the later selection and implemented collection slice.

Direction update: Joe selected OpenTelemetry as the instrumentation basis and excludes Grafana and Jaeger. This note records the initial options research; its JSONL-first recommendation is superseded. Viewer selection comes before full design; compare both hosted and self-hosted options in `telemetry-viewers.md`.

## Recommendation

Start with durable local JSONL events for benchmark lifecycle and logical model requests, plus passive samples from the existing vLLM `/metrics` endpoint and supported PGX hardware counters. Build the live view and retrospective queries from those files. This is the smallest setup that can answer whether time went into scheduler wait, provider work, compaction, tools, retries, or inactivity. Add native Pydantic AI OpenTelemetry later if the nested trace view is worth operating another service.

Monitoring failures should appear as missing samples or explicit telemetry errors, without stopping the benchmark. This is observability, with no external guard or abort coupling.

## Options

| Option | Useful for | Cost and limits |
| --- | --- | --- |
| Local JSONL and artifact queries | Progress, exact research denominators, fallback/repetition causes, scheduler and request timing, reproducible comparisons | Stdlib suffices. Requires deliberate domain fields and a small live/status query. Files are not a distributed tracing UI. |
| Native Pydantic AI OTel to Logfire | Nested model/tool timelines and querying agent behavior | Native instrumentation exists; Logfire backend requires an account, SDK installation, data export and retention decisions. |
| Native OTel to a local viewer | Same timing structure without a hosted backend | SDK/exporter and viewer still need installation/configuration. `otel-tui` is light; Jaeger provides a browser UI. Configure durable storage for research retention. |
| Prometheus/Grafana | Engine saturation and throughput trends over many runs | vLLM already publishes metrics. Prometheus storage and Grafana dashboards add services; useful after flat-file samples become inconvenient. |

## Apply the guide selectively

[Logging Sucks](https://loggingsucks.com/) argues for consistent IDs, structured queryable context, one rich completion record per request, and outcome-aware sampling. Its examples concern web-service traffic. For a benchmark, use a rich completion record at the turn or logical-request boundary, while also retaining starts, attempt failures, completed cycles, and occasional progress records. A completion-only record leaves a currently stuck request invisible; process kill or reboot can prevent it altogether. Keep small lifecycle and request records in full: sampling successful turns changes the denominator needed to compare completion rates, token use, and scheduler behavior. Large prompt/thinking bodies can remain in existing artifacts, referenced by path/ID rather than copied into every event. These are project recommendations, not claims made by the guide.

## Native tracing: useful but not complete coverage

[Pydantic AI's instrumentation guide](https://pydantic.dev/docs/ai/integrations/logfire/) documents agent, model-request and tool spans, any OTel-compatible backend, and additional HTTP client instrumentation. Installed `pydantic-ai-slim` is 2.38.0 and `pydantic-ai-harness` is 0.29.0. `uv.lock` contains OTel API and `logfire-api`, but no OTel SDK/exporter or full `logfire` SDK. The API alone does not provide an exporting setup.

Installed `models/instrumented.py:353` wraps one logical `Model.request`; it does not separately expose every wire attempt inside that request. Installed OpenAI `_base_client.py:1675` loops across SDK retries. The [official retry map](https://pydantic.dev/docs/ai/core-concepts/retries/) likewise distinguishes provider/transport retries that the agent does not see. Use HTTP client or transport instrumentation if diagnosing those attempts; label logical requests and HTTP attempts distinctly.

Installed `InstrumentationSettings` defaults to capturing content and full model-request parameters. Disable content capture when existing artifacts suffice; otherwise long histories and repeated thinking traces duplicate a lot of data. Keep raw secrets/authorization headers out of telemetry rather than copying the guide's `capture_all=True` demonstration.

Installed Harness `compaction/_summarizing_compaction.py` creates a nested `Agent` for its standard summarizer. MAM-Bench's `AgentSessions._compaction()` uses that `SummarizingCompaction`, so global agent instrumentation can cover this path. The project's existing `_RequestLogging` capability is attached to the outer decision agent and is not automatically attached to the nested summary agent. A shared-model hook is another way to cover both paths. Explicitly carry run/turn context and `purpose=agent_turn|compaction`, retain before/after context size, and verify that instrumentation neither misses nor double-counts summary requests.

Local viewers: [otel-tui's README](https://raw.githubusercontent.com/ymtdzzz/otel-tui/main/README.md) describes OTLP receivers, JSON import, and a rotating buffer of 1,000 service-root spans/logs. Treat it as a debugging viewer rather than the authoritative full-run archive. [Jaeger's quick start](https://www.jaegertracing.io/docs/latest/getting-started/) combines collector/query/UI but defaults to transient in-memory storage. Keep long-term local artifacts regardless of viewer.

## Engine metrics and request correlation

[vLLM metrics](https://docs.vllm.ai/en/latest/design/metrics/) expose running/waiting requests, KV cache usage, prefix-cache counters, prompt/generated token counters, queue/prefill/decode latency and request outcomes. Compute aggregate TPS from token-counter deltas over elapsed time, recognizing restarts/resets. Do not sum overlapping request durations to estimate wall time. Engine-wide metrics include unrelated requests unless the server is dedicated to the benchmark. Check names and available fields against each pinned container; current online docs can exceed the deployed version.

The [OpenAI-compatible server docs](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/) document `X-Request-Id` and `--enable-request-id-headers`. Record the benchmark's logical request ID, attempt number, response-body completion ID, and HTTP header request ID separately. Installed Pydantic AI `models/openai.py:1291` preserves `response.id` as `provider_response_id`; it does not preserve the SDK's `_request_id` header in that mapping. Getting the header requires HTTP-level observation. Do not claim distributed trace continuity merely because client and server both emit OTel: header propagation and receiver support need verification in the actual pinned image.

[vLLM observability config](https://docs.vllm.ai/en/latest/api/vllm/config/observability/) warns that detailed model/worker traces can use costly or blocking operations. Leave them for focused diagnostic runs; record their activation as a benchmark intervention. Ordinary metrics collection is a more appropriate baseline.

Request IDs and turn IDs are excellent event/trace fields. They are poor Prometheus labels: [Prometheus naming guidance](https://prometheus.io/docs/practices/naming/) warns that each distinct label combination creates a series. Use bounded labels such as operation, model, simulation, outcome and phase; keep arbitrary IDs in events.

## PGX utilization caveats

[NVIDIA defines GPU utilization](https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization) as the share of a recent sampling interval with at least one kernel executing. It is not fraction of theoretical FLOPs or achieved model throughput. For scheduler comparisons, combine GPU-active fraction with engine active/queued requests, tokens/s, scheduler wait and idle gaps. Include time-weighted averages, sample availability and benchmark phase. Sparse periodic snapshots can miss short gaps; other GPU workloads can inflate the device-wide reading.

[DGX Spark known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) confirms unified CPU/GPU DRAM, unsupported framebuffer memory totals in `nvidia-smi`, and limits of `cudaMemGetInfo` when reclaimable/swap memory exists. Collect system `MemAvailable` and swap alongside GPU telemetry, and report unsupported counters as unavailable rather than zero. This does not prove utilization itself is unsupported; probe the deployed driver. If coarse counters cannot distinguish a scheduler issue, a short Nsight Systems diagnostic can resolve kernel/gap timing, with profiler overhead recorded separately from scored runs.

## First-stage acceptance questions

- Which case/cycle and agents are active; when was the last completed action and request?
- For a turn, how much elapsed time was scheduler wait, model work, summary work and tools, and what were its retries/fallback/finish reasons?
- Were vLLM requests running or queued during client-side idle gaps, and what were tokens/s and KV use?
- Are missing hardware samples distinguished from measured zero utilization?
- Can failed, cancelled and killed runs be inspected without pretending they completed?

No broad vendor survey or monitoring stack deployment is needed to answer those first.
