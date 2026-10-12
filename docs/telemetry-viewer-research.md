# OpenTelemetry viewer shortlist

Checked official documentation and repositories on 2026-10-12. This is the initial viewer comparison; the later Logfire selection and implemented collection slice are documented in [logfire-design.md](logfire-design.md). Grafana and Jaeger are excluded at the user's request.

## Recommendation

For the shortest local setup, evaluate standalone Aspire Dashboard with persistence enabled. For a persistent self-hosted observability backend, compare OpenObserve's single-node deployment with SigNoz's fuller stack. Uptrace is a credible alternative but its Community edition's fixed 14-day retention is a material concern for research history. `otel-tui` is a terminal companion, not the primary archive. Hosted Logfire is being investigated separately by the parent agent.

This is a fit assessment from documentation, not an installed comparison or benchmark of these products. Before choosing, send one representative trace, a correlated structured log, a cumulative counter and a latency histogram to the chosen version and verify the actual queries and history behavior.

## Comparison

| Candidate | Traces, logs and metrics | Deployment and history | Fit and principal limitation |
| --- | --- | --- | --- |
| Aspire Dashboard | Native OTLP/gRPC and OTLP/HTTP (protobuf/JSON); structured logs, trace viewer, metrics charts | One standalone container or CLI process. SQLite persistence modes `Run` and `Resume`; default standalone mode `None` discards data | Smallest browser viewer; development/short-term diagnostics, bounded eviction, schema compatibility can prevent old-run viewing |
| OpenObserve | Unified logs, metrics and traces; OTLP and Prometheus ingestion; SQL and PromQL | One binary/container for single-node, persistent mounted data directory; larger HA deployments use additional infrastructure | Smallest full self-hosted backend in this shortlist; validate metric temporality and exact OSS/Enterprise feature set before selection |
| SigNoz | Native OTLP traces/logs/metrics, linked trace/log navigation, query builder, PromQL and ClickHouse SQL | Current supported Compose install uses Foundry; SigNoz, collector, ClickHouse, ClickHouse Keeper and PostgreSQL; minimum 4GB Docker RAM | Fuller integrated observability UI; substantially more services than Aspire/OpenObserve |
| Uptrace | Native OTLP/gRPC and HTTP for all three; log/trace context navigation; trace query language and PromQL-inspired metric exploration | ClickHouse, PostgreSQL, Redis, Uptrace; example also includes collector and optional MailHog. 2+ cores, 4GB+ RAM, 10GB+ initial disk; named persistent volumes | Good integrated UI; current Community retention fixed at 14 days, configurable retention requires Premium |
| otel-tui | OTLP traces/metrics/logs plus other receivers, trace-linked logs | One local terminal viewer; rotating buffer of 1,000 root spans/logs, JSON import | Useful quick diagnostic companion; not a full research-history backend |

## Aspire: recent persistence changes matter

[Standalone docs](https://aspire.dev/dashboard/standalone/) explicitly support arbitrary OTel-enabled apps without an AppHost, and accept all three signals via gRPC or HTTP. A .NET application rewrite is not needed. The standalone container provides a browser UI and ingestion endpoints. Console/process control features require a resource service; OTLP alone supplies telemetry, not process management.

[Persistence docs](https://aspire.dev/dashboard/data-persistence/) describe SQLite storage, `Run` mode with historical run selection and pinning, and `Resume` mode reusing history after restart. Calling Aspire in-memory-only would be outdated. Default standalone `None` uses a temporary database; persistence requires explicit configuration and a container volume. It is still intended for short-term diagnostics: telemetry limits evict oldest records, only 10 unpinned runs are retained by default, incompatible schema versions cannot be viewed, and `Resume` replaces incompatible databases. It does not provide replication/backups or a disk quota.

[Configuration](https://aspire.dev/dashboard/configuration/) exposes retention limits and the data directory. [Explore docs](https://aspire.dev/dashboard/explore/) are the UI reference for logs, traces and metrics. Detailed cumulative/delta instrument support was not established from the pages inspected; validate the intended Python export and chart semantics rather than infer support from the OTLP protocol alone. [Repository license](https://raw.githubusercontent.com/dotnet/aspire/main/LICENSE.TXT): MIT.

## OpenObserve: simple persistent backend, metric check needed

[Quickstart](https://openobserve.ai/docs/quickstart/) documents single-node binary/container deployment with a persistent data mount and cloud/self-hosted alternatives. Be careful selecting an image: the current quickstart's example uses an Enterprise image, while it lists separate OSS and Enterprise images. [Official index](https://openobserve.ai/llms.txt) states SQL for logs/traces, PromQL for metrics, and single-binary deployment. These describe capabilities rather than measured resource needs; no credible minimum CPU/RAM was established in this pass.

[Metrics ingestion](https://openobserve.ai/docs/ingestion/metrics/) explicitly lists OTLP/OTel Collector and Prometheus ingestion, linking to [OTLP ingestion](https://openobserve.ai/docs/ingestion/logs/otlp/). Delta compatibility needs confirmation: the [primary repository issue asking for delta temporality support](https://github.com/openobserve/openobserve/issues/4214) remains a relevant unresolved lead in the fetched issue content, with an assignee added in September 2026. This alone does not prove the latest code rejects delta; it means do not promise delta compatibility without testing the selected release. A cumulative metric baseline is the conservative evaluation path.

[Repository license](https://raw.githubusercontent.com/openobserve/openobserve/main/LICENSE): AGPL-3.0. The official index distinguishes commercial Enterprise capabilities, including access control/SSO/AI features. Ordinary self-hosted signal storage is available in OSS, but a UI screenshot or Enterprise demo should not be assumed to show the OSS feature set. Confirm exact trace-to-log navigation in the chosen version; generic unified-signal marketing does not prove the correlation workflow.

## SigNoz: broad integrated UI, larger stack

[Official Docker guide](https://signoz.io/docs/install/docker/) now uses Foundry to generate/manage Compose; bundled legacy `deploy/` files were deprecated at v0.130.0. It lists a 4GB minimum and the stack components above. The [metrics overview](https://signoz.io/docs/metrics-management/overview/) documents metric exploration, query builder, PromQL, ClickHouse SQL and dashboards. [Log/trace correlation](https://signoz.io/docs/traces-management/guides/correlate-traces-and-logs/) navigates both ways when records carry trace/span IDs; the application must supply/export those fields.

[Metric types and aggregation](https://signoz.io/docs/metrics-management/types-and-aggregation/) documents cumulative and delta counters/histograms. Exponential histograms require delta and collector enablement; they are currently self-hosted-only. PromQL queries do not read delta series, so choose the query builder for delta metrics or preserve cumulative metrics when relying on PromQL. This is a concrete compatibility constraint, not a generic OTel limitation.

[Repository license](https://raw.githubusercontent.com/SigNoz/signoz/main/LICENSE): MIT for core, separate Enterprise license for `ee/` and `cmd/enterprise/`. History lives in persistent databases; configure retention/volumes and verify a restart before treating it as the research archive. This pass did not verify current default retention.

## Uptrace: useful UI, retention tradeoff

[Docker guide](https://uptrace.dev/get/hosted/docker) documents resource minimums, named persistent volumes, databases and optional example components. It explicitly says Community retention is 14 days and cannot be changed; Premium enables custom project retention. This is decisive if several months of benchmark history must stay queryable.

[Direct Python OTLP guide](https://uptrace.dev/get/opentelemetry-python/otlp) shows exporters for all three signals, including active-span log correlation. [Direct Go OTLP guide](https://uptrace.dev/get/opentelemetry-go/otlp) selects delta for counters/observable counters/histograms and cumulative for other instruments; the Python example does not explicitly select temporality. Therefore both forms appear in official configurations, but the exact supported instrument matrix was not verified. [Metrics UI](https://uptrace.dev/features/metrics) supports dimensions, filters and expressions; [logs UI](https://uptrace.dev/features/logs) includes linked traces/context. [Repository license](https://raw.githubusercontent.com/uptrace/uptrace/master/LICENSE): AGPL-3.0; current Premium feature gates exist alongside the community edition.

## otel-tui

[Project README](https://raw.githubusercontent.com/ymtdzzz/otel-tui/main/README.md) documents three-signal OTLP receivers, log/trace linking, Prometheus receiving and JSON-file import. The 1,000-root-span/log buffer rotates, and several advanced metric visualization features remain TODOs. This is a short-lived debug viewer, not the durable integrated backend requested here.

## Selection checkpoints

1. Is the priority a small live debug UI or a long-term observability database?
2. Can the chosen UI query benchmark run/case IDs, display a nested agent/model/tool trace, and jump to the log that explains a retry?
3. Do cumulative counters and histograms chart correctly, including process restarts and no-traffic intervals?
4. Does a restart preserve history; what expires or becomes incompatible during upgrades?
5. Does the chosen edition provide the demonstrated features?

No services were installed or started to answer these questions. Implementation should follow viewer choice.
