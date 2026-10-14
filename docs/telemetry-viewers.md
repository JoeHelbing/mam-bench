# Choosing an OpenTelemetry viewer

Status: initial viewer comparison, checked 2026-10-12. Joe subsequently selected
Logfire; see [logfire-design.md](logfire-design.md) for the implemented collection
slice and remaining domain/server instrumentation.

Joe selected OpenTelemetry metrics, traces and logs as the instrumentation
basis, excluding Grafana and Jaeger. Compare both hosted and self-hosted
options before developing the full design. Work stays on `codex/observability`
off upstream `dev`; the existing benchmark artifacts remain intact.

## Recommendation

Shortlist Logfire for a managed service, OpenObserve for a compact persistent
self-hosted backend, and Aspire Dashboard for the smallest local debugging UI.
SigNoz is a strong alternative when a fuller integrated observability interface
justifies operating a larger database stack.

This is a fit assessment from current documentation, not a tested product
comparison. Native OTLP ingestion does not guarantee identical metric query
semantics, retention, or GenAI-specific UI support.

## Choices

| Viewer | Hosting | What you get | Main tradeoff |
| --- | --- | --- | --- |
| Logfire | Managed cloud; Enterprise self-hosting also exists | Live traces/logs, native Pydantic AI and LLM inspection, SQL exploration, metric dashboards | Account, usage/retention costs and telemetry export; Enterprise self-hosting is substantially heavier |
| Aspire Dashboard | Standalone local container/process | Structured logs, nested trace view and metric charts for arbitrary OTel-enabled apps; SQLite run history | Development/short-term diagnostics; limits evict data, and persistence/upgrade behavior needs explicit configuration |
| OpenObserve | Self-hosted single node or cloud | Logs, traces and metrics; SQL and PromQL; persistent single-binary deployment | Validate metric temporality, OSS feature availability and trace/log navigation in the chosen version |
| SigNoz | Self-hosted or cloud | Integrated metrics dashboards, trace inspection, correlated logs and multiple query modes | Larger stack: application, collector and database services |
| Uptrace | Self-hosted or cloud | Trace/log navigation and metric exploration | Current Community edition fixes retention at 14 days; configurable retention requires Premium |

For terminal debugging, `otel-tui` is another useful receiver/viewer. Its rotating
buffer and incomplete advanced metric views make it a companion rather than the
primary retained observability database.

## Logfire evidence

[Pydantic AI's official instrumentation guide](https://pydantic.dev/docs/ai/integrations/logfire/)
documents native instrumentation, including `Agent.instrument_all()` with a
standard OTel SDK and exporter. A Logfire backend or SDK is not required to emit
those spans. The viewer choice can therefore stay separate from instrumentation.

Logfire accepts [OTLP traces, metrics and logs from standard clients](https://logfire.pydantic.dev/docs/guides/advanced/alternative-clients/).
Its [Live view](https://pydantic.dev/docs/logfire/observe/live/) shows arriving
traces and supports historical search. The
[SQL Workbench](https://pydantic.dev/docs/logfire/observe/explore/) queries span/log
records and metrics. [Dashboards](https://pydantic.dev/docs/logfire/observe/dashboards/)
provide standard LLM/system panels and custom SQL charts. Some standard metric
dashboards require delta temporality; cumulative values can still be queried.

[Self-hosted Logfire](https://pydantic.dev/docs/logfire/deploy/self-hosted-deployment/overview/)
is Enterprise, uses private images, and requires Kubernetes, PostgreSQL and
object storage. For this project's small deployment, the hosted offering is the
more practical Logfire candidate. This is a deployment judgment, not a claim
that self-hosting is unavailable.

## Self-hosted evidence and limits

[Aspire standalone](https://aspire.dev/dashboard/standalone/) accepts all three
signals without requiring a .NET application or AppHost. Current
[persistence support](https://aspire.dev/dashboard/data-persistence/) includes
SQLite `Run` and `Resume` modes and historical run selection. It should not be
described as in-memory-only. Nevertheless, the docs explicitly scope it to
development history and short-term diagnostics; bounded eviction and schema
changes mean it should not be the sole paper archive. Verify these newer
features exist in the exact release selected for deployment.

[OpenObserve's quickstart](https://openobserve.ai/docs/quickstart/) supports a
persistent single-node binary/container. Its
[metrics ingestion](https://openobserve.ai/docs/ingestion/metrics/) accepts OTLP.
The research pass did not conclusively establish current delta-metric support;
validate a counter and histogram against the selected release. Use the intended
OSS/Enterprise image, not an assumption based on an Enterprise demo.

[SigNoz's Docker installation](https://signoz.io/docs/install/docker/) is a larger
Compose deployment with ClickHouse, Keeper, PostgreSQL and collector services.
[Trace/log navigation](https://signoz.io/docs/traces-management/guides/correlate-traces-and-logs/)
uses exported trace/span IDs. Its
[metric rules](https://signoz.io/docs/metrics-management/types-and-aggregation/)
support cumulative and delta series but exclude delta series from PromQL;
the query builder is an alternative.

[Uptrace's current Docker guide](https://uptrace.dev/get/hosted/docker) explicitly
limits Community retention to 14 days and reserves configurable retention for
Premium. That matters if months of experiment history must stay queryable.

More source-specific details, licensing and verified/unverified limitations are
in [telemetry-viewer-research.md](telemetry-viewer-research.md).

## What selection must resolve

- Live investigation versus months of retained, queryable experiment history.
- Managed maintenance and billing versus local storage and service ownership.
- Nested agent/model/tool traces, run/case filters and correlated structured logs.
- Counter/histogram charts with correct temporality and restart handling.
- Telemetry volume, content capture and retention appropriate to long LLM runs.
- Actual features of the selected edition and pinned release.

After selecting a viewer, the full design can define run/case/turn spans,
scheduler and compaction context, correlated logs, custom metrics, server/GPU
collection, export buffering and shutdown. Telemetry failures must remain
independent of benchmark execution; no external guard is reintroduced.
