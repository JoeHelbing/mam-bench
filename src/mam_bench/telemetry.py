"""Native OTel collection; full conversations remain in Harness persistence."""

import copy
import logging
import os
from collections.abc import Generator, Sequence
from contextlib import ExitStack, contextmanager
from pathlib import Path

from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

# This official companion package currently does not ship a py.typed marker.
from opentelemetry.instrumentation.logging.handler import (  # pyright: ignore[reportMissingTypeStubs]
    LoggingHandler,
)
from opentelemetry.sdk._logs import LoggerProvider
from opentelemetry.sdk._logs.export import (
    BatchLogRecordProcessor,
    ConsoleLogRecordExporter,
    SimpleLogRecordProcessor,
)
from opentelemetry.sdk.metrics import Counter, Histogram, MeterProvider
from opentelemetry.sdk.metrics.export import (
    AggregationTemporality,
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import Event, ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
    SpanExportResult,
)
from opentelemetry.trace import Status, Tracer
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings

from mam_bench.diagnostics import failure_trace


class _SafeSpanExporter(SpanExporter):
    """Preserve native records while excluding provider exception bodies."""

    def __init__(self, exporter: SpanExporter) -> None:
        self.exporter = exporter

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        safe: list[ReadableSpan] = []
        for span in spans:
            events = [
                Event(
                    event.name,
                    {
                        key: "Exception details omitted."
                        if key in {"exception.message", "exception.stacktrace"}
                        else value
                        for key, value in (event.attributes or {}).items()
                    },
                    event.timestamp,
                )
                for event in span.events
            ]
            safe.append(
                ReadableSpan(
                    name=span.name,
                    context=span.context,
                    parent=span.parent,
                    resource=span.resource,
                    attributes=span.attributes,
                    events=events,
                    links=span.links,
                    kind=span.kind,
                    status=Status(span.status.status_code),
                    start_time=span.start_time,
                    end_time=span.end_time,
                    instrumentation_scope=span.instrumentation_scope,
                )
            )
        return self.exporter.export(safe)

    def shutdown(self) -> None:
        self.exporter.shutdown()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self.exporter.force_flush(timeout_millis)


class _SafeLoggingHandler(LoggingHandler):
    def emit(self, record: logging.LogRecord) -> None:
        error = record.exc_info[1] if record.exc_info is not None else None
        if error is not None:
            record = copy.copy(record)
            record.__dict__["exception_type"] = type(error).__name__
            record.__dict__["exception_trace"] = failure_trace(error)
            record.exc_info = record.exc_text = None
        super().emit(record)


@contextmanager
def collect_telemetry(directory: Path, *, send_to_logfire: bool = False) -> Generator[Tracer]:
    """Collect one CLI invocation, optionally delivering metadata to Logfire.

    This owns process-wide Agent instrumentation for the invocation. Use one
    collection context per process; independent benchmark invocations use
    separate processes. Library imports never initialize providers or export.
    """
    # Validate required credentials before creating providers or reader threads.
    token = os.environ["LOGFIRE_TOKEN"] if send_to_logfire else ""
    target = directory / "telemetry"
    target.mkdir(parents=True, exist_ok=False)
    resource = Resource.create({"service.name": "mam-bench", "mam_bench.run_id": directory.name})
    temporality = {Counter: AggregationTemporality.DELTA, Histogram: AggregationTemporality.DELTA}
    with ExitStack() as stack:
        traces = stack.enter_context((target / "traces.jsonl").open("a", encoding="utf-8"))
        logs = stack.enter_context((target / "logs.jsonl").open("a", encoding="utf-8"))
        metrics = stack.enter_context((target / "metrics.jsonl").open("a", encoding="utf-8"))
        trace_provider = TracerProvider(resource=resource, shutdown_on_exit=False)
        log_provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
        readers = [
            PeriodicExportingMetricReader(
                ConsoleMetricExporter(
                    out=metrics,
                    formatter=lambda data: data.to_json(indent=None) + "\n",
                    preferred_temporality=temporality,
                ),
                export_interval_millis=30000,
            )
        ]
        trace_provider.add_span_processor(
            SimpleSpanProcessor(
                _SafeSpanExporter(
                    ConsoleSpanExporter(
                        out=traces, formatter=lambda span: span.to_json(indent=None) + "\n"
                    )
                )
            )
        )
        log_provider.add_log_record_processor(
            SimpleLogRecordProcessor(
                ConsoleLogRecordExporter(
                    out=logs,
                    formatter=lambda record: record.to_json(indent=None) + "\n",
                )
            )
        )
        stack.callback(trace_provider.shutdown)
        stack.callback(log_provider.shutdown)
        if send_to_logfire:
            base = os.environ.get("LOGFIRE_BASE_URL", "https://logfire-us.pydantic.dev").rstrip("/")
            headers = {"Authorization": token}
            trace_provider.add_span_processor(
                BatchSpanProcessor(
                    _SafeSpanExporter(
                        OTLPSpanExporter(endpoint=f"{base}/v1/traces", headers=headers, timeout=3)
                    )
                )
            )
            log_provider.add_log_record_processor(
                BatchLogRecordProcessor(
                    OTLPLogExporter(endpoint=f"{base}/v1/logs", headers=headers, timeout=3)
                )
            )
            readers.append(
                PeriodicExportingMetricReader(
                    OTLPMetricExporter(
                        endpoint=f"{base}/v1/metrics",
                        headers=headers,
                        timeout=3,
                        preferred_temporality=temporality,
                    ),
                    export_interval_millis=30000,
                )
            )
        meter_provider = MeterProvider(
            resource=resource, metric_readers=readers, shutdown_on_exit=False
        )
        stack.callback(meter_provider.shutdown, timeout_millis=3000)
        Agent.instrument_all(
            InstrumentationSettings(
                tracer_provider=trace_provider,
                meter_provider=meter_provider,
                include_content=False,
                include_binary_content=False,
                include_model_request_parameters=False,
                use_aggregated_usage_attribute_names=True,
            )
        )
        stack.callback(Agent.instrument_all, False)
        logger = logging.getLogger("mam_bench")
        old_level = logger.level
        if old_level == logging.NOTSET:
            logger.setLevel(logging.INFO)
        stack.callback(logger.setLevel, old_level)
        handler = _SafeLoggingHandler(logger_provider=log_provider)
        logger.addHandler(handler)
        stack.callback(handler.close)
        stack.callback(logger.removeHandler, handler)
        tracer = trace_provider.get_tracer("mam_bench")
        with tracer.start_as_current_span("benchmark", attributes={"run_id": directory.name}):
            yield tracer
