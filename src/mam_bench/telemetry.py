"""Native agent telemetry saved locally and optionally delivered to Logfire."""

import logging
from collections.abc import Generator
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
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
)
from opentelemetry.trace import Tracer
from pydantic import SecretStr
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings
from pydantic_settings import BaseSettings, SettingsConfigDict


class _LogfireSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LOGFIRE_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    token: SecretStr = SecretStr("")
    base_url: str = "https://logfire-us.pydantic.dev"


@contextmanager
def collect_telemetry(directory: Path, *, send_to_logfire: bool = False) -> Generator[Tracer]:
    """Collect one CLI invocation, optionally delivering it to Logfire.

    This owns process-wide Agent instrumentation for the invocation. Use one
    collection context per process; independent benchmark invocations use
    separate processes. Library imports never initialize providers or export.
    """
    connection = _LogfireSettings() if send_to_logfire else None
    token = connection.token.get_secret_value() if connection is not None else ""
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
                ConsoleSpanExporter(
                    out=traces, formatter=lambda span: span.to_json(indent=None) + "\n"
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
        if connection is not None and token:
            base = connection.base_url.rstrip("/")
            headers = {"Authorization": token}
            trace_provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=f"{base}/v1/traces", headers=headers, timeout=3)
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
                include_content=True,
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
        handler = LoggingHandler(logger_provider=log_provider)
        logger.addHandler(handler)
        stack.callback(handler.close)
        stack.callback(logger.removeHandler, handler)
        tracer = trace_provider.get_tracer("mam_bench")
        with tracer.start_as_current_span("benchmark", attributes={"run_id": directory.name}):
            logger.info(
                "telemetry.export destination=%s run_id=%s",
                "logfire" if token else "local",
                directory.name,
            )
            yield tracer
