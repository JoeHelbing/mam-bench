"""Package logging and complete exception diagnostics."""

import logging
import os
import traceback


class ExecutionFailure(RuntimeError):
    """A classified execution failure retaining its original exception as the cause."""

    def __init__(self, kind: str, message: str) -> None:
        self.kind = kind
        super().__init__(message)


def configure_logging(default_level: str = "INFO") -> None:
    """Configure CLI stderr logging; leave application/root loggers alone."""
    level = os.environ.get("MAM_BENCH_LOG_LEVEL", default_level).upper()
    if level not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ValueError("MAM_BENCH_LOG_LEVEL must be DEBUG, INFO, WARNING, ERROR, or CRITICAL")
    logger = logging.getLogger("mam_bench")
    logger.setLevel(level)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
        logger.addHandler(handler)
    logger.propagate = False


def failure_trace(error: BaseException) -> str:
    """Format the original exception chain, including messages and tracebacks."""
    return "".join(traceback.format_exception(error)).rstrip()
