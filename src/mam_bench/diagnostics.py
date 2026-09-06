"""Package logging without model payloads or exception bodies."""

import logging
import os
import traceback


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
    """Preserve exception types and stack locations, never bodies or locals."""
    chain: list[str] = []
    seen: set[int] = set()
    while id(error) not in seen:
        seen.add(id(error))
        frames = traceback.extract_tb(error.__traceback__)
        chain.append(
            type(error).__name__
            + ": "
            + " -> ".join(f"{frame.filename}:{frame.lineno} in {frame.name}" for frame in frames)
        )
        cause = error.__cause__ or error.__context__
        if cause is None:
            break
        error = cause
    return " | ".join(chain)
