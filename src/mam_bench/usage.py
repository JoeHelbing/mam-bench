"""Native model usage with benchmark request accounting and stable evidence fields."""

from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal

from pydantic_ai.usage import RequestUsage, RunUsage


@dataclass(frozen=True)
class AgentSessionUsage:
    """An independent snapshot of successful runs, including summary requests."""

    usage: RunUsage = field(default_factory=RunUsage)
    model_requests: int = 0
    cost_complete: bool = True

    def __post_init__(self) -> None:
        if self.model_requests > self.usage.requests:
            raise ValueError("session usage counted more model requests than total requests")
        object.__setattr__(self, "usage", deepcopy(self.usage))

    @property
    def summary_requests(self) -> int:
        return self.usage.requests - self.model_requests

    @property
    def total_requests(self) -> int:
        return self.usage.requests

    @property
    def cost_usd(self) -> Decimal | None:
        return self.usage.cost if self.cost_complete else None

    def __sub__(self, before: AgentSessionUsage) -> AgentSessionUsage:
        """Return the usage added since an earlier snapshot."""
        delta = self.usage - before.usage
        # Native subtraction uses None for unchanged cost; preserve a known zero.
        if self.usage.cost is not None:
            delta.cost = self.usage.cost - (before.usage.cost or Decimal(0))
        return AgentSessionUsage(
            usage=delta,
            model_requests=self.model_requests - before.model_requests,
            cost_complete=self.cost_complete and before.cost_complete,
        )


def request_usage_data(usage: RequestUsage | RunUsage) -> dict[str, object]:
    """Select the portable request-usage fields recorded in benchmark evidence."""
    return {
        "input_tokens": usage.input_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "output_tokens": usage.output_tokens,
        "input_audio_tokens": usage.input_audio_tokens,
        "cache_audio_read_tokens": usage.cache_audio_read_tokens,
        "output_audio_tokens": usage.output_audio_tokens,
        "details": dict(sorted(usage.details.items())),
        "cost_usd": str(usage.cost) if usage.cost is not None else None,
    }


def session_usage_data(
    snapshot: AgentSessionUsage, *, details_as_pairs: bool = False
) -> dict[str, object]:
    """Flatten usage while preserving the existing session and evaluation schemas."""
    data = request_usage_data(snapshot.usage)
    data.update(
        model_requests=snapshot.model_requests,
        summary_requests=snapshot.summary_requests,
        tool_calls=snapshot.usage.tool_calls,
        cost_usd=str(snapshot.cost_usd) if snapshot.cost_usd is not None else None,
    )
    if details_as_pairs:
        data["details"] = sorted(snapshot.usage.details.items())
    return data
