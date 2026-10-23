"""Run qualification: a run fails when too many model-controlled turns fall back."""

from mam_bench.diagnostics import ExecutionFailure

FALLBACK_LIMIT = 0.10
EARLY_TURNS = 48


class FallbackLedger:
    """Count one run's model-controlled turns and fallbacks.

    Provider errors never reach this ledger: they are retried, and exhausted
    retries stop the run as incomplete. Every fallback is therefore the model's.
    The run fails when fallbacks exceed 10% of turns taken by the end, exceed 10%
    of all possible turns (certain early), or reach half of the first 48 turns.
    """

    def __init__(self, possible_turns: int) -> None:
        self.possible_turns = possible_turns
        self.turns = 0
        self.fallbacks = 0

    def record(self, fallback_reason: str | None) -> None:
        self.turns += 1
        self.fallbacks += fallback_reason is not None
        self.check(final=False)

    def check(self, *, final: bool) -> None:
        if (
            self.fallbacks > FALLBACK_LIMIT * self.possible_turns
            or (self.turns <= EARLY_TURNS and 2 * self.fallbacks >= EARLY_TURNS)
            or (final and self.fallbacks > FALLBACK_LIMIT * self.turns)
        ):
            raise ExecutionFailure(
                "model_fallbacks",
                f"{self.fallbacks} of {self.turns} model-controlled turns fell back",
            )

    def summary(self) -> dict[str, int]:
        return {"turns": self.turns, "fallbacks": self.fallbacks}
