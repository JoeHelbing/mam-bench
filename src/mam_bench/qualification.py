"""Run qualification: a run fails when too many model-controlled turns fall back."""

from mam_bench.diagnostics import ExecutionFailure

FALLBACK_LIMIT = 0.10
EARLY_TURNS = 48
PROVIDER_REASONS = frozenset({"provider_error_limit"})


class FallbackLedger:
    """Count one run's model-controlled turns and fallbacks by cause.

    A cause disqualifies the run when its fallbacks exceed 10% of turns taken by
    the end, exceed 10% of all possible turns (certain early), or reach half of
    the first 48 turns. Model fallbacks fail the run; provider fallbacks leave it
    incomplete for a rerun.
    """

    def __init__(self, possible_turns: int) -> None:
        self.possible_turns = possible_turns
        self.turns = 0
        self.fallbacks = {"model": 0, "provider": 0}

    def record(self, fallback_reason: str | None) -> None:
        self.turns += 1
        if fallback_reason is not None:
            cause = "provider" if fallback_reason in PROVIDER_REASONS else "model"
            self.fallbacks[cause] += 1
        self.check(final=False)

    def check(self, *, final: bool) -> None:
        for cause, count in self.fallbacks.items():
            if (
                count > FALLBACK_LIMIT * self.possible_turns
                or (self.turns <= EARLY_TURNS and 2 * count >= EARLY_TURNS)
                or (final and count > FALLBACK_LIMIT * self.turns)
            ):
                raise ExecutionFailure(
                    f"{cause}_fallbacks",
                    f"{count} of {self.turns} model-controlled turns fell back for {cause} reasons",
                )

    def summary(self) -> dict[str, int]:
        return {"turns": self.turns, **self.fallbacks}
