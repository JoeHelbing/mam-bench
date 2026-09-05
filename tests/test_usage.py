import unittest
from decimal import Decimal

from pydantic_ai.usage import RequestUsage, RunUsage

from mam_bench.usage import AgentSessionUsage, request_usage_data, session_usage_data


class AgentSessionUsageTests(unittest.TestCase):
    def test_snapshots_are_independent_and_cache_tokens_are_inclusive(self) -> None:
        native = RunUsage(
            requests=3,
            input_tokens=100,
            cache_read_tokens=60,
            cache_write_tokens=20,
            output_tokens=20,
            details={"reasoning_tokens": 10},
        )
        snapshot = AgentSessionUsage(native, model_requests=2)
        another = AgentSessionUsage(native, model_requests=2)

        native.incr(RunUsage(requests=1, input_tokens=50, details={"reasoning_tokens": 4}))
        another.usage.details["reasoning_tokens"] = 99

        self.assertEqual(snapshot.usage.total_tokens, 120)
        self.assertEqual(snapshot.total_requests, 3)
        self.assertEqual(snapshot.summary_requests, 1)
        self.assertEqual(snapshot.usage.details, {"reasoning_tokens": 10})

    def test_deltas_include_native_fields_details_and_cost(self) -> None:
        native = RunUsage(
            requests=2,
            tool_calls=1,
            input_tokens=100,
            cache_read_tokens=60,
            output_tokens=20,
            cost=Decimal("0.25"),
            details={"reasoning_tokens": 10},
        )
        before = AgentSessionUsage(native, model_requests=1)
        native.incr(
            RunUsage(
                requests=3,
                tool_calls=2,
                input_tokens=50,
                cache_read_tokens=10,
                output_tokens=5,
                cost=Decimal("0.10"),
                details={"reasoning_tokens": 3},
            )
        )
        after = AgentSessionUsage(native, model_requests=3)
        delta = after - before

        self.assertEqual(delta.model_requests, 2)
        self.assertEqual(delta.summary_requests, 1)
        self.assertEqual(delta.usage.tool_calls, 2)
        self.assertEqual(delta.usage.total_tokens, 55)
        self.assertEqual(delta.usage.cache_read_tokens, 10)
        self.assertEqual(delta.usage.details, {"reasoning_tokens": 3})
        self.assertEqual(delta.cost_usd, Decimal("0.10"))
        self.assertEqual(after.usage.details, {"reasoning_tokens": 13})

    def test_deltas_distinguish_zero_unknown_and_initial_cost(self) -> None:
        empty = AgentSessionUsage()
        known = AgentSessionUsage(RunUsage(requests=1, cost=Decimal("0.25")), 1)
        unchanged = AgentSessionUsage(RunUsage(requests=2, cost=Decimal("0.25")), 2)
        unknown = AgentSessionUsage(
            RunUsage(requests=3, cost=Decimal("0.25")), 3, cost_complete=False
        )
        free = AgentSessionUsage(RunUsage(requests=1, cost=Decimal(0)), 1)

        self.assertEqual((known - empty).cost_usd, Decimal("0.25"))
        self.assertEqual((unchanged - known).cost_usd, Decimal(0))
        self.assertEqual((free - empty).cost_usd, Decimal(0))
        self.assertIsNone((unknown - unchanged).cost_usd)
        self.assertIsNone((empty - empty).cost_usd)
        self.assertIsNone(session_usage_data(unknown)["cost_usd"])
        self.assertEqual(session_usage_data(unchanged - known)["cost_usd"], "0.00")

    def test_request_and_session_evidence_preserve_flat_schema(self) -> None:
        request = RequestUsage(
            input_tokens=100,
            cache_write_tokens=20,
            cache_read_tokens=60,
            output_tokens=15,
            input_audio_tokens=3,
            cache_audio_read_tokens=2,
            output_audio_tokens=1,
            details={"z": 2, "a": 1},
            cost=Decimal("0.125"),
        )
        expected: dict[str, object] = {
            "input_tokens": 100,
            "cache_write_tokens": 20,
            "cache_read_tokens": 60,
            "output_tokens": 15,
            "input_audio_tokens": 3,
            "cache_audio_read_tokens": 2,
            "output_audio_tokens": 1,
            "details": {"a": 1, "z": 2},
            "cost_usd": "0.125",
        }
        self.assertEqual(request_usage_data(request), expected)

        native = RunUsage(requests=1, tool_calls=1)
        native.incr(request)
        snapshot = AgentSessionUsage(native, model_requests=1)
        expected.update(model_requests=1, summary_requests=0, tool_calls=1)
        self.assertEqual(session_usage_data(snapshot), expected)
        expected["details"] = [("a", 1), ("z", 2)]
        self.assertEqual(session_usage_data(snapshot, details_as_pairs=True), expected)


if __name__ == "__main__":
    unittest.main()
