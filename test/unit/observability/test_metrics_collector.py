"""Tests for the slimmed observability.metrics_collector module."""
import unittest

from llm_router_utils.sglang.srt.observability.metrics_collector import (
    QueueCount,
    SchedulerStats,
    compute_routing_key_stats,
)


class _FakeReq:
    """Duck-typed Req: only needs ``.priority`` for QueueCount.from_reqs."""

    def __init__(self, priority):
        self.priority = priority


class TestQueueCount(unittest.TestCase):
    def test_defaults(self):
        qc = QueueCount()
        self.assertEqual(qc.total, 0)
        self.assertIsNone(qc.by_priority)

    def test_from_reqs_no_priority(self):
        reqs = [_FakeReq(0), _FakeReq(1), _FakeReq(2)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=False)
        self.assertEqual(qc.total, 3)
        self.assertIsNone(qc.by_priority)

    def test_from_reqs_with_priority(self):
        reqs = [_FakeReq(0), _FakeReq(0), _FakeReq(1)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertEqual(qc.total, 3)
        self.assertEqual(qc.by_priority, {0: 2, 1: 1})

    def test_from_reqs_empty(self):
        qc = QueueCount.from_reqs([], enable_priority_scheduling=True)
        self.assertEqual(qc.total, 0)
        self.assertEqual(qc.by_priority, {})

    def test_from_reqs_none_priority(self):
        # Mirrors sglang's note: priority=None produces {None: N}
        reqs = [_FakeReq(None), _FakeReq(None)]
        qc = QueueCount.from_reqs(reqs, enable_priority_scheduling=True)
        self.assertEqual(qc.by_priority, {None: 2})


class TestSchedulerStats(unittest.TestCase):
    def test_defaults(self):
        s = SchedulerStats()
        self.assertIsInstance(s.num_running_reqs, QueueCount)
        self.assertEqual(s.num_running_reqs.total, 0)
        self.assertEqual(s.num_queue_reqs.total, 0)
        self.assertEqual(s.gen_throughput, 0.0)
        self.assertEqual(s.cache_hit_rate, 0.0)
        self.assertEqual(s.token_usage, 0.0)
        self.assertEqual(s.num_used_tokens, 0)
        self.assertEqual(s.routing_key_all_req_counts, [])
        self.assertEqual(s.routing_key_running_req_counts, [])
        self.assertTrue(s.fwd_occupancy != s.fwd_occupancy)  # NaN

    def test_field_count_matches_upstream_subset(self):
        # Spot-check a few fields that downstream consumers (router services)
        # are likely to read. If upstream renames or removes any of these,
        # this test will catch the drift on re-import.
        s = SchedulerStats()
        expected = [
            "num_running_reqs",
            "num_queue_reqs",
            "gen_throughput",
            "cache_hit_rate",
            "token_usage",
            "full_token_usage",
            "kv_available_tokens",
            "kv_evictable_tokens",
            "kv_used_tokens",
            "spec_accept_length",
            "num_retracted_reqs",
            "utilization",
            "is_cuda_graph",
            "lora_pool_slots_used",
            "hicache_host_used_tokens",
            "num_streaming_sessions",
            "num_unique_running_routing_keys",
            "routing_key_running_req_counts",
            "routing_key_all_req_counts",
        ]
        for name in expected:
            self.assertTrue(hasattr(s, name), f"missing field: {name}")


class TestComputeRoutingKeyStats(unittest.TestCase):
    def test_empty(self):
        n, counts = compute_routing_key_stats([])
        self.assertEqual(n, 0)
        self.assertEqual(counts, [])

    def test_all_none(self):
        n, counts = compute_routing_key_stats([None, None, None])
        self.assertEqual(n, 0)
        self.assertEqual(counts, [])

    def test_mixed(self):
        n, counts = compute_routing_key_stats(["a", "b", "a", None, "b", "a"])
        self.assertEqual(n, 2)
        self.assertEqual(sorted(counts), [2, 3])


if __name__ == "__main__":
    unittest.main()
