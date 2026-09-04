import unittest

from benchmarks.qwen3_comparison.quick_token_id_benchmark import token_summary


class TokenSummaryTest(unittest.TestCase):
    def test_equal_sequences_have_no_divergence(self) -> None:
        summary = token_summary((1, 2, 3), (1, 2, 3))
        self.assertIsNone(summary["first_divergence"])
        self.assertEqual(summary["token_count"], 3)
        self.assertEqual(summary["divergence_window"], (1, 2, 3))

    def test_value_mismatch_reports_bounded_window(self) -> None:
        summary = token_summary((1, 2, 30, 4, 5), (1, 2, 3, 4, 5))
        self.assertEqual(summary["first_divergence"], 2)
        self.assertEqual(summary["divergence_window"], (1, 2, 30, 4, 5))

    def test_length_mismatch_diverges_at_shared_prefix_end(self) -> None:
        summary = token_summary((1, 2), (1, 2, 3))
        self.assertEqual(summary["first_divergence"], 2)


if __name__ == "__main__":
    unittest.main()
