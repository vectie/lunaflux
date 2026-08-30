from __future__ import annotations

import unittest
from collections import UserDict

from benchmarks.qwen3_comparison.contract import ContractError
from benchmarks.qwen3_comparison.prepare_corpus import _direct_token_ids


class DirectTokenIdsTests(unittest.TestCase):
    def test_accepts_legacy_list(self) -> None:
        self.assertEqual(_direct_token_ids([151644, 872, 198]), [151644, 872, 198])

    def test_accepts_batch_encoding_mapping(self) -> None:
        encoded = UserDict({"input_ids": [151644, 872, 198]})
        self.assertEqual(_direct_token_ids(encoded), [151644, 872, 198])

    def test_rejects_mapping_without_input_ids(self) -> None:
        with self.assertRaises(ContractError):
            _direct_token_ids(UserDict({"attention_mask": [1, 1]}))

    def test_rejects_batched_or_invalid_ids(self) -> None:
        for value in ({"input_ids": [[1, 2]]}, {"input_ids": [True]}, "input_ids"):
            with self.subTest(value=value), self.assertRaises(ContractError):
                _direct_token_ids(value)


if __name__ == "__main__":
    unittest.main()
