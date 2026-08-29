from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from benchmarks.qwen3_comparison.adapters import AdapterError, generate


class FixtureResponse:
    def __init__(self, events):
        self.status = 200
        self.lines = []
        for event in events:
            if isinstance(event, bytes):
                self.lines.append(event)
            else:
                self.lines.append(
                    b"data: "
                    + json.dumps(event, separators=(",", ":")).encode()
                    + b"\n\n"
                )
        self.lines.append(b"data: [DONE]\n\n")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __iter__(self):
        return iter(self.lines)


class AdapterTests(unittest.TestCase):
    def engine(self, adapter):
        return {
            "adapter": adapter,
            "endpoint": "http://127.0.0.1:8199/generate",
            "model_alias": "Qwen3-0.6B",
        }

    def test_vllm_uses_exact_token_id_prompt(self):
        captured = []

        def post(url, body, timeout):
            captured.append((url, body, timeout))
            return FixtureResponse(
                [
                    {"choices": [{"text": "a"}]},
                    {"choices": [], "usage": {"completion_tokens": 1}},
                ]
            )

        with patch("benchmarks.qwen3_comparison.adapters._post_stream", post):
            observation = generate(
                self.engine("vllm-completions-sse-v1"),
                [11, 12],
                4,
                5,
                lambda text: [ord(character) for character in text],
            )
        self.assertEqual(observation.output_token_ids, [97])
        body = captured[0][1]
        self.assertEqual(body["prompt"], [11, 12])
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["max_tokens"], 4)

    def test_sglang_cumulative_stream_is_not_duplicated(self):
        captured = []

        def post(url, body, timeout):
            captured.append((url, body, timeout))
            return FixtureResponse(
                [
                    {"text": "a", "meta_info": {"completion_tokens": 1}},
                    {"text": "ab", "meta_info": {"completion_tokens": 2}},
                ]
            )

        with patch("benchmarks.qwen3_comparison.adapters._post_stream", post):
            observation = generate(
                self.engine("sglang-generate-sse-v1"),
                [21],
                8,
                5,
                lambda text: [ord(character) for character in text],
            )
        self.assertEqual(observation.output_text, "ab")
        self.assertEqual(observation.output_token_ids, [97, 98])
        body = captured[0][1]
        self.assertEqual(body["input_ids"], [21])
        self.assertEqual(body["sampling_params"]["temperature"], 0)

    def test_lunaflux_bridge_retains_exact_output_token_id(self):
        captured = []

        def post(url, body, timeout):
            captured.append((url, body, timeout))
            return FixtureResponse(
                [
                    {
                        "schema": "lunaflux.benchmark-token.v1",
                        "token_id": 7,
                        "text": "a",
                    },
                    {"schema": "lunaflux.benchmark-terminal.v1"},
                ]
            )

        with patch("benchmarks.qwen3_comparison.adapters._post_stream", post):
            observation = generate(
                self.engine("lunaflux-token-ids-sse-v1"),
                [31],
                2,
                5,
                lambda text: [],
            )
        self.assertEqual(observation.output_token_ids, [7])
        body = captured[0][1]
        self.assertEqual(body["input_token_ids"], [31])
        self.assertEqual(body["sampling"]["mode"], "greedy")

    def test_non_sse_payload_fails_closed(self):
        with patch(
            "benchmarks.qwen3_comparison.adapters._post_stream",
            lambda url, body, timeout: FixtureResponse([b"not-sse\n"]),
        ):
            with self.assertRaises(AdapterError):
                generate(
                    self.engine("vllm-completions-sse-v1"),
                    [1],
                    1,
                    5,
                    lambda text: [],
                )


if __name__ == "__main__":
    unittest.main()
