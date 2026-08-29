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
                    {"choices": [{"text": "a", "token_ids": [97]}]},
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
        self.assertIs(body["return_token_ids"], True)
        self.assertNotIn("stream_interval", body)
        self.assertIs(body["ignore_eos"], True)

    def test_sglang_cumulative_stream_is_not_duplicated(self):
        captured = []

        def post(url, body, timeout):
            captured.append((url, body, timeout))
            return FixtureResponse(
                [
                    {
                        "output_ids": [97],
                        "meta_info": {"completion_tokens": 1},
                    },
                    {
                        "output_ids": [97, 98],
                        "meta_info": {"completion_tokens": 2},
                    },
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
        self.assertEqual(observation.output_text, "")
        self.assertEqual(observation.output_token_ids, [97, 98])
        body = captured[0][1]
        self.assertEqual(body["input_ids"], [21])
        self.assertEqual(body["sampling_params"]["temperature"], 0)
        self.assertEqual(body["sampling_params"]["sampling_seed"], 0)
        self.assertIs(body["sampling_params"]["ignore_eos"], True)
        self.assertNotIn("seed", body["sampling_params"])

    def test_baselines_fail_closed_without_exact_output_token_ids(self):
        for adapter, event in (
            ("vllm-completions-sse-v1", {"choices": [{"text": "a"}]}),
            (
                "sglang-generate-sse-v1",
                {"meta_info": {"completion_tokens": 1}},
            ),
        ):
            with self.subTest(adapter=adapter), patch(
                "benchmarks.qwen3_comparison.adapters._post_stream",
                lambda url, body, timeout, event=event: FixtureResponse([event]),
            ):
                with self.assertRaises(AdapterError):
                    generate(self.engine(adapter), [1], 1, 5, lambda text: [97])

    def test_sglang_rejects_non_cumulative_output_ids(self):
        response = FixtureResponse(
            [
                {"output_ids": [97], "meta_info": {}},
                {"output_ids": [98], "meta_info": {}},
            ]
        )
        with patch(
            "benchmarks.qwen3_comparison.adapters._post_stream",
            lambda url, body, timeout: response,
        ):
            with self.assertRaisesRegex(AdapterError, "not cumulative"):
                generate(
                    self.engine("sglang-generate-sse-v1"),
                    [1],
                    2,
                    5,
                    lambda text: [97, 98],
                )

    def test_sglang_cumulative_stream_preserves_repeated_token_ids(self):
        response = FixtureResponse(
            [
                {"output_ids": [97], "meta_info": {}},
                {"output_ids": [97, 97], "meta_info": {}},
            ]
        )
        with patch(
            "benchmarks.qwen3_comparison.adapters._post_stream",
            lambda url, body, timeout: response,
        ):
            observation = generate(
                self.engine("sglang-generate-sse-v1"),
                [1],
                2,
                5,
                lambda text: [97, 97],
            )
        self.assertEqual(observation.output_token_ids, [97, 97])

    def test_stream_engine_error_fails_even_when_done_follows(self):
        response = FixtureResponse([{"error": {"message": "bad request"}}])
        with patch(
            "benchmarks.qwen3_comparison.adapters._post_stream",
            lambda url, body, timeout: response,
        ):
            with self.assertRaisesRegex(AdapterError, "engine error"):
                generate(
                    self.engine("sglang-generate-sse-v1"),
                    [1],
                    1,
                    5,
                    lambda text: [97],
                )

    def test_stream_without_done_fails_closed(self):
        response = FixtureResponse([{"choices": [{"text": "a", "token_ids": [97]}]}])
        response.lines.pop()
        with patch(
            "benchmarks.qwen3_comparison.adapters._post_stream",
            lambda url, body, timeout: response,
        ):
            with self.assertRaisesRegex(AdapterError, "without the terminal"):
                generate(
                    self.engine("vllm-completions-sse-v1"),
                    [1],
                    1,
                    5,
                    lambda text: [97],
                )

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
                    {
                        "schema": "lunaflux.benchmark-terminal.v1",
                        "text": "b",
                    },
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
        self.assertEqual(observation.output_text, "ab")
        body = captured[0][1]
        self.assertEqual(body["input_token_ids"], [31])
        self.assertEqual(body["sampling"]["mode"], "greedy")
        self.assertIs(body["sampling"]["ignore_eos"], True)

    def test_lunaflux_bridge_requires_one_typed_terminal_event(self):
        for events in (
            [
                {
                    "schema": "lunaflux.benchmark-token.v1",
                    "token_id": 7,
                    "text": "a",
                }
            ],
            [{"schema": "lunaflux.benchmark-terminal.v1"}],
            [
                {"schema": "lunaflux.benchmark-terminal.v1", "text": ""},
                {"schema": "lunaflux.benchmark-terminal.v1", "text": ""},
            ],
        ):
            with self.subTest(events=events), patch(
                "benchmarks.qwen3_comparison.adapters._post_stream",
                lambda url, body, timeout, events=events: FixtureResponse(events),
            ):
                with self.assertRaises(AdapterError):
                    generate(
                        self.engine("lunaflux-token-ids-sse-v1"),
                        [31],
                        2,
                        5,
                        lambda text: [],
                    )

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
