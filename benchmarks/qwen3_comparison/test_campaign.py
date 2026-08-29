from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.qwen3_comparison.contract import (
    ContractError,
    PROFILE_NAMES,
    canonical_json_bytes,
    latin_square_order,
    load_workload,
    sha256_bytes,
    validate_model_inventory,
    validate_campaign,
)
from benchmarks.qwen3_comparison.statistics import correctness_join, distribution, summarize


def digest(character: str = "a") -> str:
    return character * 64


def fixture_campaign():
    profiles = []
    for profile_class, output_tokens in (("prefill", 32), ("decode", 256)):
        for concurrency in (1, 8, 32):
            profiles.append(
                {
                    "name": f"{profile_class}-c{concurrency}",
                    "class": profile_class,
                    "concurrency": concurrency,
                    "output_tokens": output_tokens,
                    "request_count": 32,
                }
            )
    engines = []
    for index, (name, adapter, port) in enumerate(
        (
            ("lunaflux", "lunaflux-token-ids-sse-v1", 8100),
            ("vllm", "vllm-completions-sse-v1", 8101),
            ("sglang", "sglang-generate-sse-v1", 8102),
        )
    ):
        engines.append(
            {
                "name": name,
                "adapter": adapter,
                "endpoint": f"http://127.0.0.1:{port}/generate",
                "health_endpoint": f"http://127.0.0.1:{port}/health",
                "model_alias": "Qwen3-0.6B",
                "source_model_inventory_sha256": digest("a"),
                "model_content_sha256": digest("c"),
                "config_sha256": digest("b"),
                "tokenizer_json_sha256": digest("f"),
                "tokenizer_config_sha256": digest("1"),
                "chat_template_sha256": digest("2"),
                "revision_sha256": digest(str(index + 1)),
                "configuration_sha256": digest(str(index + 4)),
                "executable_sha256": digest(str(index + 7)),
                "environment_prefix": "native" if name == "lunaflux" else "/conda/pinned",
                "package_version": "pinned-version",
            }
        )
    return {
        "schema": "lunaflux.qwen3-comparison-campaign.v1",
        "model": {
            "family": "qwen3",
            "model_id": "Qwen3-0.6B",
            "source_model_root": "/model",
            "source_model_inventory": "/model.files.sha256",
            "source_model_inventory_sha256": digest("a"),
            "config_sha256": digest("b"),
            "model_content_sha256": digest("c"),
            "lunaflux_numeric_artifact_sha256": digest("d"),
            "lunaflux_weight_route_manifest_sha256": digest("e"),
        },
        "tokenizer": {
            "tokenizer_json_sha256": digest("f"),
            "tokenizer_config_sha256": digest("1"),
            "chat_template_sha256": digest("2"),
            "add_generation_prompt": True,
        },
        "sampling": {"mode": "greedy", "temperature": 0, "top_p": 1, "seed": 0},
        "profiles": profiles,
        "engines": engines,
        "gpu": {
            "nvidia_smi": "/usr/bin/nvidia-smi",
            "uuid": "GPU-exact",
            "sample_interval_millis": 50,
        },
        "trials_per_profile": 3,
        "warmup_requests_per_profile": 2,
        "request_timeout_seconds": 600,
        "hardware_concurrency_ceiling": 32,
        "ollama_inference_rule": "forbidden: no Ollama result may be inferred",
    }


class ContractTests(unittest.TestCase):
    def test_exact_qwen_matrix_and_latin_square(self):
        admitted = validate_campaign(fixture_campaign())
        self.assertEqual(tuple(profile["name"] for profile in admitted["profiles"]), PROFILE_NAMES)
        self.assertEqual(latin_square_order(0), ("lunaflux", "vllm", "sglang"))
        self.assertEqual(latin_square_order(1), ("vllm", "sglang", "lunaflux"))
        self.assertEqual(latin_square_order(2), ("sglang", "lunaflux", "vllm"))

    def test_other_model_families_and_fourth_engines_fail_closed(self):
        hostile = copy.deepcopy(fixture_campaign())
        hostile["model"]["family"] = "other"
        with self.assertRaises(ContractError):
            validate_campaign(hostile)
        hostile = copy.deepcopy(fixture_campaign())
        hostile["engines"][2]["name"] = "ollama"
        with self.assertRaises(ContractError):
            validate_campaign(hostile)

    def test_digest_sampling_and_capacity_drift_fail_closed(self):
        for mutate in ("digest", "sampling", "capacity"):
            hostile = copy.deepcopy(fixture_campaign())
            if mutate == "digest":
                hostile["tokenizer"]["chat_template_sha256"] = "A" * 64
            elif mutate == "sampling":
                hostile["sampling"]["temperature"] = 0.1
            else:
                hostile["hardware_concurrency_ceiling"] = 8
            with self.assertRaises(ContractError):
                validate_campaign(hostile)

    def test_baseline_environment_prefixes_are_absolute(self):
        hostile = copy.deepcopy(fixture_campaign())
        hostile["engines"][1]["environment_prefix"] = "named-environment"
        with self.assertRaises(ContractError):
            validate_campaign(hostile)

    def test_workload_requires_exact_ids_and_32_cases_per_class(self):
        payload = bytearray()
        for profile_class in ("prefill", "decode"):
            for ordinal in range(32):
                ids = [ordinal + 1, ordinal + 2]
                payload.extend(
                    canonical_json_bytes(
                        {
                            "schema": "lunaflux.qwen3-tokenized-request.v1",
                            "case_id": f"{profile_class}-{ordinal}",
                            "profile_class": profile_class,
                            "chat_render_sha256": digest("a"),
                            "input_token_ids": ids,
                            "input_token_ids_sha256": sha256_bytes(canonical_json_bytes(ids)),
                            "input_tokens": 2,
                        }
                    )
                )
        self.assertEqual(len(load_workload(bytes(payload))), 64)
        row = json.loads(payload.splitlines()[0])
        row["input_token_ids"][0] += 1
        hostile = canonical_json_bytes(row) + b"\n".join(payload.splitlines()[1:]) + b"\n"
        with self.assertRaises(ContractError):
            load_workload(hostile)

    def test_source_inventory_binds_the_exact_qwen_file_set(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory).resolve()
            root = parent / "model"
            root.mkdir()
            (root / "config.json").write_bytes(b'{"model_type":"qwen3"}\n')
            (root / "tokenizer.json").write_bytes(b"{}\n")
            inventory = parent / "model.files.sha256"
            inventory.write_bytes(
                b"".join(
                    f"{sha256_bytes((root / relative).read_bytes())}  {relative}\n".encode()
                    for relative in ("config.json", "tokenizer.json")
                )
            )
            inventory_sha = sha256_bytes(inventory.read_bytes())
            validate_model_inventory(root, inventory, inventory_sha)
            (root / "ambient.bin").write_bytes(b"ambient")
            with self.assertRaises(ContractError):
                validate_model_inventory(root, inventory, inventory_sha)


class SummaryTests(unittest.TestCase):
    def test_descriptive_statistics_have_median_p95_and_ci(self):
        summary = distribution([1, 2, 3, 4], "fixture")
        self.assertEqual(summary["median"], 2.5)
        self.assertIsNotNone(summary["p95"])
        self.assertEqual(len(summary["median_ci95"]), 2)

    def test_correctness_join_requires_all_three_exact_greedy_hashes(self):
        rows = []
        for engine in ("lunaflux", "vllm", "sglang"):
            rows.append(
                {
                    "engine": engine,
                    "profile": "decode-c1",
                    "trial_ordinal": 0,
                    "request_ordinal": 0,
                    "case_id": "decode-0",
                    "ok": True,
                    "output_token_ids_sha256": digest("a"),
                }
            )
        self.assertTrue(correctness_join(rows)[0]["exact_greedy_match"])
        rows[-1]["output_token_ids_sha256"] = digest("b")
        self.assertFalse(correctness_join(rows)[0]["exact_greedy_match"])

    def test_summary_reports_errors_and_never_selects_a_winner(self):
        requests = [
            {
                "engine": "lunaflux",
                "profile": "prefill-c1",
                "ok": True,
                "ttft_millis": 1.0,
                "inter_token_latency_millis": [2.0],
                "e2e_millis": 3.0,
            },
            {
                "engine": "lunaflux",
                "profile": "prefill-c1",
                "ok": False,
                "ttft_millis": None,
                "inter_token_latency_millis": [],
                "e2e_millis": 4.0,
            },
        ]
        trials = [
            {
                "engine": "lunaflux",
                "profile": "prefill-c1",
                "request_throughput_per_second": 1.0,
                "output_token_throughput_per_second": 2.0,
                "gpu_memory_used_peak_mib": 3,
                "gpu_memory_used_delta_peak_mib": 1,
            }
        ]
        summary = summarize(requests, trials)[0]
        self.assertEqual(summary["error_count"], 1)
        self.assertNotIn("winner", summary)


if __name__ == "__main__":
    unittest.main()
