from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
from benchmarks.qwen3_comparison.campaign import warm_profile


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
                "endpoint": (
                    f"http://127.0.0.1:{port}/benchmark/v1/token-ids"
                    if name == "lunaflux"
                    else (
                        f"http://127.0.0.1:{port}/v1/completions"
                        if name == "vllm"
                        else f"http://127.0.0.1:{port}/generate"
                    )
                ),
                "health_endpoint": f"http://127.0.0.1:{port}/health",
                "model_alias": "Qwen3-0.6B",
                "source_model_inventory_sha256": digest("a"),
                "model_content_sha256": digest("c"),
                "config_sha256": digest("b"),
                "tokenizer_json_sha256": digest("f"),
                "tokenizer_config_sha256": digest("1"),
                "chat_template_sha256": digest("2"),
                "revision_sha256": digest("1") if name == "lunaflux" else None,
                "configuration_sha256": digest(str(index + 4)),
                "executable_sha256": digest(str(index + 7)),
                "environment_prefix": "native" if name == "lunaflux" else "/conda/pinned",
                "package_version": digest("1") if name == "lunaflux" else "pinned-version",
                "authenticated_capacity_receipt": (
                    "/capacity.json#sha256=" + digest("a") if name == "lunaflux" else None
                ),
                "lunaflux_lifecycle": (
                    {
                        "runtime_executable": "/runtime/lunaflux#sha256=" + digest("7"),
                        "supervisor_executable": "/runtime/supervisor#sha256=" + digest("a"),
                        "bridge_executable": "/runtime/bridge#sha256=" + digest("b"),
                        "deployment": "/deployment#sha256=" + digest("4"),
                        "tokenizer_json": "/model/tokenizer.json#sha256=" + digest("f"),
                        "launch_file": "/deployment/lunaflux.launch.json#sha256="
                        + digest("4"),
                        "release_bind": "/release-bind.v1#sha256=" + digest("e"),
                        "runtime_address": "127.0.0.1:8200",
                        "model_plan_sha256": digest("d"),
                        "max_input_tokens": 4096,
                        "max_output_tokens": 256,
                        "max_token_id": 151935,
                        "max_context_tokens": 40960,
                    }
                    if name == "lunaflux"
                    else None
                ),
                "lifecycle": {
                    "launcher": f"/launchers/{name}",
                    "launcher_sha256": digest("a"),
                    "runtime_executable_sha256": digest(str(index + 7)),
                    "startup_timeout_seconds": 600,
                    "shutdown_timeout_seconds": 60,
                    "drain_quiet_millis": 1000,
                    "expected_exit_codes": [-15, 0],
                },
                "execution_policy": {
                    "prefix_reuse": False,
                    "scheduler_policy": "fcfs",
                    "kv_cache_dtype": "auto",
                    "max_concurrent_sequences": 32,
                    "ignore_eos": True,
                },
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
        "sampling": {
            "mode": "greedy",
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "ignore_eos": True,
        },
        "profiles": profiles,
        "engines": engines,
        "gpu": {
            "nvidia_smi": "/usr/bin/nvidia-smi",
            "uuid": "GPU-exact",
            "sample_interval_millis": 50,
            "clean_memory_used_max_mib": 1024,
            "clean_timeout_seconds": 300,
            "clean_poll_interval_millis": 1000,
            "cooldown_seconds": 10,
        },
        "trials_per_profile": 3,
        "warmup_rounds_per_profile": 2,
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

    def test_only_lunaflux_can_claim_an_exact_source_revision(self):
        hostile = copy.deepcopy(fixture_campaign())
        hostile["engines"][1]["revision_sha256"] = digest("6")
        with self.assertRaises(ContractError):
            validate_campaign(hostile)
        hostile = copy.deepcopy(fixture_campaign())
        hostile["engines"][0]["revision_sha256"] = None
        with self.assertRaises(ContractError):
            validate_campaign(hostile)

    def test_lifecycle_and_lunaflux_bridge_fail_closed(self):
        hostile = copy.deepcopy(fixture_campaign())
        hostile["engines"][0]["endpoint"] = "http://127.0.0.1:8100/v1/responses"
        with self.assertRaises(ContractError):
            validate_campaign(hostile)
        hostile = copy.deepcopy(fixture_campaign())
        hostile["engines"][1]["execution_policy"]["prefix_reuse"] = True
        with self.assertRaises(ContractError):
            validate_campaign(hostile)
        hostile = copy.deepcopy(fixture_campaign())
        hostile["engines"][1]["lifecycle"]["launcher"] = "relative-launcher"
        with self.assertRaises(ContractError):
            validate_campaign(hostile)
        hostile = copy.deepcopy(fixture_campaign())
        hostile["engines"][0]["authenticated_capacity_receipt"] = None
        with self.assertRaises(ContractError):
            validate_campaign(hostile)
        hostile = copy.deepcopy(fixture_campaign())
        hostile["engines"][0]["lunaflux_lifecycle"] = None
        with self.assertRaises(ContractError):
            validate_campaign(hostile)

    def test_baseline_endpoint_paths_are_exact(self):
        hostile = copy.deepcopy(fixture_campaign())
        hostile["engines"][1]["endpoint"] = "http://127.0.0.1:8101/generate"
        with self.assertRaises(ContractError):
            validate_campaign(hostile)
        hostile = copy.deepcopy(fixture_campaign())
        hostile["engines"][2]["health_endpoint"] = "http://127.0.0.1:8102/health?full=1"
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
    def test_warmup_rounds_fill_the_profile_concurrency(self):
        profile = {
            "name": "prefill-c8",
            "class": "prefill",
            "concurrency": 8,
            "output_tokens": 32,
            "request_count": 32,
        }
        workload = [
            {
                "case_id": f"prefill-{ordinal}",
                "profile_class": "prefill",
                "input_token_ids": [ordinal],
                "input_tokens": 1,
                "input_token_ids_sha256": digest("a"),
            }
            for ordinal in range(32)
        ]
        observed = []

        def request(*args):
            observed.append(args[4])
            return {
                "ok": True,
                "token_timing_exact": True,
                "output_count_consistent": True,
            }

        with patch(
            "benchmarks.qwen3_comparison.campaign._request_row", side_effect=request
        ):
            warm_profile(
                {"name": "vllm"}, profile, workload, 2, 60, lambda text: []
            )
        self.assertEqual(sorted(observed), list(range(16)))

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
