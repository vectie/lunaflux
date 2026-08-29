from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmarks.qwen3_comparison.contract import ContractError, canonical_json_bytes, sha256_bytes
from benchmarks.qwen3_comparison.lifecycle import (
    _lunaflux_process_identities,
    create_model_admission,
    lifecycle_environment,
    lifecycle_argv,
    require_clean_gpu,
    validate_lunaflux_capacity,
    validate_model_admission,
)
from benchmarks.qwen3_comparison.test_campaign import digest, fixture_campaign


class LifecycleTests(unittest.TestCase):
    def test_launcher_argv_is_fixed_and_never_shell_text(self):
        campaign = fixture_campaign()
        engine = campaign["engines"][1]
        argv = lifecycle_argv(engine, campaign, "/stage/admission.json#sha256=" + digest("a"))
        self.assertEqual(argv[0], "/launchers/vllm")
        self.assertEqual(argv[-2:], ["127.0.0.1", "8101"])
        self.assertEqual(len(argv), 7)
        self.assertNotIn("-c", argv)

    def test_lunaflux_launcher_argv_binds_combined_owned_runtime(self):
        campaign = fixture_campaign()
        engine = campaign["engines"][0]
        admission = "/stage/admission.json#sha256=" + digest("a")
        argv = lifecycle_argv(engine, campaign, admission)
        self.assertEqual(len(argv), 22)
        self.assertEqual(
            argv[:7],
            [
                "/launchers/lunaflux",
                "native",
                digest("1"),
                "/model",
                admission,
                "127.0.0.1",
                "8100",
            ],
        )
        self.assertEqual(argv[7], engine["lunaflux_lifecycle"]["runtime_executable"])
        self.assertEqual(argv[9], engine["lunaflux_lifecycle"]["bridge_executable"])
        self.assertEqual(argv[20:], ["151935", "40960"])

    def test_server_environment_isolates_exact_measured_gpu(self):
        environment = lifecycle_environment(fixture_campaign())
        self.assertEqual(environment["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
        self.assertEqual(environment["CUDA_VISIBLE_DEVICES"], "GPU-exact")
        self.assertEqual(environment["PYTHONNOUSERSITE"], "1")

    def test_once_authenticated_model_receipt_is_small_and_root_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            model = root / "model"
            model.mkdir()
            inventory = root / "inventory.sha256"
            inventory.write_text("fixture\n")
            campaign = fixture_campaign()
            campaign["model"]["source_model_root"] = str(model)
            campaign["model"]["source_model_inventory"] = str(inventory)
            path = root / "admission.json"
            _, receipt_sha = create_model_admission(campaign, digest("9"), path)
            receipt, observed_path, observed_sha = validate_model_admission(
                f"{path}#sha256={receipt_sha}", model
            )
            self.assertEqual(observed_path, path)
            self.assertEqual(observed_sha, receipt_sha)
            self.assertEqual(receipt["model_root"], str(model))
            other = root / "other"
            other.mkdir()
            with self.assertRaises(ContractError):
                validate_model_admission(f"{path}#sha256={receipt_sha}", other)

    def test_lunaflux_capacity_requires_authenticated_c32_and_exact_runtime(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            engine = fixture_campaign()["engines"][0]
            receipt = {
                "schema": "lunaflux.qwen3-authenticated-capacity.v1",
                "authenticated": True,
                "max_concurrency": 32,
                "model_content_sha256": digest("c"),
                "configuration_sha256": engine["configuration_sha256"],
                "runtime_executable_sha256": engine["executable_sha256"],
                "token_id_sse_bridge_sha256": digest("b"),
                "authentication_sha256": digest("d"),
            }
            path = root / "capacity.json"
            payload = canonical_json_bytes(receipt)
            path.write_bytes(payload)
            engine["authenticated_capacity_receipt"] = f"{path}#sha256={sha256_bytes(payload)}"
            self.assertEqual(
                validate_lunaflux_capacity(engine, digest("c"))["max_concurrency"],
                32,
            )
            receipt["max_concurrency"] = 31
            payload = canonical_json_bytes(receipt)
            path.write_bytes(payload)
            engine["authenticated_capacity_receipt"] = f"{path}#sha256={sha256_bytes(payload)}"
            with self.assertRaises(ContractError):
                validate_lunaflux_capacity(engine, digest("c"))

    def test_lunaflux_capacity_rejects_a_different_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            engine = fixture_campaign()["engines"][0]
            receipt = {
                "schema": "lunaflux.qwen3-authenticated-capacity.v1",
                "authenticated": True,
                "max_concurrency": 32,
                "model_content_sha256": digest("c"),
                "configuration_sha256": engine["configuration_sha256"],
                "runtime_executable_sha256": engine["executable_sha256"],
                "token_id_sse_bridge_sha256": digest("e"),
                "authentication_sha256": digest("d"),
            }
            path = Path(directory).resolve() / "capacity.json"
            payload = canonical_json_bytes(receipt)
            path.write_bytes(payload)
            engine["authenticated_capacity_receipt"] = (
                f"{path}#sha256={sha256_bytes(payload)}"
            )
            with self.assertRaisesRegex(ContractError, "capacity"):
                validate_lunaflux_capacity(engine, digest("c"))

    def test_lunaflux_process_group_binds_runtime_supervisor_and_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            engine = fixture_campaign()["engines"][0]
            observed = {}
            for role, key, pid in (
                ("native_runtime", "runtime_executable", 11),
                ("native_supervisor", "supervisor_executable", 12),
                ("token_id_bridge", "bridge_executable", 13),
            ):
                path = root / role
                path.write_bytes(role.encode())
                digest_value = sha256_bytes(role.encode())
                engine["lunaflux_lifecycle"][key] = f"{path}#sha256={digest_value}"
                observed[str(path)] = [pid]
            with patch(
                "benchmarks.qwen3_comparison.lifecycle._process_group_executables",
                return_value=observed,
            ):
                identities = _lunaflux_process_identities(engine, 99)
            self.assertEqual(identities["native_runtime"]["pid"], 11)
            self.assertEqual(identities["token_id_bridge"]["pid"], 13)

    def test_clean_gpu_rejects_foreign_compute_processes(self):
        gpu = fixture_campaign()["gpu"]
        responses = iter((["4321"], ["100"], ["4321"], ["100"]))
        with patch(
            "benchmarks.qwen3_comparison.lifecycle._run_nvidia_smi",
            side_effect=lambda unused_gpu, unused_query: next(responses),
        ), patch("benchmarks.qwen3_comparison.lifecycle.time.monotonic", side_effect=[0, 999]):
            with self.assertRaisesRegex(Exception, "not clean"):
                require_clean_gpu({**gpu, "clean_timeout_seconds": 1})

    def test_clean_gpu_admits_no_processes_under_memory_floor(self):
        gpu = fixture_campaign()["gpu"]
        responses = iter(([], ["100"]))
        with patch(
            "benchmarks.qwen3_comparison.lifecycle._run_nvidia_smi",
            side_effect=lambda unused_gpu, unused_query: next(responses),
        ):
            self.assertEqual(require_clean_gpu(gpu)["memory_used_mib"], 100)


if __name__ == "__main__":
    unittest.main()
