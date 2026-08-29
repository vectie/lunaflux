"""Exclusive-GPU server lifecycle for the external Qwen3 benchmark."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from benchmarks.qwen3_comparison.adapters import AdapterError, check_health
from benchmarks.qwen3_comparison.contract import (
    ContractError,
    canonical_json_bytes,
    engine_bind,
    read_digest_suffixed,
    require_sha256,
    sha256_bytes,
    sha256_file,
)


MODEL_ADMISSION_KEYS = {
    "schema",
    "campaign_sha256",
    "model_root",
    "source_model_inventory",
    "source_model_inventory_sha256",
    "config_sha256",
    "model_content_sha256",
    "tokenizer_json_sha256",
    "tokenizer_config_sha256",
    "chat_template_sha256",
}


def create_model_admission(
    campaign: dict[str, Any], campaign_sha256: str, path: Path
) -> tuple[dict[str, Any], str]:
    """Write the once-authenticated model identity used by every restart."""
    model = campaign["model"]
    tokenizer = campaign["tokenizer"]
    receipt = {
        "schema": "lunaflux.qwen3-model-admission.v1",
        "campaign_sha256": require_sha256(campaign_sha256, "campaign digest"),
        "model_root": str(Path(model["source_model_root"]).resolve()),
        "source_model_inventory": str(Path(model["source_model_inventory"]).resolve()),
        "source_model_inventory_sha256": model["source_model_inventory_sha256"],
        "config_sha256": model["config_sha256"],
        "model_content_sha256": model["model_content_sha256"],
        "tokenizer_json_sha256": tokenizer["tokenizer_json_sha256"],
        "tokenizer_config_sha256": tokenizer["tokenizer_config_sha256"],
        "chat_template_sha256": tokenizer["chat_template_sha256"],
    }
    payload = canonical_json_bytes(receipt)
    path.write_bytes(payload)
    return receipt, sha256_bytes(payload)


def validate_model_admission(
    argument: str, model_root: Path
) -> tuple[dict[str, Any], Path, str]:
    path, payload, digest = read_digest_suffixed(argument, "model admission receipt")
    try:
        receipt = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ContractError("model admission receipt is invalid JSON") from error
    if not isinstance(receipt, dict) or set(receipt) != MODEL_ADMISSION_KEYS:
        raise ContractError("model admission receipt fields are not exact")
    if receipt["schema"] != "lunaflux.qwen3-model-admission.v1":
        raise ContractError("model admission receipt schema mismatch")
    if canonical_json_bytes(receipt) != payload:
        raise ContractError("model admission receipt is not canonical")
    if (
        not model_root.is_absolute()
        or not model_root.is_dir()
        or model_root.resolve() != model_root
        or model_root.is_symlink()
    ):
        raise ContractError("model root is not a canonical directory")
    if receipt["model_root"] != str(model_root):
        raise ContractError("model admission root mismatch")
    for key, value in receipt.items():
        if key.endswith("_sha256"):
            require_sha256(value, f"model admission {key}")
    return receipt, path, digest


def validate_lunaflux_capacity(
    engine: dict[str, Any], model_content_sha256: str
) -> dict[str, Any]:
    _, payload, _ = read_digest_suffixed(
        engine["authenticated_capacity_receipt"], "LunaFlux capacity receipt"
    )
    try:
        receipt = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ContractError("LunaFlux capacity receipt is invalid JSON") from error
    keys = {
        "schema",
        "authenticated",
        "max_concurrency",
        "model_content_sha256",
        "configuration_sha256",
        "runtime_executable_sha256",
        "token_id_sse_bridge_sha256",
        "authentication_sha256",
    }
    if not isinstance(receipt, dict) or set(receipt) != keys:
        raise ContractError("LunaFlux capacity receipt fields are not exact")
    if canonical_json_bytes(receipt) != payload:
        raise ContractError("LunaFlux capacity receipt is not canonical")
    native = engine["lunaflux_lifecycle"]
    if (
        receipt["schema"] != "lunaflux.qwen3-authenticated-capacity.v1"
        or receipt["authenticated"] is not True
        or not isinstance(receipt["max_concurrency"], int)
        or isinstance(receipt["max_concurrency"], bool)
        or receipt["max_concurrency"] < 32
        or receipt["model_content_sha256"] != model_content_sha256
        or receipt["configuration_sha256"] != engine["configuration_sha256"]
        or receipt["runtime_executable_sha256"] != engine["executable_sha256"]
        or receipt["token_id_sse_bridge_sha256"]
        != native["bridge_executable"].rsplit("#sha256=", 1)[1]
    ):
        raise ContractError("LunaFlux authenticated concurrency-32 capacity is not admitted")
    require_sha256(receipt["token_id_sse_bridge_sha256"], "token-ID SSE bridge digest")
    require_sha256(receipt["authentication_sha256"], "capacity authentication digest")
    return receipt


def _run_nvidia_smi(gpu: dict[str, Any], query: str) -> list[str]:
    completed = subprocess.run(
        [
            gpu["nvidia_smi"],
            f"--id={gpu['uuid']}",
            f"--query-{query}",
            "--format=csv,noheader,nounits",
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        timeout=10,
    )
    if completed.returncode != 0:
        raise AdapterError(f"nvidia-smi clean-GPU query failed: {completed.stderr[:256]}")
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def require_clean_gpu(gpu: dict[str, Any]) -> dict[str, Any]:
    deadline = time.monotonic() + gpu["clean_timeout_seconds"]
    last_processes: list[str] = []
    last_memory = -1
    while True:
        process_lines = _run_nvidia_smi(gpu, "compute-apps=pid")
        last_processes = [line for line in process_lines if line.isdecimal()]
        memory_lines = _run_nvidia_smi(gpu, "gpu=memory.used")
        if len(memory_lines) != 1 or not memory_lines[0].isdecimal():
            raise AdapterError("nvidia-smi returned an invalid memory value")
        last_memory = int(memory_lines[0])
        if not last_processes and last_memory <= gpu["clean_memory_used_max_mib"]:
            return {"compute_pids": [], "memory_used_mib": last_memory}
        if time.monotonic() >= deadline:
            raise AdapterError(
                f"target GPU is not clean: pids={last_processes}, memory_used_mib={last_memory}"
            )
        time.sleep(gpu["clean_poll_interval_millis"] / 1000)


def _verify_launcher(engine: dict[str, Any]) -> Path:
    lifecycle = engine["lifecycle"]
    launcher = Path(lifecycle["launcher"])
    if (
        not launcher.is_file()
        or launcher.is_symlink()
        or launcher.resolve() != launcher
        or not os.access(launcher, os.X_OK)
        or sha256_file(launcher) != lifecycle["launcher_sha256"]
    ):
        raise ContractError(f"{engine['name']} launcher identity mismatch")
    return launcher


def _verify_package_version(engine: dict[str, Any]) -> None:
    if engine["name"] == "lunaflux":
        if engine["package_version"] != engine["revision_sha256"]:
            raise ContractError("LunaFlux package revision identity mismatch")
        return
    distribution = {"vllm": "vllm", "sglang": "sglang"}[engine["name"]]
    interpreter = Path(engine["environment_prefix"]) / "bin/python"
    completed = subprocess.run(
        [
            str(interpreter),
            "-c",
            "import importlib.metadata,sys;sys.stdout.write(importlib.metadata.version(sys.argv[1]))",
            distribution,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        text=True,
        timeout=30,
    )
    if completed.returncode != 0 or completed.stdout != engine["package_version"]:
        raise ContractError(f"{engine['name']} package version identity mismatch")


def lifecycle_argv(
    engine: dict[str, Any], campaign: dict[str, Any], model_admission_argument: str
) -> list[str]:
    """Construct the only admitted launcher argv; no campaign shell text is accepted."""
    host, port = engine_bind(engine)
    argv = [
        engine["lifecycle"]["launcher"],
        engine["environment_prefix"],
        engine["package_version"],
        campaign["model"]["source_model_root"],
        model_admission_argument,
        host,
        str(port),
    ]
    if engine["name"] == "lunaflux":
        native = engine["lunaflux_lifecycle"]
        argv.extend(
            [
                native["runtime_executable"],
                native["supervisor_executable"],
                native["bridge_executable"],
                native["deployment"],
                native["tokenizer_json"],
                native["launch_file"],
                native["release_bind"],
                engine["authenticated_capacity_receipt"],
                native["runtime_address"],
                campaign["model"]["model_content_sha256"],
                native["model_plan_sha256"],
                str(native["max_input_tokens"]),
                str(native["max_output_tokens"]),
                str(native["max_token_id"]),
                str(native["max_context_tokens"]),
            ]
        )
    return argv


def lifecycle_environment(
    campaign: dict[str, Any], engine: dict[str, Any] | None = None
) -> dict[str, str]:
    """Return the minimal environment with the measured GPU explicitly isolated."""
    path = "/usr/bin:/bin"
    if engine is not None and engine["environment_prefix"] != "native":
        path = f'{engine["environment_prefix"]}/bin:{path}'
    return {
        "LC_ALL": "C",
        "LANG": "C",
        "TZ": "UTC",
        "PATH": path,
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": campaign["gpu"]["uuid"],
        "PYTHONNOUSERSITE": "1",
    }


def _digest_suffixed_regular_file(argument: str, label: str) -> tuple[Path, str]:
    raw_path, expected = argument.rsplit("#sha256=", 1)
    path = Path(raw_path)
    if (
        not path.is_absolute()
        or not path.is_file()
        or path.is_symlink()
        or path.resolve() != path
        or sha256_file(path) != expected
    ):
        raise ContractError(f"{label} identity mismatch")
    return path, expected


def _process_group_executables(process_group: int) -> dict[str, list[int]]:
    """Return exact executable paths for all live Linux process-group members."""
    observed: dict[str, list[int]] = {}
    for proc_root in Path("/proc").iterdir():
        if not proc_root.name.isdecimal():
            continue
        try:
            stat = (proc_root / "stat").read_text()
            fields = stat[stat.rfind(")") + 2 :].split()
            if len(fields) < 3 or int(fields[2]) != process_group:
                continue
            executable = str((proc_root / "exe").resolve(strict=True))
        except (FileNotFoundError, PermissionError, ValueError):
            continue
        observed.setdefault(executable, []).append(int(proc_root.name))
    return observed


def _lunaflux_process_identities(
    engine: dict[str, Any], process_group: int
) -> dict[str, dict[str, Any]]:
    native = engine["lunaflux_lifecycle"]
    observed = _process_group_executables(process_group)
    identities: dict[str, dict[str, Any]] = {}
    for role, key in (
        ("native_runtime", "runtime_executable"),
        ("native_supervisor", "supervisor_executable"),
        ("token_id_bridge", "bridge_executable"),
    ):
        path, digest = _digest_suffixed_regular_file(native[key], f"LunaFlux {role}")
        pids = observed.get(str(path), [])
        if len(pids) != 1:
            raise ContractError(
                f"LunaFlux {role} is not the unique owned process-group member"
            )
        identities[role] = {
            "pid": pids[0],
            "executable": str(path),
            "executable_sha256": digest,
        }
    return identities


class ServerLifecycle:
    """Own one server process group and its unmeasured lifecycle."""

    def __init__(
        self,
        engine: dict[str, Any],
        campaign: dict[str, Any],
        model_admission_argument: str,
        log_root: Path,
        coordinate: str,
    ) -> None:
        self.engine = engine
        self.campaign = campaign
        self.model_admission_argument = model_admission_argument
        self.log_root = log_root
        self.coordinate = coordinate
        self.process: subprocess.Popen[bytes] | None = None
        self.stdout = None
        self.stderr = None
        self.identity: dict[str, Any] = {}

    def start(self) -> dict[str, Any]:
        clean = require_clean_gpu(self.campaign["gpu"])
        launcher = _verify_launcher(self.engine)
        _verify_package_version(self.engine)
        argv = lifecycle_argv(self.engine, self.campaign, self.model_admission_argument)
        if argv[0] != str(launcher):
            raise ContractError("launcher path changed after identity verification")
        self.stdout = (self.log_root / f"{self.coordinate}.stdout.log").open("xb")
        self.stderr = (self.log_root / f"{self.coordinate}.stderr.log").open("xb")
        self.process = subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=self.stdout,
            stderr=self.stderr,
            env=lifecycle_environment(self.campaign, self.engine),
            close_fds=True,
            start_new_session=True,
        )
        try:
            deadline = time.monotonic() + self.engine["lifecycle"]["startup_timeout_seconds"]
            while True:
                if self.process.poll() is not None:
                    raise AdapterError(f"{self.engine['name']} exited before health readiness")
                try:
                    check_health(self.engine["health_endpoint"])
                    break
                except AdapterError:
                    if time.monotonic() >= deadline:
                        raise AdapterError(f"{self.engine['name']} health readiness timed out")
                    time.sleep(0.25)
            proc_root = Path("/proc") / str(self.process.pid)
            executable = (proc_root / "exe").resolve(strict=True)
            executable_sha = sha256_file(executable)
            command_bytes = (proc_root / "cmdline").read_bytes()
            process_group = os.getpgid(self.process.pid)
            if process_group != self.process.pid:
                raise ContractError(f"{self.engine['name']} did not create an owned process group")
            if self.engine["name"] == "lunaflux":
                owned_executables = _lunaflux_process_identities(
                    self.engine, process_group
                )
            else:
                if executable_sha != self.engine["lifecycle"]["runtime_executable_sha256"]:
                    raise ContractError(
                        f"{self.engine['name']} runtime executable identity mismatch"
                    )
                owned_executables = {
                    "server": {
                        "pid": self.process.pid,
                        "executable": str(executable),
                        "executable_sha256": executable_sha,
                    }
                }
            self.identity = {
                "schema": "lunaflux.qwen3-server-lifecycle.v1",
                "engine": self.engine["name"],
                "coordinate": self.coordinate,
                "pid": self.process.pid,
                "process_group_id": process_group,
                "launcher": str(launcher),
                "launcher_sha256": self.engine["lifecycle"]["launcher_sha256"],
                "package_version": self.engine["package_version"],
                "execution_policy": self.engine["execution_policy"],
                "process_group_leader_executable": str(executable),
                "process_group_leader_executable_sha256": executable_sha,
                "process_group_leader_command_sha256": sha256_bytes(command_bytes),
                "owned_executables": owned_executables,
                "model_admission_sha256": self.model_admission_argument.rsplit("#sha256=", 1)[1],
                "pre_start_clean_gpu": clean,
                "health_ready": True,
                "startup_excluded": True,
            }
            return self.identity
        except BaseException:
            if self.process.poll() is None:
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait(timeout=10)
            if self.stdout is not None:
                self.stdout.close()
            if self.stderr is not None:
                self.stderr.close()
            raise

    def stop(self) -> dict[str, Any]:
        if self.process is None:
            return self.identity
        quiet = self.engine["lifecycle"]["drain_quiet_millis"] / 1000
        if quiet:
            time.sleep(quiet)
        process = self.process
        if process.poll() is None:
            try:
                if self.engine["name"] == "lunaflux":
                    os.kill(process.pid, signal.SIGTERM)
                else:
                    os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=self.engine["lifecycle"]["shutdown_timeout_seconds"])
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=10)
                raise AdapterError(f"{self.engine['name']} did not drain and stop")
        return_code = process.returncode
        lifecycle_error: AdapterError | None = None
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            pass
        else:
            os.killpg(process.pid, signal.SIGKILL)
            lifecycle_error = AdapterError(
                f"{self.engine['name']} left processes in its owned group"
            )
        if return_code not in self.engine["lifecycle"]["expected_exit_codes"]:
            lifecycle_error = AdapterError(
                f"{self.engine['name']} exited with unexpected status {return_code}"
            )
        if self.stdout is not None:
            self.stdout.close()
        if self.stderr is not None:
            self.stderr.close()
        time.sleep(self.campaign["gpu"]["cooldown_seconds"])
        self.identity["exit_status"] = return_code
        self.identity["post_stop_clean_gpu"] = require_clean_gpu(self.campaign["gpu"])
        self.identity["process_exited"] = True
        if lifecycle_error is not None:
            raise lifecycle_error
        return self.identity
