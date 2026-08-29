"""Strict inert campaign and workload admission for Qwen3 comparisons."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ENGINES = ("lunaflux", "vllm", "sglang")
PROFILE_CLASSES = ("prefill", "decode")
CONCURRENCIES = (1, 8, 32)
PROFILE_NAMES = tuple(
    f"{profile_class}-c{concurrency}"
    for profile_class in PROFILE_CLASSES
    for concurrency in CONCURRENCIES
)
HEX = frozenset("0123456789abcdef")
SAFE_RELATIVE = re.compile(r"^[A-Za-z0-9._/-]+$")


class ContractError(ValueError):
    """The campaign is not the exact Qwen-only comparison contract."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(4 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in HEX for c in value):
        raise ContractError(f"{label} must be lowercase SHA-256")
    return value


def read_digest_suffixed(argument: str, label: str) -> tuple[Path, bytes, str]:
    marker = "#sha256="
    if marker not in argument:
        raise ContractError(f"{label} must be digest suffixed")
    raw_path, expected = argument.rsplit(marker, 1)
    require_sha256(expected, f"{label} digest")
    path = Path(raw_path)
    if not path.is_absolute() or not path.is_file() or path.is_symlink():
        raise ContractError(f"{label} must be an absolute regular file")
    if path.resolve() != path:
        raise ContractError(f"{label} path must be canonical")
    payload = path.read_bytes()
    if sha256_bytes(payload) != expected:
        raise ContractError(f"{label} digest mismatch")
    return path, payload, expected


def validate_digest_suffixed_lexical(argument: Any, label: str) -> None:
    if not isinstance(argument, str) or "#sha256=" not in argument:
        raise ContractError(f"{label} must be digest suffixed")
    raw_path, expected = argument.rsplit("#sha256=", 1)
    if not Path(raw_path).is_absolute():
        raise ContractError(f"{label} path must be absolute")
    require_sha256(expected, f"{label} digest")


def engine_bind(engine: dict[str, Any]) -> tuple[str, int]:
    endpoint = urlsplit(engine["endpoint"])
    health = urlsplit(engine["health_endpoint"])
    if (
        endpoint.scheme != "http"
        or health.scheme != "http"
        or endpoint.hostname != "127.0.0.1"
        or health.hostname != "127.0.0.1"
        or endpoint.port is None
        or endpoint.port != health.port
        or endpoint.username is not None
        or endpoint.password is not None
        or health.username is not None
        or health.password is not None
    ):
        raise ContractError("engine endpoints must share one loopback HTTP port")
    return "127.0.0.1", endpoint.port


def validate_model_inventory(root: Path, inventory: Path, expected_sha256: str) -> None:
    require_sha256(expected_sha256, "source model inventory digest")
    if not root.is_absolute() or not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise ContractError("source model root must be a canonical regular directory")
    if not inventory.is_absolute() or not inventory.is_file() or inventory.is_symlink():
        raise ContractError("source model inventory must be an absolute regular file")
    if inventory.resolve() != inventory or sha256_file(inventory) != expected_sha256:
        raise ContractError("source model inventory identity mismatch")
    try:
        inventory.relative_to(root)
    except ValueError:
        pass
    else:
        raise ContractError("source model inventory must be outside the model root")
    payload = inventory.read_bytes()
    if not payload or not payload.endswith(b"\n") or b"\r" in payload:
        raise ContractError("source model inventory is not canonical text")
    declared: list[str] = []
    for raw_line in payload.splitlines():
        try:
            line = raw_line.decode("ascii")
        except UnicodeDecodeError as error:
            raise ContractError("source model inventory is not ASCII") from error
        parts = line.split("  ")
        if len(parts) != 2:
            raise ContractError("source model inventory line is malformed")
        digest, relative = parts
        require_sha256(digest, "source model file digest")
        if (
            not relative
            or not SAFE_RELATIVE.fullmatch(relative)
            or relative.startswith("/")
            or relative.endswith("/")
            or "//" in relative
            or "/./" in relative
            or "/../" in relative
            or relative in (".", "..")
            or relative.endswith("/.")
            or relative.endswith("/..")
        ):
            raise ContractError("source model inventory path is unsafe")
        path = root / relative
        if not path.is_file() or path.is_symlink() or sha256_file(path) != digest:
            raise ContractError(f"source model file mismatch: {relative}")
        declared.append(relative)
    if declared != sorted(declared) or len(declared) != len(set(declared)):
        raise ContractError("source model inventory paths must be sorted and unique")
    actual = []
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ContractError("source model root contains a symbolic link")
        if path.is_file():
            actual.append(path.relative_to(root).as_posix())
        elif not path.is_dir():
            raise ContractError("source model root contains a special file")
    if sorted(actual) != declared:
        raise ContractError("source model inventory is not the exact root file set")


def _require_exact_keys(value: dict[str, Any], keys: set[str], label: str) -> None:
    if set(value) != keys:
        raise ContractError(f"{label} fields are not exact")


def validate_campaign(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ContractError("campaign must be an object")
    _require_exact_keys(
        value,
        {
            "schema",
            "model",
            "tokenizer",
            "sampling",
            "profiles",
            "engines",
            "gpu",
            "trials_per_profile",
            "warmup_rounds_per_profile",
            "request_timeout_seconds",
            "hardware_concurrency_ceiling",
            "ollama_inference_rule",
        },
        "campaign",
    )
    if value["schema"] != "lunaflux.qwen3-comparison-campaign.v1":
        raise ContractError("campaign schema mismatch")
    model = value["model"]
    if not isinstance(model, dict):
        raise ContractError("model must be an object")
    _require_exact_keys(
        model,
        {
            "family",
            "model_id",
            "source_model_root",
            "source_model_inventory",
            "source_model_inventory_sha256",
            "config_sha256",
            "model_content_sha256",
            "lunaflux_numeric_artifact_sha256",
            "lunaflux_weight_route_manifest_sha256",
        },
        "model",
    )
    if model["family"] != "qwen3" or model["model_id"] != "Qwen3-0.6B":
        raise ContractError("only the pinned dense Qwen3-0.6B model is admitted")
    root = Path(model["source_model_root"])
    if not root.is_absolute():
        raise ContractError("source model root must be absolute")
    inventory = Path(model["source_model_inventory"])
    if not inventory.is_absolute():
        raise ContractError("source model inventory must be absolute")
    for key in model:
        if key.endswith("_sha256"):
            require_sha256(model[key], f"model.{key}")

    tokenizer = value["tokenizer"]
    if not isinstance(tokenizer, dict):
        raise ContractError("tokenizer must be an object")
    _require_exact_keys(
        tokenizer,
        {
            "tokenizer_json_sha256",
            "tokenizer_config_sha256",
            "chat_template_sha256",
            "add_generation_prompt",
        },
        "tokenizer",
    )
    for key in ("tokenizer_json_sha256", "tokenizer_config_sha256", "chat_template_sha256"):
        require_sha256(tokenizer[key], f"tokenizer.{key}")
    if tokenizer["add_generation_prompt"] is not True:
        raise ContractError("Qwen chat rendering must add the generation prompt")

    if value["sampling"] != {
        "mode": "greedy",
        "temperature": 0,
        "top_p": 1,
        "seed": 0,
        "ignore_eos": True,
    }:
        raise ContractError("sampling must be exact deterministic greedy")
    if value["trials_per_profile"] != 3:
        raise ContractError("three counterbalanced trials are required")
    warmups = value["warmup_rounds_per_profile"]
    if not isinstance(warmups, int) or isinstance(warmups, bool) or warmups < 1:
        raise ContractError("at least one excluded warmup request is required")
    timeout = value["request_timeout_seconds"]
    if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout < 1 or timeout > 3600:
        raise ContractError("request timeout is outside bounds")
    ceiling = value["hardware_concurrency_ceiling"]
    if not isinstance(ceiling, int) or isinstance(ceiling, bool) or ceiling < 32:
        raise ContractError("hardware must admit the declared concurrency-32 profiles")
    if value["ollama_inference_rule"] != "forbidden: no Ollama result may be inferred":
        raise ContractError("Ollama non-inference rule is missing")

    profiles = value["profiles"]
    if not isinstance(profiles, list) or len(profiles) != 6:
        raise ContractError("the six prefill/decode concurrency profiles are required")
    observed_profiles: list[str] = []
    for profile in profiles:
        if not isinstance(profile, dict):
            raise ContractError("profile must be an object")
        _require_exact_keys(
            profile,
            {"name", "class", "concurrency", "output_tokens", "request_count"},
            "profile",
        )
        expected_name = f"{profile['class']}-c{profile['concurrency']}"
        if profile["class"] not in PROFILE_CLASSES or profile["concurrency"] not in CONCURRENCIES:
            raise ContractError("profile class or concurrency is outside the matrix")
        if profile["name"] != expected_name:
            raise ContractError("profile name is not canonical")
        if profile["concurrency"] > ceiling:
            raise ContractError("profile exceeds declared hardware capacity")
        if not isinstance(profile["output_tokens"], int) or profile["output_tokens"] <= 0:
            raise ContractError("output-token limit must be positive")
        if not isinstance(profile["request_count"], int) or profile["request_count"] < profile["concurrency"]:
            raise ContractError("request count must fill the declared concurrency")
        observed_profiles.append(profile["name"])
    if tuple(observed_profiles) != PROFILE_NAMES:
        raise ContractError("profile order must be fixed prefill/decode then 1/8/32")

    engines = value["engines"]
    if not isinstance(engines, list) or len(engines) != 3:
        raise ContractError("exactly LunaFlux, vLLM, and SGLang are required")
    observed_engines: list[str] = []
    for engine in engines:
        if not isinstance(engine, dict):
            raise ContractError("engine must be an object")
        _require_exact_keys(
            engine,
            {
                "name",
                "adapter",
                "endpoint",
                "health_endpoint",
                "model_alias",
                "source_model_inventory_sha256",
                "model_content_sha256",
                "config_sha256",
                "tokenizer_json_sha256",
                "tokenizer_config_sha256",
                "chat_template_sha256",
                "revision_sha256",
                "configuration_sha256",
                "executable_sha256",
                "environment_prefix",
                "package_version",
                "lifecycle",
                "authenticated_capacity_receipt",
                "execution_policy",
                "lunaflux_lifecycle",
            },
            "engine",
        )
        name = engine["name"]
        expected_adapter = {
            "lunaflux": "lunaflux-token-ids-sse-v1",
            "vllm": "vllm-completions-sse-v1",
            "sglang": "sglang-generate-sse-v1",
        }.get(name)
        if expected_adapter is None or engine["adapter"] != expected_adapter:
            raise ContractError("engine or adapter is outside the Qwen comparison")
        if not isinstance(engine["endpoint"], str) or not isinstance(engine["health_endpoint"], str):
            raise ContractError("engine endpoints must be strings")
        engine_bind(engine)
        endpoint = urlsplit(engine["endpoint"])
        health = urlsplit(engine["health_endpoint"])
        expected_path = {
            "lunaflux": "/benchmark/v1/token-ids",
            "vllm": "/v1/completions",
            "sglang": "/generate",
        }[name]
        if (
            endpoint.path != expected_path
            or endpoint.query
            or endpoint.fragment
            or health.path != "/health"
            or health.query
            or health.fragment
        ):
            raise ContractError(f"engine.{name} endpoint paths are not exact")
        if engine["model_alias"] != "Qwen3-0.6B":
            raise ContractError("all engines must expose the same Qwen model alias")
        shared_identities = {
            "source_model_inventory_sha256": model["source_model_inventory_sha256"],
            "model_content_sha256": model["model_content_sha256"],
            "config_sha256": model["config_sha256"],
            "tokenizer_json_sha256": tokenizer["tokenizer_json_sha256"],
            "tokenizer_config_sha256": tokenizer["tokenizer_config_sha256"],
            "chat_template_sha256": tokenizer["chat_template_sha256"],
        }
        for key, expected in shared_identities.items():
            if engine[key] != expected:
                raise ContractError(f"engine.{name}.{key} differs from the shared Qwen input")
        for key in ("configuration_sha256", "executable_sha256"):
            require_sha256(engine[key], f"engine.{name}.{key}")
        if not isinstance(engine["package_version"], str) or not engine["package_version"]:
            raise ContractError("engine package version pin is missing")
        environment = engine["environment_prefix"]
        if name == "lunaflux":
            require_sha256(engine["revision_sha256"], "engine.lunaflux.revision_sha256")
            if engine["package_version"] != engine["revision_sha256"]:
                raise ContractError("LunaFlux package revision identity differs")
            if environment != "native":
                raise ContractError("LunaFlux environment must be the native runtime")
            validate_digest_suffixed_lexical(
                engine["authenticated_capacity_receipt"],
                "LunaFlux authenticated concurrency-32 capacity receipt",
            )
            native = engine["lunaflux_lifecycle"]
            if not isinstance(native, dict):
                raise ContractError("LunaFlux combined runtime lifecycle is missing")
            _require_exact_keys(
                native,
                {
                    "runtime_executable",
                    "supervisor_executable",
                    "bridge_executable",
                    "deployment",
                    "tokenizer_json",
                    "launch_file",
                    "release_bind",
                    "runtime_address",
                    "model_plan_sha256",
                    "max_input_tokens",
                    "max_output_tokens",
                    "max_token_id",
                    "max_context_tokens",
                },
                "LunaFlux combined runtime lifecycle",
            )
            for key in (
                "runtime_executable",
                "supervisor_executable",
                "bridge_executable",
                "deployment",
                "tokenizer_json",
                "launch_file",
                "release_bind",
            ):
                validate_digest_suffixed_lexical(native[key], f"LunaFlux {key}")
            if native["runtime_executable"].rsplit("#sha256=", 1)[1] != engine[
                "executable_sha256"
            ]:
                raise ContractError("LunaFlux runtime executable identity differs")
            tokenizer_path = native["tokenizer_json"].rsplit("#sha256=", 1)[0]
            if (
                tokenizer_path != str(root / "tokenizer.json")
                or native["tokenizer_json"].rsplit("#sha256=", 1)[1]
                != tokenizer["tokenizer_json_sha256"]
            ):
                raise ContractError("LunaFlux bridge tokenizer identity differs")
            deployment_path, deployment_digest = native["deployment"].rsplit(
                "#sha256=", 1
            )
            launch_path, launch_digest = native["launch_file"].rsplit("#sha256=", 1)
            if (
                launch_path != str(Path(deployment_path) / "lunaflux.launch.json")
                or deployment_digest != launch_digest
                or deployment_digest != engine["configuration_sha256"]
            ):
                raise ContractError("LunaFlux deployment launch identity differs")
            runtime_endpoint = urlsplit("luna+tcp://" + native["runtime_address"])
            _, bridge_port = engine_bind(engine)
            if (
                runtime_endpoint.hostname != "127.0.0.1"
                or runtime_endpoint.port is None
                or runtime_endpoint.port == bridge_port
                or runtime_endpoint.username is not None
                or runtime_endpoint.password is not None
                or runtime_endpoint.path
                or runtime_endpoint.query
                or runtime_endpoint.fragment
            ):
                raise ContractError("LunaFlux native runtime address is not exact loopback")
            require_sha256(native["model_plan_sha256"], "LunaFlux model plan digest")
            for key in (
                "max_input_tokens",
                "max_output_tokens",
                "max_token_id",
                "max_context_tokens",
            ):
                item = native[key]
                if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
                    raise ContractError(f"LunaFlux {key} is invalid")
            if native["max_input_tokens"] + native["max_output_tokens"] > native[
                "max_context_tokens"
            ]:
                raise ContractError("LunaFlux benchmark limits exceed context")
            if native["max_output_tokens"] < max(
                profile["output_tokens"] for profile in profiles
            ):
                raise ContractError("LunaFlux output limit does not admit every profile")
        elif not isinstance(environment, str) or not environment.startswith("/"):
            raise ContractError("baseline Conda environment prefix must be absolute")
        elif engine["revision_sha256"] is not None:
            raise ContractError("baseline engines cannot claim an unauthenticated revision")
        elif engine["authenticated_capacity_receipt"] is not None:
            raise ContractError("baseline engines cannot claim LunaFlux capacity authority")
        elif engine["lunaflux_lifecycle"] is not None:
            raise ContractError("baseline engines cannot claim LunaFlux lifecycle authority")
        lifecycle = engine["lifecycle"]
        if not isinstance(lifecycle, dict):
            raise ContractError("engine lifecycle must be an object")
        _require_exact_keys(
            lifecycle,
            {
                "launcher",
                "launcher_sha256",
                "runtime_executable_sha256",
                "startup_timeout_seconds",
                "shutdown_timeout_seconds",
                "drain_quiet_millis",
                "expected_exit_codes",
            },
            "engine lifecycle",
        )
        if not isinstance(lifecycle["launcher"], str) or not Path(lifecycle["launcher"]).is_absolute():
            raise ContractError("lifecycle launcher must be absolute")
        require_sha256(lifecycle["launcher_sha256"], "lifecycle launcher digest")
        require_sha256(lifecycle["runtime_executable_sha256"], "runtime executable digest")
        if lifecycle["runtime_executable_sha256"] != engine["executable_sha256"]:
            raise ContractError("runtime executable identity differs from engine identity")
        for key, lower, upper in (
            ("startup_timeout_seconds", 1, 3600),
            ("shutdown_timeout_seconds", 1, 300),
            ("drain_quiet_millis", 0, 60000),
        ):
            item = lifecycle[key]
            if not isinstance(item, int) or isinstance(item, bool) or item < lower or item > upper:
                raise ContractError(f"lifecycle {key} is outside bounds")
        exit_codes = lifecycle["expected_exit_codes"]
        if not isinstance(exit_codes, list) or not exit_codes or any(
            not isinstance(code, int) or isinstance(code, bool) or code < -255 or code > 255
            for code in exit_codes
        ):
            raise ContractError("lifecycle expected exit codes are invalid")
        policy = engine["execution_policy"]
        if policy != {
            "prefix_reuse": False,
            "scheduler_policy": "fcfs",
            "kv_cache_dtype": "auto",
            "max_concurrent_sequences": 32,
            "ignore_eos": True,
        }:
            raise ContractError("engine execution/cache policy is not the fair fixed policy")
        observed_engines.append(name)
    if tuple(observed_engines) != ENGINES:
        raise ContractError("engine order must be LunaFlux, vLLM, SGLang")

    gpu = value["gpu"]
    if not isinstance(gpu, dict):
        raise ContractError("gpu must be an object")
    _require_exact_keys(
        gpu,
        {
            "nvidia_smi",
            "uuid",
            "sample_interval_millis",
            "clean_memory_used_max_mib",
            "clean_timeout_seconds",
            "clean_poll_interval_millis",
            "cooldown_seconds",
        },
        "gpu",
    )
    if not Path(gpu["nvidia_smi"]).is_absolute() or not gpu["uuid"].startswith("GPU-"):
        raise ContractError("GPU sampler identity is incomplete")
    interval = gpu["sample_interval_millis"]
    if not isinstance(interval, int) or isinstance(interval, bool) or interval < 10 or interval > 1000:
        raise ContractError("GPU sampling interval is outside bounds")
    for key, lower, upper in (
        ("clean_memory_used_max_mib", 0, 8192),
        ("clean_timeout_seconds", 1, 3600),
        ("clean_poll_interval_millis", 50, 10000),
        ("cooldown_seconds", 0, 600),
    ):
        item = gpu[key]
        if not isinstance(item, int) or isinstance(item, bool) or item < lower or item > upper:
            raise ContractError(f"gpu.{key} is outside bounds")
    return value


def load_workload(payload: bytes) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    counts = {profile_class: 0 for profile_class in PROFILE_CLASSES}
    for ordinal, raw_line in enumerate(payload.splitlines()):
        if not raw_line:
            raise ContractError("workload contains an empty line")
        try:
            row = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise ContractError(f"workload row {ordinal} is invalid JSON") from error
        if not isinstance(row, dict):
            raise ContractError("workload row must be an object")
        _require_exact_keys(
            row,
            {
                "schema",
                "case_id",
                "profile_class",
                "chat_render_sha256",
                "input_token_ids",
                "input_token_ids_sha256",
                "input_tokens",
            },
            "workload row",
        )
        if row["schema"] != "lunaflux.qwen3-tokenized-request.v1":
            raise ContractError("workload row schema mismatch")
        case_id = row["case_id"]
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ContractError("workload case IDs must be unique and nonempty")
        seen.add(case_id)
        profile_class = row["profile_class"]
        if profile_class not in PROFILE_CLASSES:
            raise ContractError("workload profile class is invalid")
        token_ids = row["input_token_ids"]
        if not isinstance(token_ids, list) or not token_ids or any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in token_ids
        ):
            raise ContractError("input token IDs must be nonnegative integers")
        if row["input_tokens"] != len(token_ids):
            raise ContractError("input token count mismatch")
        encoded_ids = canonical_json_bytes(token_ids)
        if sha256_bytes(encoded_ids) != require_sha256(
            row["input_token_ids_sha256"], "input token ID digest"
        ):
            raise ContractError("input token ID digest mismatch")
        require_sha256(row["chat_render_sha256"], "chat rendering digest")
        counts[profile_class] += 1
        rows.append(row)
    if any(count < 32 for count in counts.values()):
        raise ContractError("workload needs at least 32 exact token-ID cases per class")
    return rows


def latin_square_order(trial_ordinal: int) -> tuple[str, str, str]:
    if trial_ordinal not in (0, 1, 2):
        raise ContractError("trial ordinal is outside the Latin square")
    return tuple(ENGINES[(trial_ordinal + offset) % 3] for offset in range(3))
