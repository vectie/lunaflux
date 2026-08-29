#!/usr/bin/env python3
"""Run a result-neutral, counterbalanced Qwen3 engine comparison."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.qwen3_comparison.adapters import (  # noqa: E402
    AdapterError,
    GpuMemorySampler,
    check_health,
    generate,
)
from benchmarks.qwen3_comparison.contract import (  # noqa: E402
    ContractError,
    canonical_json_bytes,
    latin_square_order,
    load_workload,
    read_digest_suffixed,
    sha256_bytes,
    validate_model_inventory,
    validate_campaign,
)
from benchmarks.qwen3_comparison.statistics import correctness_join, summarize  # noqa: E402


class TokenEncoder:
    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer
        self.lock = threading.Lock()

    def __call__(self, text: str) -> list[int]:
        with self.lock:
            encoded = self.tokenizer.encode(text, add_special_tokens=False)
        return [int(value) for value in encoded]


def _sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def load_tokenizer(campaign: dict[str, Any]) -> TokenEncoder:
    model = campaign["model"]
    tokenizer_contract = campaign["tokenizer"]
    root = Path(model["source_model_root"])
    if not root.is_dir() or root.is_symlink() or root.resolve() != root:
        raise ContractError("source model root must be a canonical regular directory")
    required = {
        "config.json": model["config_sha256"],
        "tokenizer.json": tokenizer_contract["tokenizer_json_sha256"],
        "tokenizer_config.json": tokenizer_contract["tokenizer_config_sha256"],
    }
    for relative, expected in required.items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or _sha256_file(path) != expected:
            raise ContractError(f"pinned Qwen file mismatch: {relative}")
    validate_model_inventory(
        root,
        Path(model["source_model_inventory"]),
        model["source_model_inventory_sha256"],
    )
    try:
        from transformers import AutoTokenizer
    except ImportError as error:
        raise ContractError("the benchmark Conda environment lacks transformers") from error
    tokenizer = AutoTokenizer.from_pretrained(
        str(root), local_files_only=True, trust_remote_code=False, use_fast=True
    )
    template = tokenizer.chat_template
    if not isinstance(template, str) or sha256_bytes(template.encode()) != tokenizer_contract["chat_template_sha256"]:
        raise ContractError("Qwen chat template digest mismatch")
    return TokenEncoder(tokenizer)


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(canonical_json_bytes(value))


def _write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    with path.open("wb") as output:
        for value in values:
            output.write(canonical_json_bytes(value))


def _request_row(
    engine: dict[str, Any],
    profile: dict[str, Any],
    workload: dict[str, Any],
    trial_ordinal: int,
    request_ordinal: int,
    timeout_seconds: int,
    encoder: TokenEncoder,
) -> dict[str, Any]:
    submitted = time.perf_counter_ns()
    base = {
        "schema": "lunaflux.qwen3-comparison-request.v1",
        "engine": engine["name"],
        "profile": profile["name"],
        "profile_class": profile["class"],
        "concurrency": profile["concurrency"],
        "trial_ordinal": trial_ordinal,
        "request_ordinal": request_ordinal,
        "case_id": workload["case_id"],
        "input_tokens": workload["input_tokens"],
        "input_token_ids_sha256": workload["input_token_ids_sha256"],
        "output_token_limit": profile["output_tokens"],
        "sampling": "greedy",
    }
    try:
        observation = generate(
            engine,
            workload["input_token_ids"],
            profile["output_tokens"],
            timeout_seconds,
            encoder,
        )
        timestamps = observation.token_timestamps_ns
        ttft = (timestamps[0] - submitted) / 1_000_000 if timestamps else None
        intervals = [
            (timestamps[index] - timestamps[index - 1]) / 1_000_000
            for index in range(1, len(timestamps))
        ]
        output_ids_sha = sha256_bytes(canonical_json_bytes(observation.output_token_ids))
        return {
            **base,
            "ok": True,
            "error_type": None,
            "error_message": None,
            "http_status": observation.http_status,
            "ttft_millis": ttft,
            "inter_token_latency_millis": intervals,
            "e2e_millis": (observation.terminal_ns - submitted) / 1_000_000,
            "output_tokens": len(observation.output_token_ids),
            "server_reported_output_tokens": observation.server_reported_output_tokens,
            "output_token_ids_sha256": output_ids_sha,
            "output_text_utf8_sha256": sha256_bytes(observation.output_text.encode()),
            "stream_chunk_count": observation.stream_chunk_count,
            "token_timing_exact": bool(observation.output_token_ids)
            and observation.stream_chunk_count == len(observation.output_token_ids),
            "output_count_consistent": observation.server_reported_output_tokens
            in (None, len(observation.output_token_ids)),
        }
    except Exception as error:
        terminal = time.perf_counter_ns()
        return {
            **base,
            "ok": False,
            "error_type": type(error).__name__,
            "error_message": str(error)[:1024],
            "http_status": None,
            "ttft_millis": None,
            "inter_token_latency_millis": [],
            "e2e_millis": (terminal - submitted) / 1_000_000,
            "output_tokens": 0,
            "server_reported_output_tokens": None,
            "output_token_ids_sha256": None,
            "output_text_utf8_sha256": None,
            "stream_chunk_count": 0,
            "token_timing_exact": False,
            "output_count_consistent": False,
        }


def _cases_for_trial(
    workload: list[dict[str, Any]], profile: dict[str, Any], trial_ordinal: int
) -> list[dict[str, Any]]:
    candidates = [row for row in workload if row["profile_class"] == profile["class"]]
    count = profile["request_count"]
    offset = trial_ordinal * count
    return [candidates[(offset + ordinal) % len(candidates)] for ordinal in range(count)]


def run_trial(
    engine: dict[str, Any],
    profile: dict[str, Any],
    cases: list[dict[str, Any]],
    trial_ordinal: int,
    order_position: int,
    campaign: dict[str, Any],
    encoder: TokenEncoder,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    check_health(engine["health_endpoint"])
    gpu = campaign["gpu"]
    sampler = GpuMemorySampler(
        gpu["nvidia_smi"], gpu["uuid"], gpu["sample_interval_millis"]
    )
    baseline = sampler.start()
    started = time.perf_counter_ns()
    rows: list[dict[str, Any] | None] = [None] * len(cases)
    with ThreadPoolExecutor(max_workers=profile["concurrency"]) as executor:
        futures = {
            executor.submit(
                _request_row,
                engine,
                profile,
                case,
                trial_ordinal,
                ordinal,
                campaign["request_timeout_seconds"],
                encoder,
            ): ordinal
            for ordinal, case in enumerate(cases)
        }
        for future in as_completed(futures):
            rows[futures[future]] = future.result()
    terminal = time.perf_counter_ns()
    _, peak = sampler.finish()
    check_health(engine["health_endpoint"])
    admitted = [row for row in rows if row is not None]
    if len(admitted) != len(cases):
        raise AdapterError("trial lost a request result")
    duration_seconds = (terminal - started) / 1_000_000_000
    successful_tokens = sum(row["output_tokens"] for row in admitted if row["ok"])
    trial = {
        "schema": "lunaflux.qwen3-comparison-trial.v1",
        "engine": engine["name"],
        "profile": profile["name"],
        "trial_ordinal": trial_ordinal,
        "order_position": order_position,
        "concurrency": profile["concurrency"],
        "request_count": len(admitted),
        "success_count": sum(1 for row in admitted if row["ok"]),
        "error_count": sum(1 for row in admitted if not row["ok"]),
        "duration_seconds": duration_seconds,
        "request_throughput_per_second": len(admitted) / duration_seconds,
        "output_token_throughput_per_second": successful_tokens / duration_seconds,
        "gpu_memory_measurement_scope": "whole-device-with-all-persistent-servers-resident",
        "gpu_memory_used_baseline_mib": baseline,
        "gpu_memory_used_peak_mib": peak,
        "gpu_memory_used_delta_peak_mib": peak - baseline,
        "warmup_excluded": True,
    }
    return admitted, trial


def warm_profile(
    engine: dict[str, Any],
    profile: dict[str, Any],
    workload: list[dict[str, Any]],
    count: int,
    timeout_seconds: int,
    encoder: TokenEncoder,
) -> None:
    cases = [row for row in workload if row["profile_class"] == profile["class"]]
    for ordinal in range(count):
        row = _request_row(engine, profile, cases[ordinal % len(cases)], -1, ordinal, timeout_seconds, encoder)
        if not row["ok"]:
            raise AdapterError(f"excluded warmup failed for {engine['name']}/{profile['name']}")


def run_campaign(
    campaign: dict[str, Any],
    workload: list[dict[str, Any]],
    campaign_sha: str,
    workload_sha: str,
    output: Path,
) -> None:
    if not output.is_absolute() or output.exists() or output.parent.resolve() != output.parent:
        raise ContractError("output must be a new canonical absolute path")
    encoder = load_tokenizer(campaign)
    engines = {engine["name"]: engine for engine in campaign["engines"]}
    for engine in engines.values():
        check_health(engine["health_endpoint"])
    stage = Path(tempfile.mkdtemp(prefix=".qwen3-comparison-stage.", dir=output.parent))
    published = False
    try:
        raw_root = stage / "raw"
        raw_root.mkdir()
        request_rows: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        order_rows: list[dict[str, Any]] = []
        for profile_index, profile in enumerate(campaign["profiles"]):
            warm_order = tuple(
                ("lunaflux", "vllm", "sglang")[(profile_index + offset) % 3]
                for offset in range(3)
            )
            for engine_name in warm_order:
                warm_profile(
                    engines[engine_name],
                    profile,
                    workload,
                    campaign["warmup_requests_per_profile"],
                    campaign["request_timeout_seconds"],
                    encoder,
                )
            profile_root = raw_root / profile["name"]
            profile_root.mkdir()
            for trial_ordinal in range(3):
                cases = _cases_for_trial(workload, profile, trial_ordinal)
                for order_position, engine_name in enumerate(latin_square_order(trial_ordinal)):
                    rows, trial = run_trial(
                        engines[engine_name],
                        profile,
                        cases,
                        trial_ordinal,
                        order_position,
                        campaign,
                        encoder,
                    )
                    _write_jsonl(
                        profile_root
                        / f"trial-{trial_ordinal + 1}-position-{order_position + 1}-{engine_name}.jsonl",
                        rows,
                    )
                    request_rows.extend(rows)
                    trial_rows.append(trial)
                    order_rows.append(
                        {
                            "profile": profile["name"],
                            "trial_ordinal": trial_ordinal,
                            "order_position": order_position,
                            "engine": engine_name,
                        }
                    )
        summaries = summarize(request_rows, trial_rows)
        correctness = correctness_join(request_rows)
        request_measurements_complete = bool(request_rows) and all(
            row["ok"] and row["token_timing_exact"] and row["output_count_consistent"]
            for row in request_rows
        )
        speed_comparison_valid = (
            request_measurements_complete
            and bool(correctness)
            and all(row["exact_greedy_match"] for row in correctness)
        )
        for summary in summaries:
            summary["speed_comparison_valid"] = speed_comparison_valid
            summary["speed_metrics_are_descriptive_only"] = not speed_comparison_valid
        _write_jsonl(stage / "trials.jsonl", trial_rows)
        _write_jsonl(stage / "summary.jsonl", summaries)
        _write_jsonl(stage / "correctness.jsonl", correctness)
        _write_json(
            stage / "manifest.json",
            {
                "schema": "lunaflux.qwen3-comparison-result.v1",
                "campaign_sha256": campaign_sha,
                "workload_sha256": workload_sha,
                "model_id": "Qwen3-0.6B",
                "engine_order": ["lunaflux", "vllm", "sglang"],
                "execution_order": order_rows,
                "warmup_excluded": True,
                "persistent_servers_required": True,
                "summary_policy": "median,p95,deterministic-bootstrap-median-ci95",
                "winner_assumption": "none",
                "speed_comparison_valid": speed_comparison_valid,
                "correctness_failure_invalidates_speed_comparison": True,
                "request_measurements_complete": request_measurements_complete,
                "ollama_inference_rule": "forbidden: no Ollama result may be inferred",
                "correctness_exact_match_count": sum(
                    1 for row in correctness if row["exact_greedy_match"]
                ),
                "correctness_mismatch_or_incomplete_count": sum(
                    1 for row in correctness if not row["exact_greedy_match"]
                ),
            },
        )
        for path in sorted(stage.rglob("*"), reverse=True):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(stage, 0o555)
        os.replace(stage, output)
        published = True
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, help="ABSOLUTE_JSON#sha256=HEX")
    parser.add_argument("--workload", required=True, help="ABSOLUTE_JSONL#sha256=HEX")
    parser.add_argument("--output", required=True, type=Path)
    arguments = parser.parse_args()
    try:
        _, campaign_bytes, campaign_sha = read_digest_suffixed(arguments.campaign, "campaign")
        _, workload_bytes, workload_sha = read_digest_suffixed(arguments.workload, "workload")
        campaign = validate_campaign(json.loads(campaign_bytes))
        workload = load_workload(workload_bytes)
        run_campaign(campaign, workload, campaign_sha, workload_sha, arguments.output)
    except (ContractError, AdapterError, json.JSONDecodeError) as error:
        print(f"Qwen3 comparison rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
