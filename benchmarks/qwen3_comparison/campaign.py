#!/usr/bin/env python3
"""Run a result-neutral, counterbalanced Qwen3 engine comparison."""

from __future__ import annotations

import argparse
import heapq
import json
import os
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
from benchmarks.qwen3_comparison.lifecycle import (  # noqa: E402
    ServerLifecycle,
    create_model_admission,
    validate_lunaflux_capacity,
    validate_model_admission,
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

    def decode(self, token_ids: list[int]) -> str:
        with self.lock:
            return str(
                self.tokenizer.decode(
                    token_ids,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )


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


def _failure_output_path(output: Path) -> Path:
    candidate = output.with_name(output.name + ".failed")
    ordinal = 1
    while candidate.exists():
        candidate = output.with_name(f"{output.name}.failed-{ordinal}")
        ordinal += 1
    return candidate


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_bytes().splitlines():
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ContractError(f"benchmark JSONL row is not an object: {path}")
        rows.append(value)
    return rows


def _reconstruct_trial(
    rows: list[dict[str, Any]],
    profile: dict[str, Any],
    trial_ordinal: int,
    order_position: int,
    engine: dict[str, Any],
) -> dict[str, Any]:
    """Recover aggregate timing after a preserved-stage restart.

    The executor submits requests in ordinal order to a fixed-size worker pool.
    Replaying those measured per-request durations through the same work-conserving
    pool recovers the coordinate makespan without rerunning completed GPU work.
    """
    workers = [0.0] * profile["concurrency"]
    heapq.heapify(workers)
    for row in sorted(rows, key=lambda value: value["request_ordinal"]):
        available = heapq.heappop(workers)
        heapq.heappush(workers, available + row["e2e_millis"] / 1000)
    duration_seconds = max(workers)
    successful_tokens = sum(row["output_tokens"] for row in rows if row["ok"])
    return {
        "schema": "lunaflux.qwen3-comparison-trial.v1",
        "engine": engine["name"],
        "profile": profile["name"],
        "trial_ordinal": trial_ordinal,
        "order_position": order_position,
        "concurrency": profile["concurrency"],
        "request_count": len(rows),
        "success_count": sum(1 for row in rows if row["ok"]),
        "error_count": sum(1 for row in rows if not row["ok"]),
        "duration_seconds": duration_seconds,
        "request_throughput_per_second": len(rows) / duration_seconds,
        "output_token_throughput_per_second": successful_tokens / duration_seconds,
        "gpu_memory_measurement_scope": "unavailable-after-preserved-stage-recovery",
        "gpu_memory_used_baseline_mib": None,
        "gpu_memory_used_peak_mib": None,
        "gpu_memory_used_delta_peak_mib": None,
        "warmup_excluded": True,
        "execution_policy": engine["execution_policy"],
        "duration_reconstructed_from_request_e2e": True,
    }


def _load_completed_engine_group(
    raw_root: Path,
    profiles: list[dict[str, Any]],
    trial_ordinal: int,
    order_position: int,
    engine: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
    grouped_rows: list[dict[str, Any]] = []
    trials: list[dict[str, Any]] = []
    order: list[dict[str, Any]] = []
    for profile in profiles:
        path = (
            raw_root
            / profile["name"]
            / (
                f"trial-{trial_ordinal + 1}-position-"
                f"{order_position + 1}-{engine['name']}.jsonl"
            )
        )
        if not path.is_file():
            return None
        rows = _read_jsonl(path)
        if (
            len(rows) != profile["request_count"]
            or any(
                row.get("engine") != engine["name"]
                or row.get("profile") != profile["name"]
                or row.get("trial_ordinal") != trial_ordinal
                for row in rows
            )
        ):
            return None
        grouped_rows.extend(rows)
        trials.append(
            _reconstruct_trial(rows, profile, trial_ordinal, order_position, engine)
        )
        order.append(
            {
                "profile": profile["name"],
                "trial_ordinal": trial_ordinal,
                "order_position": order_position,
                "engine": engine["name"],
            }
        )
    return grouped_rows, trials, order


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
        output_text = encoder.decode(observation.output_token_ids)
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
            "output_text_utf8_sha256": sha256_bytes(output_text.encode()),
            "stream_chunk_count": observation.stream_chunk_count,
            "token_timing_exact": bool(observation.output_token_ids)
            and observation.stream_chunk_count == len(observation.output_token_ids),
            "output_count_consistent": observation.server_reported_output_tokens
            in (None, len(observation.output_token_ids))
            and len(observation.output_token_ids) == profile["output_tokens"],
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
    selected = [candidates[(offset + ordinal) % len(candidates)] for ordinal in range(count)]
    target = profile.get("input_tokens")
    if target is None:
        return selected
    resized = []
    for row in selected:
        token_ids = row["input_token_ids"]
        if len(token_ids) < target:
            raise ContractError(
                f"workload case {row['case_id']} cannot fill input-token shape {target}"
            )
        if len(token_ids) == target:
            resized.append(row)
            continue
        # Keep the Qwen chat prefix and the seven-token assistant-generation
        # suffix while selecting an exact, deterministic input length.
        suffix_tokens = 7
        shaped_ids = token_ids[: target - suffix_tokens] + token_ids[-suffix_tokens:]
        shaped = dict(row)
        shaped["case_id"] = f"{row['case_id']}-i{target}"
        shaped["input_token_ids"] = shaped_ids
        shaped["input_tokens"] = target
        shaped["input_token_ids_sha256"] = sha256_bytes(canonical_json_bytes(shaped_ids))
        resized.append(shaped)
    return resized


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
        "gpu_memory_measurement_scope": "whole-device-single-engine-resident",
        "gpu_memory_used_baseline_mib": baseline,
        "gpu_memory_used_peak_mib": peak,
        "gpu_memory_used_delta_peak_mib": peak - baseline,
        "warmup_excluded": True,
        "execution_policy": engine["execution_policy"],
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
    cases = _cases_for_trial(workload, profile, 0)
    for round_ordinal in range(count):
        with ThreadPoolExecutor(max_workers=profile["concurrency"]) as executor:
            futures = [
                executor.submit(
                    _request_row,
                    engine,
                    profile,
                    cases[(round_ordinal * profile["concurrency"] + ordinal) % len(cases)],
                    -1,
                    round_ordinal * profile["concurrency"] + ordinal,
                    timeout_seconds,
                    encoder,
                )
                for ordinal in range(profile["concurrency"])
            ]
            rows = [future.result() for future in as_completed(futures)]
        failed = [
            row
            for row in rows
            if not row["ok"]
            or not row["output_count_consistent"]
        ]
        if failed:
            details = [
                {
                    "request_ordinal": row["request_ordinal"],
                    "ok": row["ok"],
                    "token_timing_exact": row["token_timing_exact"],
                    "output_count_consistent": row["output_count_consistent"],
                    "output_tokens": row["output_tokens"],
                    "stream_chunk_count": row["stream_chunk_count"],
                    "error_message": row["error_message"],
                }
                for row in failed[:3]
            ]
            raise AdapterError(
                f"excluded warmup failed for {engine['name']}/{profile['name']}: "
                + json.dumps(details, sort_keys=True)
            )


def run_campaign(
    campaign: dict[str, Any],
    workload: list[dict[str, Any]],
    campaign_sha: str,
    workload_sha: str,
    output: Path,
    resume: Path | None = None,
) -> None:
    if not output.is_absolute() or output.exists() or output.parent.resolve() != output.parent:
        raise ContractError("output must be a new canonical absolute path")
    if resume is None:
        stage = Path(tempfile.mkdtemp(prefix=".qwen3-comparison-stage.", dir=output.parent))
    else:
        if (
            not resume.is_absolute()
            or not resume.is_dir()
            or resume.is_symlink()
            or resume.resolve() != resume
            or resume.parent != output.parent
        ):
            raise ContractError("resume stage must be a canonical directory beside output")
        stage = resume
    published = False
    try:
        # This is the campaign's only full source-model inventory scan. Each
        # server restart consumes the small digest-bound admission receipt.
        encoder = load_tokenizer(campaign)
        engines = {engine["name"]: engine for engine in campaign["engines"]}
        capacity = validate_lunaflux_capacity(
            engines["lunaflux"], campaign["model"]["model_content_sha256"]
        )
        _, admission_sha = create_model_admission(
            campaign, campaign_sha, stage / "model-admission.json"
        )
        admission_argument = f"{stage / 'model-admission.json'}#sha256={admission_sha}"
        validate_model_admission(
            admission_argument, Path(campaign["model"]["source_model_root"])
        )
        raw_root = stage / "raw"
        raw_root.mkdir(exist_ok=True)
        log_root = stage / "server-logs"
        log_root.mkdir(exist_ok=True)
        request_rows: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        lifecycle_rows: list[dict[str, Any]] = []
        order_rows: list[dict[str, Any]] = []
        engine_order = tuple(engine["name"] for engine in campaign["engines"])
        for profile in campaign["profiles"]:
            (raw_root / profile["name"]).mkdir(exist_ok=True)
        for trial_ordinal in range(campaign["trials_per_profile"]):
            for order_position, engine_name in enumerate(
                latin_square_order(trial_ordinal, engine_order)
            ):
                base_coordinate = (
                    f"trial-{trial_ordinal + 1}-position-"
                    f"{order_position + 1}-{engine_name}"
                )
                recovered = _load_completed_engine_group(
                    raw_root,
                    campaign["profiles"],
                    trial_ordinal,
                    order_position,
                    engines[engine_name],
                )
                if recovered is not None:
                    recovered_requests, recovered_trials, recovered_order = recovered
                    request_rows.extend(recovered_requests)
                    trial_rows.extend(recovered_trials)
                    order_rows.extend(recovered_order)
                    lifecycle_rows.append(
                        {
                            "schema": "lunaflux.qwen3-server-lifecycle-recovery.v1",
                            "engine": engine_name,
                            "coordinate": base_coordinate,
                            "measured_run_complete": True,
                            "measured_profile_count": len(campaign["profiles"]),
                            "warmup_excluded": True,
                            "recovered_from_preserved_raw": True,
                            "live_process_identity_unavailable_after_harness_failure": True,
                        }
                    )
                    continue
                last_error: AdapterError | None = None
                prior_attempts = len(
                    list(log_root.glob(f"{base_coordinate}-attempt-*.stderr.log"))
                )
                for attempt in range(2):
                    attempt_ordinal = prior_attempts + attempt
                    coordinate = f"{base_coordinate}-attempt-{attempt_ordinal + 1}"
                    lifecycle = ServerLifecycle(
                        engines[engine_name],
                        campaign,
                        admission_argument,
                        log_root,
                        coordinate,
                    )
                    completed_profiles = 0
                    engine_request_rows: list[dict[str, Any]] = []
                    engine_trial_rows: list[dict[str, Any]] = []
                    engine_order_rows: list[dict[str, Any]] = []
                    measurement_error: AdapterError | None = None
                    identity: dict[str, Any] = {}
                    try:
                        lifecycle.start()
                        for profile in campaign["profiles"]:
                            cases = _cases_for_trial(workload, profile, trial_ordinal)
                            warm_profile(
                                engines[engine_name],
                                profile,
                                workload,
                                campaign["warmup_rounds_per_profile"],
                                campaign["request_timeout_seconds"],
                                encoder,
                            )
                            rows, trial = run_trial(
                                engines[engine_name],
                                profile,
                                cases,
                                trial_ordinal,
                                order_position,
                                campaign,
                                encoder,
                            )
                            completed_profiles += 1
                            _write_jsonl(
                                raw_root
                                / profile["name"]
                                / (
                                    f"trial-{trial_ordinal + 1}-position-"
                                    f"{order_position + 1}-{engine_name}.jsonl"
                                ),
                                rows,
                            )
                            engine_request_rows.extend(rows)
                            engine_trial_rows.append(trial)
                            engine_order_rows.append(
                                {
                                    "profile": profile["name"],
                                    "trial_ordinal": trial_ordinal,
                                    "order_position": order_position,
                                    "engine": engine_name,
                                }
                            )
                    except AdapterError as error:
                        measurement_error = error
                    try:
                        identity = lifecycle.stop()
                    except AdapterError as error:
                        if measurement_error is None:
                            measurement_error = error
                    if measurement_error is not None:
                        last_error = measurement_error
                        if attempt == 0:
                            continue
                        raise AdapterError(
                            f"coordinate {base_coordinate} failed twice: {last_error}"
                        ) from last_error
                    if completed_profiles != len(campaign["profiles"]):
                        raise AdapterError(
                            f"coordinate {coordinate} did not complete measurement"
                        )
                    identity["warmup_excluded"] = True
                    identity["measured_run_complete"] = True
                    identity["measured_profile_count"] = completed_profiles
                    identity["attempt_ordinal"] = attempt_ordinal
                    request_rows.extend(engine_request_rows)
                    trial_rows.extend(engine_trial_rows)
                    order_rows.extend(engine_order_rows)
                    lifecycle_rows.append(identity)
                    break
        summaries = summarize(request_rows, trial_rows)
        correctness = correctness_join(request_rows)
        request_measurements_complete = bool(request_rows) and all(
            row["ok"] and row["output_count_consistent"]
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
        _write_jsonl(stage / "lifecycle.jsonl", lifecycle_rows)
        _write_jsonl(stage / "summary.jsonl", summaries)
        _write_jsonl(stage / "correctness.jsonl", correctness)
        _write_json(
            stage / "manifest.json",
            {
                "schema": "lunaflux.qwen3-comparison-result.v1",
                "campaign_sha256": campaign_sha,
                "workload_sha256": workload_sha,
                "model_id": "Qwen3-0.6B",
                "engine_order": list(engine_order),
                "execution_order": order_rows,
                "warmup_excluded": True,
                "startup_time_excluded": True,
                "persistent_servers_required": True,
                "server_lifecycle": "one-engine-per-target-gpu-trial",
                "model_inventory_full_scan_count": 1,
                "model_admission_sha256": admission_sha,
                "lunaflux_authenticated_max_concurrency": capacity["max_concurrency"],
                "lunaflux_protocol": "diagnostic-token-id-sse-bridge-only",
                "summary_policy": "median,p95,deterministic-bootstrap-median-ci95",
                "winner_assumption": "none",
                "speed_comparison_valid": speed_comparison_valid,
                "correctness_failure_invalidates_speed_comparison": True,
                "request_measurements_complete": request_measurements_complete,
                "token_timing_exact_request_count": sum(
                    1 for row in request_rows if row["token_timing_exact"]
                ),
                "token_timing_inexact_request_count": sum(
                    1 for row in request_rows if not row["token_timing_exact"]
                ),
                "ollama_inference_rule": "forbidden: no Ollama result may be inferred",
                "correctness_exact_match_count": sum(
                    1 for row in correctness if row["exact_greedy_match"]
                ),
                "correctness_mismatch_or_incomplete_count": sum(
                    1 for row in correctness if not row["exact_greedy_match"]
                ),
            },
        )
        failure_marker = stage / "FAILURE.json"
        if failure_marker.exists():
            failure_marker.unlink()
        for path in sorted(stage.rglob("*"), reverse=True):
            os.chmod(path, 0o555 if path.is_dir() else 0o444)
        os.chmod(stage, 0o555)
        os.replace(stage, output)
        published = True
    except BaseException as error:
        if stage.exists():
            try:
                _write_json(
                    stage / "FAILURE.json",
                    {
                        "schema": "lunaflux.qwen3-comparison-failure.v1",
                        "error_type": type(error).__name__,
                        "error_message": str(error)[:4096],
                    },
                )
                failure = _failure_output_path(output)
                os.replace(stage, failure)
                raise AdapterError(
                    f"campaign failed; diagnostics retained at {failure}: {error}"
                ) from error
            except AdapterError:
                raise
            except BaseException:
                pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True, help="ABSOLUTE_JSON#sha256=HEX")
    parser.add_argument("--workload", required=True, help="ABSOLUTE_JSONL#sha256=HEX")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    arguments = parser.parse_args()
    try:
        _, campaign_bytes, campaign_sha = read_digest_suffixed(arguments.campaign, "campaign")
        _, workload_bytes, workload_sha = read_digest_suffixed(arguments.workload, "workload")
        campaign = validate_campaign(json.loads(campaign_bytes))
        workload = load_workload(workload_bytes)
        run_campaign(
            campaign,
            workload,
            campaign_sha,
            workload_sha,
            arguments.output,
            arguments.resume,
        )
    except (ContractError, AdapterError, json.JSONDecodeError) as error:
        print(f"Qwen3 comparison rejected: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
