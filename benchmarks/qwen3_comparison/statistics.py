"""Deterministic descriptive summaries; this module never declares a winner."""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Any, Iterable


def percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    coordinate = (len(ordered) - 1) * quantile
    lower = math.floor(coordinate)
    upper = math.ceil(coordinate)
    fraction = coordinate - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def median(values: list[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def median_ci95(values: list[float], label: str, resamples: int = 2000) -> list[float] | None:
    if not values:
        return None
    seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "big")
    generator = random.Random(seed)
    medians = []
    for _ in range(resamples):
        sample = [values[generator.randrange(len(values))] for _ in values]
        medians.append(median(sample))
    return [percentile(medians, 0.025), percentile(medians, 0.975)]


def distribution(values: Iterable[float | int | None], label: str) -> dict[str, Any]:
    admitted = [float(value) for value in values if value is not None]
    return {
        "count": len(admitted),
        "median": median(admitted) if admitted else None,
        "p95": percentile(admitted, 0.95),
        "median_ci95": median_ci95(admitted, label),
    }


def summarize(
    request_rows: list[dict[str, Any]],
    trial_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    requests: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    trials: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in request_rows:
        requests[(row["engine"], row["profile"])].append(row)
    for row in trial_rows:
        trials[(row["engine"], row["profile"])].append(row)
    result = []
    for engine, profile in sorted(requests):
        rows = requests[(engine, profile)]
        successful = [row for row in rows if row["ok"]]
        trial_set = trials[(engine, profile)]
        prefix = f"{engine}/{profile}"
        result.append(
            {
                "schema": "lunaflux.qwen3-comparison-summary.v1",
                "engine": engine,
                "profile": profile,
                "request_count": len(rows),
                "success_count": len(successful),
                "error_count": len(rows) - len(successful),
                "error_rate": (len(rows) - len(successful)) / len(rows),
                "ttft_millis": distribution(
                    (row["ttft_millis"] for row in successful), f"{prefix}/ttft"
                ),
                "inter_token_latency_millis": distribution(
                    (
                        interval
                        for row in successful
                        for interval in row["inter_token_latency_millis"]
                    ),
                    f"{prefix}/itl",
                ),
                "e2e_millis": distribution(
                    (row["e2e_millis"] for row in successful), f"{prefix}/e2e"
                ),
                "request_throughput_per_second": distribution(
                    (row["request_throughput_per_second"] for row in trial_set),
                    f"{prefix}/request-throughput",
                ),
                "output_token_throughput_per_second": distribution(
                    (row["output_token_throughput_per_second"] for row in trial_set),
                    f"{prefix}/token-throughput",
                ),
                "gpu_memory_used_peak_mib": distribution(
                    (row["gpu_memory_used_peak_mib"] for row in trial_set),
                    f"{prefix}/gpu-memory",
                ),
                "gpu_memory_used_delta_peak_mib": distribution(
                    (row["gpu_memory_used_delta_peak_mib"] for row in trial_set),
                    f"{prefix}/gpu-memory-delta",
                ),
            }
        )
    return result


def correctness_join(request_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    joined: dict[tuple[str, int, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in request_rows:
        coordinate = (row["profile"], row["trial_ordinal"], row["request_ordinal"])
        joined[coordinate][row["engine"]] = row
    report = []
    for coordinate in sorted(joined):
        by_engine = joined[coordinate]
        hashes = {
            engine: row["output_token_ids_sha256"]
            for engine, row in by_engine.items()
            if row["ok"]
        }
        complete = set(by_engine) == {"lunaflux", "vllm", "sglang"}
        exact_match = complete and len(hashes) == 3 and len(set(hashes.values())) == 1
        report.append(
            {
                "schema": "lunaflux.qwen3-correctness-join.v1",
                "profile": coordinate[0],
                "trial_ordinal": coordinate[1],
                "request_ordinal": coordinate[2],
                "case_ids": {engine: row["case_id"] for engine, row in sorted(by_engine.items())},
                "output_token_ids_sha256": hashes,
                "complete": complete,
                "exact_greedy_match": exact_match,
            }
        )
    return report
