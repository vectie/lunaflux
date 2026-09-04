#!/usr/bin/env python3
"""Small dependency-free benchmark for the LunaFlux token-ID endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import time
import urllib.request


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("empty distribution")
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def invoke(
    url: str, body: bytes, timeout: float
) -> tuple[float, tuple[int, ...], bool, str]:
    request = urllib.request.Request(
        url,
        data=body,
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
        method="POST",
    )
    started = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = response.read().decode("utf-8")
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
    tokens = []
    terminal_events = 0
    done_events = 0
    for line in payload.splitlines():
        if line == "data: [DONE]":
            done_events += 1
            continue
        if not line.startswith("data: {"):
            continue
        event = json.loads(line[6:])
        if event.get("schema") == "lunaflux.benchmark-token.v1":
            tokens.append(int(event["token_id"]))
        elif event.get("schema") == "lunaflux.benchmark-terminal.v1":
            terminal_events += 1
    complete = terminal_events == 1 and done_events == 1
    return elapsed_ms, tuple(tokens), complete, payload


def token_summary(
    actual: tuple[int, ...], expected: tuple[int, ...]
) -> dict[str, object]:
    shared = min(len(actual), len(expected))
    divergence = next(
        (index for index in range(shared) if actual[index] != expected[index]),
        shared if len(actual) != len(expected) else None,
    )
    encoded = ",".join(str(token) for token in actual).encode("ascii")
    window_start = 0 if divergence is None else max(0, divergence - 3)
    window_end = min(len(actual), window_start + 8)
    return {
        "token_count": len(actual),
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "first_divergence": divergence,
        "divergence_window": actual[window_start:window_end],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("request_json")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--warmups", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--expected", default="92648,4532")
    parser.add_argument(
        "--learn-expected",
        action="store_true",
        help="learn deterministic output token IDs from one excluded request",
    )
    arguments = parser.parse_args()
    if arguments.concurrency <= 0 or arguments.requests <= 0 or arguments.warmups < 0:
        parser.error("concurrency/requests must be positive and warmups non-negative")
    body = open(arguments.request_json, "rb").read()
    expected = tuple(int(value) for value in arguments.expected.split(",") if value)
    if arguments.learn_expected:
        _, expected, complete, payload = invoke(arguments.url, body, arguments.timeout)
        if not complete:
            raise RuntimeError(
                "expected-output learning request is incomplete: " f"payload={payload!r}"
            )
    for _ in range(arguments.warmups):
        _, tokens, complete, payload = invoke(arguments.url, body, arguments.timeout)
        if tokens != expected or not complete:
            raise RuntimeError(
                "warmup output differs or is incomplete: "
                f"tokens={tokens!r}; complete={complete!r}; payload={payload!r}"
            )
    started = time.perf_counter_ns()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=arguments.concurrency
    ) as executor:
        futures = [
            executor.submit(invoke, arguments.url, body, arguments.timeout)
            for _ in range(arguments.requests)
        ]
        observations = [future.result() for future in futures]
    wall_seconds = (time.perf_counter_ns() - started) / 1_000_000_000.0
    latencies = [elapsed for elapsed, _, _, _ in observations]
    mismatches = [
        {
            "index": index,
            **token_summary(tokens, expected),
            "complete": complete,
        }
        for index, (_, tokens, complete, _) in enumerate(observations)
        if tokens != expected or not complete
    ]
    output_tokens = sum(len(tokens) for _, tokens, _, _ in observations)
    print(
        json.dumps(
            {
                "concurrency": arguments.concurrency,
                "requests": arguments.requests,
                "output_tokens": output_tokens,
                "correct_requests": arguments.requests - len(mismatches),
                "incorrect_requests": len(mismatches),
                "mismatches": mismatches[:8],
                "wall_seconds": wall_seconds,
                "requests_per_second": arguments.requests / wall_seconds,
                "output_tokens_per_second": output_tokens / wall_seconds,
                "latency_ms": {
                    "mean": sum(latencies) / len(latencies),
                    "min": min(latencies),
                    "p50": percentile(latencies, 0.50),
                    "p95": percentile(latencies, 0.95),
                    "p99": percentile(latencies, 0.99),
                    "max": max(latencies),
                },
                "expected": token_summary(expected, expected),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
