"""Loopback streaming adapters used only by the external benchmark harness."""

from __future__ import annotations

import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class GenerationObservation:
    output_text: str
    output_token_ids: list[int]
    token_timestamps_ns: list[int]
    terminal_ns: int
    http_status: int
    server_reported_output_tokens: int | None
    stream_chunk_count: int


class AdapterError(RuntimeError):
    pass


def check_health(url: str, timeout_seconds: float = 5.0) -> None:
    request = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            if response.status < 200 or response.status >= 300:
                raise AdapterError(f"health status {response.status}")
    except (OSError, urllib.error.URLError) as error:
        raise AdapterError(f"persistent server is not healthy: {url}") from error


def _event_payloads(response: Any):
    for raw_line in response:
        line = raw_line.decode("utf-8").strip()
        if not line or line.startswith(":"):
            continue
        if not line.startswith("data:"):
            raise AdapterError("stream contains a non-SSE data line")
        payload = line[5:].strip()
        if payload == "[DONE]":
            return
        try:
            value = json.loads(payload)
        except json.JSONDecodeError as error:
            raise AdapterError("stream contains malformed JSON") from error
        if not isinstance(value, dict):
            raise AdapterError("stream event is not an object")
        if "error" in value:
            raise AdapterError("stream reported an engine error")
        yield value
    raise AdapterError("stream ended without the terminal [DONE] event")


def _post_stream(url: str, body: dict[str, Any], timeout_seconds: float):
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
        method="POST",
    )
    return urllib.request.urlopen(request, timeout=timeout_seconds)


def generate(
    engine: dict[str, Any],
    input_token_ids: list[int],
    output_tokens: int,
    timeout_seconds: float,
    encode: Callable[[str], list[int]],
) -> GenerationObservation:
    adapter = engine["adapter"]
    if adapter == "vllm-completions-sse-v1":
        return _generate_vllm(engine, input_token_ids, output_tokens, timeout_seconds, encode)
    if adapter == "sglang-generate-sse-v1":
        return _generate_sglang(engine, input_token_ids, output_tokens, timeout_seconds, encode)
    if adapter == "lunaflux-token-ids-sse-v1":
        return _generate_lunaflux(engine, input_token_ids, output_tokens, timeout_seconds)
    raise AdapterError("unknown engine adapter")


def _generate_vllm(
    engine: dict[str, Any],
    input_ids: list[int],
    output_limit: int,
    timeout: float,
    _encode: Callable[[str], list[int]],
) -> GenerationObservation:
    body = {
        "model": engine["model_alias"],
        "prompt": input_ids,
        "max_tokens": output_limit,
        "temperature": 0,
        "top_p": 1,
        "n": 1,
        "seed": 0,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "return_token_ids": True,
    }
    output = ""
    token_ids: list[int] = []
    timestamps: list[int] = []
    chunks = 0
    reported: int | None = None
    with _post_stream(engine["endpoint"], body, timeout) as response:
        status = response.status
        if status < 200 or status >= 300:
            raise AdapterError(f"vLLM returned HTTP {status}")
        for event in _event_payloads(response):
            usage = event.get("usage")
            if isinstance(usage, dict) and isinstance(usage.get("completion_tokens"), int):
                reported = usage["completion_tokens"]
            choices = event.get("choices", [])
            if not choices:
                continue
            if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                raise AdapterError("vLLM stream choices are ambiguous")
            fragment = choices[0].get("text", "")
            if not isinstance(fragment, str):
                raise AdapterError("vLLM stream text is invalid")
            raw_token_ids = choices[0].get("token_ids")
            if raw_token_ids is None and not fragment:
                raw_token_ids = []
            if not isinstance(raw_token_ids, list) or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                for token_id in raw_token_ids
            ):
                raise AdapterError("vLLM stream omitted exact output token IDs")
            if raw_token_ids:
                chunks += 1
                timestamps.extend([time.perf_counter_ns()] * len(raw_token_ids))
                token_ids.extend(raw_token_ids)
            output += fragment
        terminal = time.perf_counter_ns()
    return GenerationObservation(output, token_ids, timestamps, terminal, status, reported, chunks)


def _generate_sglang(
    engine: dict[str, Any],
    input_ids: list[int],
    output_limit: int,
    timeout: float,
    _encode: Callable[[str], list[int]],
) -> GenerationObservation:
    body = {
        "input_ids": input_ids,
        "sampling_params": {
            "max_new_tokens": output_limit,
            "temperature": 0,
            "top_p": 1,
            "sampling_seed": 0,
            "ignore_eos": True,
        },
        "stream": True,
    }
    output = ""
    token_ids: list[int] = []
    timestamps: list[int] = []
    chunks = 0
    reported: int | None = None
    with _post_stream(engine["endpoint"], body, timeout) as response:
        status = response.status
        if status < 200 or status >= 300:
            raise AdapterError(f"SGLang returned HTTP {status}")
        for event in _event_payloads(response):
            raw_token_ids = event.get("output_ids")
            if not isinstance(raw_token_ids, list) or any(
                not isinstance(token_id, int)
                or isinstance(token_id, bool)
                or token_id < 0
                for token_id in raw_token_ids
            ):
                raise AdapterError("SGLang stream omitted exact output token IDs")
            if raw_token_ids[: len(token_ids)] != token_ids:
                raise AdapterError("SGLang output-token stream is not cumulative")
            new_token_ids = raw_token_ids[len(token_ids) :]
            token_ids = list(raw_token_ids)
            if new_token_ids:
                chunks += 1
                timestamps.extend([time.perf_counter_ns()] * len(new_token_ids))
            meta = event.get("meta_info")
            if isinstance(meta, dict) and isinstance(meta.get("completion_tokens"), int):
                reported = meta["completion_tokens"]
        terminal = time.perf_counter_ns()
    return GenerationObservation(output, token_ids, timestamps, terminal, status, reported, chunks)


def _generate_lunaflux(
    engine: dict[str, Any],
    input_ids: list[int],
    output_limit: int,
    timeout: float,
) -> GenerationObservation:
    body = {
        "schema": "lunaflux.benchmark-token-ids-request.v1",
        "model": engine["model_alias"],
        "input_token_ids": input_ids,
        "max_output_tokens": output_limit,
        "sampling": {
            "mode": "greedy",
            "temperature": 0,
            "top_p": 1,
            "seed": 0,
            "ignore_eos": True,
        },
        "stream": True,
    }
    output_fragments: list[str] = []
    token_ids: list[int] = []
    timestamps: list[int] = []
    chunks = 0
    terminal_seen = False
    with _post_stream(engine["endpoint"], body, timeout) as response:
        status = response.status
        if status < 200 or status >= 300:
            raise AdapterError(f"LunaFlux returned HTTP {status}")
        for event in _event_payloads(response):
            if event.get("schema") == "lunaflux.benchmark-token.v1":
                if terminal_seen:
                    raise AdapterError("LunaFlux emitted a token after its terminal event")
                token_id = event.get("token_id")
                text = event.get("text")
                if not isinstance(token_id, int) or isinstance(token_id, bool) or token_id < 0:
                    raise AdapterError("LunaFlux token ID is invalid")
                if not isinstance(text, str):
                    raise AdapterError("LunaFlux token text is invalid")
                token_ids.append(token_id)
                timestamps.append(time.perf_counter_ns())
                output_fragments.append(text)
                chunks += 1
            elif event.get("schema") == "lunaflux.benchmark-terminal.v1":
                if terminal_seen:
                    raise AdapterError("LunaFlux emitted duplicate terminal events")
                text = event.get("text")
                if not isinstance(text, str):
                    raise AdapterError("LunaFlux terminal text is invalid")
                output_fragments.append(text)
                terminal_seen = True
            else:
                raise AdapterError("LunaFlux benchmark stream event is invalid")
        terminal = time.perf_counter_ns()
    if not terminal_seen:
        raise AdapterError("LunaFlux stream omitted its terminal event")
    return GenerationObservation(
        "".join(output_fragments), token_ids, timestamps, terminal, status, len(token_ids), chunks
    )


class GpuMemorySampler:
    """Samples whole-device memory; it deliberately makes no per-process attribution."""

    def __init__(self, executable: str, uuid: str, interval_millis: int):
        self.executable = executable
        self.uuid = uuid
        self.interval_seconds = interval_millis / 1000.0
        self.values: list[int] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> int:
        completed = subprocess.run(
            [
                self.executable,
                "--id=" + self.uuid,
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5,
        )
        lines = completed.stdout.strip().splitlines()
        if len(lines) != 1 or not lines[0].strip().isdigit():
            raise AdapterError("GPU memory sample is not canonical")
        return int(lines[0].strip())

    def start(self) -> int:
        baseline = self._sample()
        self.values.append(baseline)

        def loop() -> None:
            while not self._stop.wait(self.interval_seconds):
                try:
                    self.values.append(self._sample())
                except Exception as error:  # retained and failed after trial cleanup
                    self.error = type(error).__name__ + ": " + str(error)
                    self._stop.set()

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        return baseline

    def finish(self) -> tuple[int, int]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.error is not None:
            raise AdapterError("GPU memory sampling failed: " + self.error)
        self.values.append(self._sample())
        return min(self.values), max(self.values)
