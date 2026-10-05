# Explicit exp2 serving propagation — 2026-10-05

## Scope and current status

The [dual-Spark kernel comparison](BENCHMARK_AKO_LONG_EXPONENTIAL_2026-10-05.md)
found a 4.37–5.36% median selected-kernel gain. This follow-up tests whether that
gain survives real serving. End-to-end results are pending; the kernel gain must
not be reported as whole-engine gain.

`.178` and `.179` are available concurrently. The preceding sweep split paired
cells across both GPUs. This follow-up separates long-history memory checking
on `.178` from route calibration/serving on `.179`, one GPU workload per device.
No extra agents or production defaults are introduced.

The serving experiment reuses the pinned Qwen3-0.6B BF16 worker
`dd4e8e2b5ddfb66cc6c901cf96b5ca14f12f9138841644c4fc2c4b4480928622`.
Only baseline prefill changes to the existing explicit `approx-base2-f32-v1`
symbol `lunaflux_attention_prefill_tile_compiler_exp2_v1`, cubin
`66881bc0055ce4c4340e6001e7746d0abd64a8b7f9a02b741e6b87463f0e2922`.
The separate wide-prefill module remains strict. The other eight module hashes,
symbols and launch geometries are checked unchanged. Approximation is declared
in the bundle, never inferred from a model name or requested implicitly.

The new route scope is
`86ea2cd5831ad63f532332b46b88d7a81d56d8ad74b82ee0379606ceaa6dd60d`.
Six fresh paired route cells completed: three prefill, two mixed, one exact
eight-row-envelope C2 decode chain. They select ordinary prefill/mixed and
partitioned pure C2 decode. Old timings were not relabeled into the new scope.
After binding failures, the completed records are copied unchanged into a new
output directory rather than rerun or overwritten.

## Finite serving comparison

Four fresh starts, ABBA: strict, approximate, approximate, strict. The strict
baseline already contains the previous C2 decode route improvement, so it is
not credited again. Each start uses the unchanged varied token generator,
one warmup and three measured trials per cell:

- 16,384 input tokens, C1, 64 requested output tokens.
- 32,512 input tokens, C1, 64 requested output tokens.
- 32,512 input tokens per request, C2, 64 requested output tokens per request.

Retain completion time, TTFT, TPOT, throughput and every per-request output
token vector, including repeated-start differences. Non-bitwise BF16 probe
acceptance is not model-quality or deterministic-token parity admission.
No fresh vLLM/SGLang result or current cross-framework gap is claimed here.

Serving children retain MemoryMax64G, MemorySwapMax0 and continuous 32GiB
MemAvailable reserve monitoring. The outer automation's smaller CPU limit is
not incorrectly claimed to bound sibling serving units. Route calibration
uses MemoryMax16G, zero swap and a 600-second limit; its observed peak was
854.5MiB. Bundle preparation uses MemoryMax8G, zero swap and a 300-second limit.

## Completed long-history memory gate

`.178`: Q2048, active rows2, history28672; actual runtime bucket2048/rows32,
grid63×16×1. Compute Sanitizer memcheck reports zero errors. The same probe
reports maxabs0.000488281 and FP64 oracle maxabs0.000377474, below ceiling0.003.
This is a long-history memory/numerical probe, not full serving/model quality.

Remote output:
`/home/wlc003s/lunaflux-ako-long-exp-20261005.YDjCs0sP/long-memcheck`.
Downloaded archive:
`/tmp/lunaflux-ako-exp2-serving-20261005.VLE9f1Iv/long-memcheck178.tar.gz`.
SHA-256:
`876a3671fb4b9c88286a897cd97849b1a7ce11aafb41fb39e89656ba121ba6e7`.
Remote/local hashes match and all seven manifest entries verify.

## Preserved preparation failures

Experiment root:
`/home/wlc004s/lunaflux-ako-exp2-serving-20261005.t6QFbUwT`.

1. `facade`: export succeeds, nested materialization cannot find `moon` under
   systemd's noninteractive PATH. The offline adapter now pins the frozen
   toolchain PATH and source working directory, as existing campaign helpers do.
   `facade-v2` completes preparation, peak1.2GiB, no swap.
2. `facade-v2/routes`: route rebind/materialization/preflight succeed, but
   capacity receipt creation rejects the facade's symlinked runtime path as
   noncanonical. The route helper now resolves the frozen source path before
   rendering executable identities. No validation is weakened. The failed
   directory remains preserved; the replacement binding uses `routes-bound-v2`.

Changes are offline MoonBit automation only. Native warning-denied script
checks and affected tests pass. `moon info` does not accept these standalone
scripts as packages in this toolchain; no production public API is changed.
The compiler's semantic/numeric → schedule/selection → ownership/effects →
device-lowering boundaries remain unchanged. AKO determines the paired finite
budget, explicit approximation and end-to-end propagation requirement.
