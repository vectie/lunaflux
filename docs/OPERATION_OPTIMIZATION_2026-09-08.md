# Operation optimization follow-up

This work follows the measured gaps in `OPERATION_PROFILE_2026-09-08.md`.
Changes are evaluated independently before a combined Qwen serving comparison.
Compiler policies remain shape/capability based; CUDA mapping stays in lowering.

## Residual launch bounds

Production residual/RMSNorm V2 is row-local: one workgroup writes one token's
residual and normalized values. Previously its prepared graph retained the
1024-token grid at C1/C8. Startup planning now carries the live-token bucket
bound for this admitted ABI into ordinary, wide-prefill, partitioned-prefill,
and split-decode graph variants. Unknown/diagnostic ABIs retain fixed geometry.
No arithmetic, synchronization, kernel arguments, or live-row memory effects
change. The full-envelope fallback is preserved.

The new regressions cover C1/C8 through the full envelope, ABI exclusion,
undersized source envelopes, and rule remapping across attention expansion.
The initial device-step native suite passed 174/174. The combined physical
serving comparison below includes this change; its isolated contribution is
not inferred from that combined gain.

## Combined serving benchmark

Runtime commit `8f3708b` includes all five operation tracks. The control uses
the previous combined kernel bundle and fixed split-decode worker `57e7417`.
Both use Qwen3-0.6B BF16, the same model bytes and input token IDs, a 1024-token
execution profile, and 1024/1024 scheduler budget/prefill chunk. Tests run on
the RTX 5060 Ti alone, without simultaneous GPU jobs or profiler instrumentation.
No production deployment was changed.

LunaFlux was run control→optimized and optimized→control. Each cell has one
excluded warmup and two measured trials in each order: four timed trials per
LunaFlux arm. The table averages their output-token throughput. Fresh vLLM
0.24.0/FA2 and SGLang 0.5.2/FlashInfer runs each have one excluded warmup and
two measured trials. These are fixed installed baselines, not a claim about
the latest upstream releases. Numbers are tok/s, not pure decode-kernel rates.

| Input/output tokens | C | Previous LF | Optimized LF | LF gain | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 59/256 | 1 | 240.38 | 248.24 | +3.27% | 278.56 | 264.88 |
| 59/256 | 8 | 1169.14 | 1411.93 | +20.77% | 1850.88 | 1796.18 |
| 128/128 | 1 | 231.47 | 239.93 | +3.65% | 267.51 | 258.06 |
| 128/128 | 8 | 1080.46 | 1283.22 | +18.77% | 1741.50 | 1651.62 |
| 512/64 | 1 | 188.10 | 194.83 | +3.58% | 241.06 | 242.89 |
| 512/64 | 8 | 574.64 | 646.47 | +12.50% | 1182.45 | 1136.53 |
| 1528/32 | 1 | 91.23 | 97.94 | +7.35% | 175.85 | 174.91 |
| 1528/32 | 8 | 139.59 | 159.83 | +14.50% | 444.45 | 419.67 |

Optimized throughput between the two orders differs by no more than 0.51%
in these samples. All 144 timed optimized request sequences match their paired
LunaFlux control exactly, and all requested output-token counts are correct.
Cross-framework token sequences are **not uniformly identical**; this measures
equal token-count workloads, not independently established numerical or quality
equivalence. No confidence interval or workload-wide aggregate speedup is claimed.

For short C8, vLLM/SGLang are now 1.31×/1.27× faster than LunaFlux. For long
C8 they remain 2.78×/2.63× faster. Thus this iteration improves every measured
vector but does not achieve performance parity, especially for long prefill.

## Isolated operation results

These same-run GPU microbenchmarks are separate from the serving results.
Their speedups cannot be added or multiplied to predict request throughput.

| Operation / workload | Previous µs | New µs | Speedup |
| --- | ---: | ---: | ---: |
| Output projection, C8, rotating weights | 25.472 | 15.336 | 1.66× |
| QKV projection, C8, rotating weights | 56.08 | 49.346 | 1.14× |
| Vocabulary head, C8 | 1523.005 | 767.949 | 1.98× |
| Greedy sampling, C8, both new stages included | 70.196 | 8.606 | 8.16× |

The functional compiler changes work distribution and operand lifetimes,
not model-family semantics: the head uses an immutable K128 storage-strip
plan around the existing ordered K16 matrix folds; projection choices are
device/shape-scoped offline measurements; sampling is an associative summary
reduction with deterministic tie handling. The executor derives row-local
residual launch bounds at startup. NVIDIA workgroup, copy and matrix details
remain in CUDA lowering. No request-path JIT or new token-step allocation is
introduced.

Attention now preserves incomparable transfer schedules for measurement and
uses the compiler-selected schedule rather than forcing the widest query tile.
The deep partition policy excludes already-parallel query tile distributions.
However, async-copy schedule316 was 20–24% slower than synchronous312 on the
measured Qwen geometry. It remains available but is deliberately not selected.
This is not a claim that the long-prefill attention bottleneck is solved.

See `HEAD_STRIP_PERFORMANCE_2026-09-08.md`, `QKV_WORK_DISTRIBUTION_2026-09-08.md`,
`QWEN_ATTENTION_TRANSFER_2026-09-08.md`, and `SEGMENTED_GREEDY_2026-09-08.md`
for operation methodology, correctness and sanitizer results.

## Reproduction

The checked-in measured tuning snapshot is
`benchmarks/operation_optimization_20260908/projection-tuning.v1`. Head bound256
records measure eight actual selected rows inside a separately compiled
256-token-capable kernel; they are not 256-row latency measurements. The earlier
MLP record is retained unchanged. No new C1 head strategy is forced.

Source archive SHA-256:
`22bf2c83b283c3b4f21548db1a6624d0a08b4dbda5c4c8b252c4ca94598fb65a`.
Remote source: `/dev/shm/lunaflux-five-ops-20260908-r2/source`.
Remote runtime: `/dev/shm/lunaflux-five-ops-runtime-20260908-r2`.
Local result download: `/private/tmp/lunaflux-five-ops-20260908-results-r1`.
The download contains the exact source archive, build/runtime/benchmark scripts,
all six raw measurement arms, the summary, and an independently checked archive
checksum. Results are preserved separately from subsequent diagnostic profiles.
Result archive SHA-256:
`4b9f5947dfdad254e896ed833a6fe894414ba1eda8ce0faae52133dce7b8998a`.
The compact machine-readable table is checked in alongside the tuning snapshot
as `benchmarks/operation_optimization_20260908/summary.json`.
