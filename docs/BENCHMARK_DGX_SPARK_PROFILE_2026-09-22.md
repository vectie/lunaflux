# DGX Spark long-sequence diagnosis — 2026-09-22

## Status

Paired ordinary measurements and Nsight Systems kernel traces are collected.
Nsight Compute hardware-counter collection is **not complete**. Its vLLM kernel
replay exhausted available unified memory; this failed diagnostic is excluded
from all performance results. Recovery commands have been issued; host recovery
and restoration of the paused GLM container still require confirmation.

Same Qwen3-0.6B BF16 model and frozen runtime as
[the three-engine benchmark](BENCHMARK_DGX_SPARK_2026-09-22.md). Engines ran
sequentially. This follow-up uses one warmup and one measured batch per cell,
so these are reproduction samples, not new three-run medians.

## Ordinary serving reproduction

All rows generate 64 tokens. Time includes the entire batch completing.

| Input tokens | Concurrency | LunaFlux ms | vLLM ms | LF / vLLM |
| --- | ---: | ---: | ---: | ---: |
| 512 | 1 | 486 | 561 | 0.87 |
| 512 | 8 | 979 | 648 | 1.51 |
| 512 | 16 | 1323 | 881 | 1.50 |
| 4096 | 1 | 842 | 765 | 1.10 |
| 4096 | 8 | 3701 | 2233 | 1.66 |
| 4096 | 16 | 6934 | 4124 | 1.68 |

For 4096/C16, mean per-request TTFT is 2352.88 vs 1113.56 ms. Mean
time after the first token is 4394.25 vs 2743.06 ms. The latter is **not pure
decode**: other requests may still be prefilling. Do not add these averages to
derive batch wall time.

## Selected-kernel timeline

Separate Nsight Systems captures use CUDA graph-node tracing. Profiled LF
4096/C16 takes 6965 ms, versus 6934 ms without tracing. Selected kernel durations
sum to 6832.18 ms over a 6946.68 ms first-to-last-kernel span. This strongly
points toward GPU work rather than a large host-idle bubble; a precise idle
percentage still requires interval-union accounting.

| LF selected family, 4096/C16 | Calls | Summed GPU ms |
| --- | ---: | ---: |
| Decode attention | 2184 | 2284.75 |
| Full QKV + QKNorm + RoPE + KV-write | 2688 | 2018.42 |
| Prefill attention | 896 | 1017.90 |
| Main gate/up | 939 | 509.12 |
| Main down | 939 | 290.29 |
| Main output projection | 939 | 195.76 |
| Segmented greedy head | 83 | 116.04 |

Rows8/rows16 variants and smaller families are additional, not included in the
main projection rows above. The QKV fusion divides into 924 launches with
grid 128x32, averaging 1964.53 us (1815.23 ms total), and 1764 launches with
grid 1x32, averaging 115.19 us (203.20 ms). Both use 128 threads, 56 registers
per thread and 8704 bytes static shared memory. These are static launch
resources, **not measured occupancy or stall counters**.

In the final vLLM 4096/C16 trace batch, FlashAttention kernels sum to 2617.96 ms.
Its 896 QKV projection launches with grid 128x2 average 227.68 us (204.01 ms).
The projection alone is **not equivalent work** to LF's full ingress fusion:
QKNorm, RoPE and KV-write must be added before claiming a chain-level speedup.
The first forward's kernel order identifies this projection; the generic
`Kernel2` name must not be used to classify every launch as QKV.

The immediate targets are therefore the **selected full-ingress prefill
implementation** and attention. Fusion count, occupancy, or generic claims
about memory bandwidth are not yet a causal hardware diagnosis.

## Profiler startup repair (diagnostic only)

Two independent capture problems were identified:

1. Nsight's inherited descriptors collide with LF's optional startup channels:
   FD 6 inference credential and FD 7 promotion-verifier key. Unexpanded startup
   errors initially appeared as SIGABRT. A temporary parent prints the errors
   and reserves different optional-channel descriptors for this benchmark,
   which supplies neither credential. Production validation is unchanged.
2. The worker spawn cleans its environment and uses descriptor-based execution.
   A disposable parent forwards profiler injection to the unchanged worker and
   preserves diagnostic stderr. Systems successfully captures worker kernels.
   Compute initially captured only the parent. An additional diagnostic change
   to execute the same opened file through `/proc/self/fd/5` has been built but
   its Compute capture is not yet verified.

The model, deployed worker and kernel artifacts remain unchanged. These
diagnostic parent changes are not a production deployment or a general fix to
the credential protocol. A production-safe solution needs explicit descriptor
ownership/hand-off semantics, not silently accepting malformed credentials.

Administrator authentication succeeded. No global driver profiling permission
was changed. vLLM Compute ran in a separate diagnostic container with
SYS_ADMIN; its original 0.5 memory-utilization configuration plus profiler
replay exhausted Spark's unified memory (121 GiB used, 274 MiB available,
3.6 GiB swap used). Subsequent counters must use a bounded, single-request KV
budget and must not be confused with the original end-to-end configuration.

## Artifacts

Remote directory: `/home/wlc004s/lunaflux-spark-profile.MDNOHx`.

- `plain-lunaflux-ready`, `plain-vllm-ready`: ordinary reproduction requests.
- `profile-lunaflux-marked`: capture requests with epoch-nanosecond boundaries.
- `fixed3-trace.nsys-rep`, `fixed3-trace.sqlite`: successful LF capture.
- `vllm-trace.nsys-rep`, `vllm-trace.sqlite`: successful vLLM capture.
- `lunaflux-telemetry.csv`, `vllm-telemetry.csv`: 100 ms NVML samples; not
  per-instruction hardware counters.
- `enable-profile-parent.mbtx`: disposable-parent modifications, no production
  source mutation.

LF final measured batch bounds relative to its trace epoch:
87946489789–94917161251 ns. vLLM's final batch was located from the last request
group and the preceding >100 ms kernel gap near 99.253 seconds; it lacks the
explicit per-batch epoch markers added to the subsequent LF capture. Recollect
marked vLLM boundaries before fine-grained cross-engine step attribution.

No instruction-level cause, tensor utilization, DRAM traffic or bank-conflict
claim is made without successful Compute counters. Remote artifacts have not
yet been downloaded/sealed for this follow-up.
