# Fresh four-engine benchmark and bottleneck comparison — 2026-09-06

This is a diagnostic comparison, not an optimization or production deployment.
The largest measured gaps against vLLM/SGLang are small-batch matrix execution
and long-prefill attention. Single-request LM-head execution is already close;
the same operation at C8 is not. Adding compiler passes indiscriminately is not
the remedy: physical schedules must improve for the shapes actually executed.

## Configuration and measurement boundary

- One RTX 5060 Ti, CUDA UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`,
  PCI `00000000:17:00.0`; frameworks ran sequentially without a competing GPU
  workload. The RTX 2080 was not used. CUDA 13.1.115.
- Qwen3-0.6B BF16, identical input token-ID vectors, greedy, ignore EOS, fixed
  output length, prefix reuse disabled. No quantized-model comparison.
- LunaFlux selected execution matches `5924c1c`; `365e3fe` differs only in two
  documentation files. The disposable executable forwards profiler environment
  variables at worker startup, not a different kernel implementation. SHA-256:
  `6f1f8d690ae35503d5b23f462629a42f76bb1b5e0d16d2e9e6ef8390e2958507`.
- vLLM 0.24.0, BF16, max length 40960, max sequences 32, asynchronous scheduling.
- SGLang 0.5.2, BF16, FlashInfer, max length 40960, max running requests 32,
  chunked prefill 2048, max prefill tokens 16384.
- llama.cpp build 8045 (`0d00ef65e`), BF16 GGUF, BF16 K/V, Flash Attention,
  29/29 layers offloaded, continuous batching, parallel 32, total context 65536
  (2048 per slot), batch 2048, microbatch 512. Every tested request fits its slot.
  These results do not describe its faster quantized configurations.
- New MoonBit streaming client invokes unbuffered curl per request. All four
  engines use this same client. Each vector has one warmup and two measured
  trials; numbers below are arithmetic means, not confidence intervals. C8
  means eight simultaneous requests, not an assertion of constant batch eight
  throughout every server execution.
- Throughput is total completed output tokens / batch wall time. TTFT is mean
  request start-to-first-token time; decode TPOT is mean first-to-last-token
  time / (output length - 1). Client clock resolution is 1 ms and includes curl
  startup/HTTP overhead. Do not mix these with previous-client measurements.
- Ordinary timings have no profiler. Separate Nsight Systems 2025.5.2 captures
  use CUDA graph-node tracing; their kernel times explain costs, not unprofiled
  throughput. No Nsight Compute replay durations are used here.

## Unprofiled output throughput

Output tokens/second; C8 is aggregate throughput.

| Input → output | C | LunaFlux | vLLM | SGLang | llama.cpp BF16 | vLLM / LunaFlux |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 59 → 256 | 1 | 209.1 | 278.9 | 264.2 | 84.3 | 1.33× |
| 59 → 256 | 8 | 702.6 | 1850.0 | 1746.7 | 31.1 | 2.63× |
| 128 → 128 | 1 | 204.2 | 267.0 | 254.7 | 88.9 | 1.31× |
| 128 → 128 | 8 | 658.1 | 1740.0 | 1666.6 | 30.7 | 2.64× |
| 512 → 64 | 1 | 162.2 | 244.3 | 240.6 | 31.9 | 1.51× |
| 512 → 64 | 8 | 382.5 | 1185.2 | 1137.8 | 21.1 | 3.10× |
| 1528 → 32 | 1 | 79.6 | 180.3 | 173.4 | 4.9 | 2.26× |
| 1528 → 32 | 8 | 108.9 | 446.4 | 417.6 | 4.8 | 4.10× |

### Time to first token

Mean request TTFT, milliseconds. This includes queueing, prefill, first-token
execution and client overhead; it is not pure attention or pure prefill time.

| Input → output | C | LunaFlux | vLLM | SGLang | llama.cpp BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 59 → 256 | 1 | 21.0 | 16.0 | 27.0 | 46.5 |
| 59 → 256 | 8 | 49.6 | 34.3 | 59.4 | 354.9 |
| 128 → 128 | 1 | 26.5 | 30.0 | 30.5 | 59.0 |
| 128 → 128 | 8 | 94.9 | 50.2 | 60.4 | 744.2 |
| 512 → 64 | 1 | 64.0 | 28.0 | 28.0 | 624.5 |
| 512 → 64 | 8 | 343.6 | 92.3 | 98.5 | 4343.3 |
| 1528 → 32 | 1 | 213.0 | 54.0 | 57.5 | 5145.0 |
| 1528 → 32 | 8 | 1043.6 | 220.9 | 247.9 | 25390.1 |

### Decode interval after the first token

Mean per-request milliseconds/output token. In C8, one request's decode can
overlap another request's prefill; these are experienced streaming intervals,
not isolated decode-kernel service times.

| Input → output | C | LunaFlux | vLLM | SGLang | llama.cpp BF16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 59 → 256 | 1 | 4.704 | 3.522 | 3.678 | 11.722 |
| 59 → 256 | 8 | 11.192 | 4.179 | 4.320 | 257.119 |
| 128 → 128 | 1 | 4.701 | 3.512 | 3.693 | 10.866 |
| 128 → 128 | 8 | 11.359 | 4.188 | 4.317 | 256.385 |
| 512 → 64 | 1 | 5.198 | 3.659 | 3.738 | 21.929 |
| 512 → 64 | 8 | 15.681 | 5.309 | 5.511 | 313.938 |
| 1528 → 32 | 1 | 6.016 | 3.903 | 4.000 | 46.484 |
| 1528 → 32 | 8 | 40.544 | 10.827 | 11.649 | 894.448 |

## Output equivalence limitations

All requests returned the requested number of token IDs and a terminal event.
This is not an independent correctness oracle. Greedy BF16 sequences differ
between engines and sometimes between batch sizes or repeated trials.

For 59/256 C8, only 4/8 LunaFlux, 6/8 vLLM, 4/8 SGLang and 0/8 llama.cpp
sequences match the corresponding request index in the other measured trial.
Request index is client order, not guaranteed scheduler order. On 1528/32 C8,
the first request in all four engines matches all 32 vLLM token IDs. These
examples neither prove a correctness defect nor establish numerical parity.
The complete prefix/repeat comparisons are retained with the analysis script.
Treat the tables as execution-performance diagnostics, not a quality-equivalent
production claim.

## Trace interpretation

Each measured window is bounded by client-recorded start/end timestamps mapped
to Nsight's epoch. Warmups are separate; there is no hidden learning request.
Kernel busy time is the union of CUDA kernel intervals; gaps are everything
else inside first-to-last kernel, not exclusively scheduler overhead. Memory
copies and CPU/driver/synchronization effects can also occupy gaps. Kernel
family totals are duration sums, not necessarily critical-path contributions.

Baseline matrix names are generic library symbols. QKV/output/gate-up/down/head
classification uses the repeated 28-layer × 4 + 1 dispatch order, checked
against LM-head geometry every 113th matrix dispatch. All checked head
positions match. SGLang's 1528/32 C8 window includes 39 complete groups plus a
97-dispatch in-flight tail; its totals must not be interpreted as a whole number
of complete model forwards. QKV auxiliary categorization is approximate and
includes QK norm, rotary and KV-store kernels (plus small final-norm overlap).

### Matched workload costs: short input, C8

59 input / 256 output tokens × 8 requests. Whole measured-window kernel
duration sums in milliseconds, including both prefill and decode. MLP includes
the baselines' separate SiLU/multiply activation; LunaFlux fuses it. QKV group
includes the approximate auxiliary category above, not just the bare GEMM;
unclassified generic copies remain outside that group.

| Functional group | LunaFlux | vLLM | SGLang | Interpretation |
| --- | ---: | ---: | ---: | --- |
| MLP gate-up, activation, down | 820.2 | 398.3 | 407.2 | ~2× matrix/schedule gap |
| QKV + QK norm / rotary / KV write (approx.) | 625.7 | 185.9 | 235.6 | Fewer fused launches did not make ours faster |
| LM head | 639.4 | 190.5 | 192.2 | ~3.3×; primarily a small-batch problem |
| Attention, including split/merge | 400.7 | 155.7 | 142.4 | ~2.6–2.8× |
| Attention output projection | 185.1 | 88.8 | 93.2 | ~2× |
| Residual / norm category | 66.2 | 15.1 | 22.9 | Real gap, much smaller absolute opportunity |

Representative steady C8 kernel launches show the schedule differences:

| Operation | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| LM head, µs | 2478.5 | 741.1 | 742.1 |
| Gate-up, µs | 85.76 (activation fused) | 34.14 + ~0.79 activation | 34.38 + ~1.32 activation |
| QKV, µs | 85.70 (QK norm / rotary / KV write fused) | 22.09 projection only | 22.08 projection only |
| Decode attention, µs | 55.72 | 21.69 | 17.85 + ~1.93 merge |

The QKV launch row is deliberately NOT a like-for-like speedup ratio: compare
the functional-group table to include the separate operations. vLLM/SGLang
use CUTLASS WMMA 16×16 small-row kernels here, with a 32-thread block. LunaFlux
also uses Tensor Core/WMMA paths; this is not “they have Tensor Cores, we do
not.” Layout, work partition, weight reuse and resource scheduling differ.
Different register/shared-memory footprints alone do not prove causation.

### C1 is different: avoid optimizing an already competitive LM head

For 59/256 C1, LunaFlux's whole-window LM-head time is 188.6 ms vs vLLM
187.9 ms and SGLang 188.2 ms. MLP is 351.2 vs 349.0 / 364.4 ms, including
activation; output projection is 80.3 vs 82.4 / 86.6 ms. These categories are
already close in this shape. The C8 deficits must not be generalized to C1.

Attention is different: LunaFlux takes 202.6 ms vs 71.5 / 51.6 ms. Its QKV
group takes 248.1 ms vs approximately 172.1 / 219.3 ms. Residual/final norm
takes 68.3 ms vs the baselines' residual-norm categories 14.6 / 23.8 ms
(final-norm accounting differs slightly). These, plus gaps between kernels,
explain why C1 can lag even when the main MLP/LM-head kernels are competitive.

### Long input: prefill is the largest attention opportunity

1528 input / 32 output tokens × 8 requests; same whole-window accounting:

| Functional group | LunaFlux ms | vLLM ms | SGLang ms |
| --- | ---: | ---: | ---: |
| Prefill attention | 570.0 | 84.1 | 49.5 |
| Decode attention + merge | 347.7 | 104.9 | 111.6 |
| MLP including activation | 687.4 | 187.2 | 202.0 |
| QKV group (approx.) | 390.8 | 95.6 | 122.2 |
| Attention output projection | 150.1 | 41.3 | 44.7 |
| LM head | 124.7 | 28.2 | 28.9 |

LunaFlux issues 1344 prefill-attention launches; each baseline issues 196
across ALL prefill shapes. LunaFlux's fixed 256-token chunk path does more
small forward segments; this costs ~6.8× vLLM / ~11.5× SGLang in the measured
prefill-attention category. C1 also shows 168 launches / 63.3 ms vs 28 launches
/ 8.3 ms (vLLM) and 28 / 7.9 ms (SGLang).

Do not call the launch-count ratio “tokens recomputed that many times.” Smaller
chunks contain less work each. The measurements implicate chunk policy,
repeated weight/KV traversal and attention's physical schedule jointly. A
same-runtime chunk-size A/B with identical attention lowering, followed by a
kernel-schedule A/B with fixed chunks, is needed to separate them causally.

An earlier scratch reading counted only SGLang's largest prefill shape
(140 launches / ~40 ms). That is NOT the total. The table uses the corrected
196 launches / 49.5 ms across all shapes. The generic name “splitkv” in vLLM
also does not imply a split was used on every request: short C8 has no combine
kernel, while other vectors have split/merge work. Shape-adaptive partition
selection matters more than enabling split-K everywhere.

### GPU windows and non-kernel gaps

First-to-last-kernel interval / kernel union, milliseconds:

| Input → output | C | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: | ---: |
| 59 → 256 | 1 | 1292.9 / 1175.6 | 918.5 / 886.4 | 975.3 / 912.9 |
| 59 → 256 | 8 | 2978.2 / 2773.2 | 1111.7 / 1076.4 | 1157.9 / 1084.9 |
| 1528 → 32 | 1 | 402.7 / 384.2 | 165.8 / 160.9 | 176.1 / 167.6 |
| 1528 → 32 | 8 | 2357.1 / 2305.2 | 564.0 / 555.3 | 613.5 / 590.8 |

LunaFlux's short C8 gaps are 205.0 ms (6.9% of the window); long C8 gaps
are 52.0 ms (2.2%). Even deleting every gap inside those windows would not
close their ~2.6× / ~4× serving-throughput differences. This does not rule out
queueing/client costs outside the GPU window, or useful graph/scheduler work;
it rules out attributing the whole deficit to host dispatch.

### llama.cpp: kernel speed is not serving speed

In its C1 59/256 profile, first-to-last kernel spans 3298.2 ms, with only
874.2 ms kernel busy time (26.5%). For 1528/32, it spans 6578.9 ms with only
162.8 ms kernel busy time (2.5%). Its BF16 LM-head kernel is about 763 µs,
close to the other engines' C1 LM head, despite much worse end-to-end results.
This implicates substantial non-kernel delays in this particular configuration;
CUDA-only tracing does not establish whether CPU computation, synchronization,
memory transfer or server behavior is responsible. Do not label all that time
as Python overhead (llama.cpp is native), or generalize this to all llama.cpp
settings. It was not faster on any of the eight tested throughput vectors.

## Compiler work this diagnosis actually supports

1. **Shape-aware small-batch physical schedules**, especially LM head, QKV and
   MLP. Keep matrix semantics pure; vary tile/layout, cooperative loads, reuse
   and occupancy under a device-specific cost model. Autotune C1 and C8
   separately. The C1 LM head is a control case, not the primary target.
2. **Prefill chunk policy and tiled attention jointly, but experimentally
   separated.** Larger legal query tiles/chunks, KV reuse, and split/merge
   choices are general compiler/planner concerns. Do not just add split-K
   unconditionally, and do not credit reduced launches as proven speedup.
3. **Decode attention partition selection by context and batch shape.** The
   baselines adapt split/merge; ours still has expensive direct-kernel shapes.
4. **Measure fusion profitability, not fusion count.** Our fused QKV has fewer
   launches yet higher total cost. Hoisting/rematerialization, shared layout
   and resource constraints deserve isolated A/B tests; splitting a fusion can
   also be a legal result if it is faster under equivalent semantics.
5. Norm, sampling and graph/dispatch improvements remain worthwhile, but have
   smaller absolute budgets than the above C8 and long-prefill categories.

These remain compatible with functional programming: pure operation semantics,
explicit memory/effect dependencies, legal rewrites and separate physical
lowering. Functional purity does not automatically choose a fast tile shape
or memory layout. The measured problem is physical schedule quality and
shape policy, not evidence that another generic CSE pass alone is missing.
This turn does not implement or claim any resulting speedup.

## Reproduction artifacts

Raw SSE, token timestamps/IDs, trial JSON, launch arguments, server logs,
Nsight reports, SQLite traces and derived SQL/CSV are retained under
`/private/tmp/lunaflux-baselines-20260906-r1-{engine}-{plain,profile}`.
All four engines have ordinary timing runs. vLLM, SGLang and LunaFlux have
short/long C1/C8 profiles; llama.cpp has short/long C1 profiles only.

Local combined archive (75 MiB):
`/private/tmp/lunaflux-baseline-comparison-20260906-r1.tar.gz`, SHA-256
`a1116be6adf09d659686fd672c55eb6d22b1fb7f781c31f9d4b1cd2ede8dcb5e`.
It includes all eight raw result directories, derived CSV/SQL, summary/token
comparisons and the `.mbtx` runner/analysis scripts. The downloaded LunaFlux
profile archive additionally matches its remote SHA-256
`df60e013ee896c93acbf1f16a52d3890364d35a8fb1a4a56f0b7b91c3b962d9d`.
These are temporary filesystem locations, not a published benchmark dataset.

The runner uses `moon run lunaflux-baselines-20260906.mbtx start ENGINE KIND`,
then `measure ENGINE KIND`, where KIND is `plain` or `profile`. Paths and model
launch receipts are deliberately pinned in this diagnostic runner. Re-running
requires fresh output roots, readiness checking and serialized GPU ownership.
All benchmark services were stopped and the final GPU process list was empty.

Diagnostic automation is MoonBit `.mbtx`, following `moonbit-agent-guide`;
existing framework launchers were reused. No production dependency, kernel,
scheduler or compiler code changed. Unrelated dirty working-tree changes were
left untouched.
