# Head, down and gate/up: measured operand-supply optimization

## Result

The three diagnosed bottlenecks now have compiler/backend changes and fresh
physical measurements. Against frozen LunaFlux `f3d57a2`, selected C8 cold-cache
replay improves **head 1.94x, down 3.15x, gate/up 2.41x**. Ordinary Qwen3-0.6B
59-input/256-output C8 throughput improves **789.21 -> 1280.80 output tokens/s
(+62.3%)**. The vLLM throughput ratio narrows from 2.34x to **1.44x**, and
SGLang from 2.21x to **1.36x**. This is substantial progress, not parity.

Tested implementation: `987db7a7d79369ca9722ba3fe6ca86dbb00cbb05`.
Its clean Git source archive SHA-256 is
`9735648857d061696b29c0d93b53428f4ff62aede809b6062ca802fd9fda43d7`.
Unrelated dirty working-tree changes are excluded. No production deployment
or external baseline source was changed.

## What the baseline implementations teach

The inspected vLLM path is `UnquantizedLinearMethod.apply` in
`../vlm/vllm/model_executor/layers/linear.py`, through
`dispatch_unquantized_gemm` / `default_unquantized_gemm` in `layers/utils.py`.
The inspected SGLang path is the unquantized `F.linear` route in
`../sgl/python/sglang/srt/layers/quantization/unquant.py`.

Captured calls from the installed benchmark versions select **PyTorch/cuBLAS
BF16 GEMM**, not a custom vLLM/SGLang projection kernel. Newer optional source
branches in the neighboring checkouts are not assumed to have executed.
The comparison retains the captured transposition, dimensions, FP32 compute,
workspace, selected symbols, launch geometry and library versions from the
[matched-counter diagnosis](CONCURRENCY_GAP_CAUSES_2026-09-10.md).

The useful lesson is how to supply operands: keep independent loads ahead of
their consumers, overlap transfers with the ordered computation, and eliminate
runtime layout selection. Adding resident warps alone did not hide the head's
latency; vectorizing down alone could not remove its already-vectorized
LDG-to-STS dependency. We did not replace LunaTile with a library call.

## Functional compiler changes

### 1. Head: bounded immutable operand lookahead

`ProjectionSelectedRowGather` now carries `min(K / 16, 8)` future fragments.
The selected-row map is invariant; the compiler evaluates immutable input and
weight fragments ahead of use. CUDA renders statically named register slots,
primes them, evaluates the next operand, consumes the current operand, then
reuses its slot. Tail guards prohibit out-of-range loads and extra reductions.

The **K16 accumulator update order and BF16 store are unchanged**. This is
evaluation scheduling, not floating-point reassociation. The single-token
route retains its previous numerical behavior. The lookahead participates in
canonical schedule identity.

The four-fragment intermediate was faster than the old code; eight fragments
improved it further. A weights-only lookahead with just-in-time input loads
was also physically tested: it was bitwise equal but approximately 16% slower
than the selected eight-fragment implementation at C8, so it was discarded.

The selected head still has **128 registers and a 24-byte spill stack**, verified
in the original uninstrumented CUBIN as well as the profile. It reaches 55.84%,
not the baseline's 95.98%, of peak DRAM throughput. It is not fully optimized.
NCU's spill-instruction metric counts compiler-generated spill loads/stores;
it is not a byte count ([NVIDIA metric definition](https://docs.nvidia.com/nsight-compute/ReleaseNotes/topics/updates-2025-1.html)).

### 2. Down: overlap the next transfer with the current fold

The materialized consumer lowering issues next-slot asynchronous transfers
before consuming the current slot, now also for small workgroups. The old
`groups >= 4` restriction excluded the diagnosed C8 path. Transfer width stays
64, reductions stay ordered K16, and the shared layout and BF16 boundary are
unchanged. This removes the direct global-load-result -> shared-store wait
from the producer sequence; it is not a bank-conflict workaround.

The C8 down grid remains 32 CTAs in the actual old/new pair. Its gain therefore
does not come from pretending that twice as many blocks were launched.

### 3. Gate/up: specialize operand domains and retain independent work

The pure sibling plan uses 64-element transfers for a bounded single row tile
with at most eight consumer groups. Wider live products retain 32-element
transfers so their storage stays within the admitted 49,152-byte envelope.
Both use two slots and the same ordered K16 fold.

The CUDA backend separately renders input, gate and up affine copy domains.
It no longer decides the operand source with a per-copy runtime selector.
Packed matrix-fragment loads replace scalar shared loads and permutations.
The masked input prime is materialized synchronously: there is no preceding
fold to overlap yet. Steady-state input and weight copies remain asynchronous.
That phase specialization removed 10,752 source-attributed excess shared
wavefronts observed in the intermediate asynchronous-prime version.

Pure `workgroup_count` planning is now carried through optional
`bounded_work_grid_x` artifact metadata into startup-prepared launch records.
The old launch heuristic reduced the selected C8 gate/up executable to 96 CTAs;
the new launch preserves its 192 independent CTAs. The value is checked and
clamped against the published maximum at startup. Missing metadata retains
legacy behavior; companion down geometry is separate.

No model-name conditional, production Python dependency, runtime JIT,
token-step allocation, filesystem scan or cryptography was added. Semantic
graphs and ordered folds remain pure; CUDA instructions and storage effects
remain in CUDA lowering. This is reusable planning for supported projection
families, not arbitrary-DAG lowering or a universal autotuner.

## Selected-kernel counters

RTX 5060 Ti 16 GB, 36 SMs; C8 selected immutable CUBINs; cold-cache NCU replay
with deterministic synthetic operands. Time is microseconds. The old and
baseline columns are the matched-counter campaign; the new column is fresh.
These are not ordinary-serving per-step timings.

| Kernel | Old LF us | New LF us | vLLM us | Old/new speedup | New LF / vLLM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Head | 2483.936 | 1279.040 | 744.288 | 1.94x | 1.72x |
| Down | 92.640 | 29.440 | 20.256 | 3.15x | 1.45x |
| Gate/up | 79.392 | 32.960 | 34.464 | 2.41x | 0.96x |
| Output | 32.448 | 32.736 | 13.152 | 0.99x | 2.49x |
| QKV | 26.304 | 24.800 | 22.656 | 1.06x | 1.09x |

QKV/output are controls, not claimed optimizations from this change. Small
differences on their unchanged paths should not be credited as compiler gains.
LF gate/up also fuses activation; the library replay here is GEMM only.

| Counter | Head old -> new | Down old -> new | Gate/up old -> new |
| --- | ---: | ---: | ---: |
| DRAM throughput, % peak | 28.79 -> 55.84 | 15.54 -> 48.92 | 36.03 -> 86.80 |
| Long-scoreboard stalls, % | 98.24 -> 88.81 | 79.65 -> 46.74 | — |
| Warp instructions | 15,506,968 -> 25,287,848 | 532,608 -> 566,400 | 2,519,616 -> 1,103,424 |
| Barrier stalls, % | — | — | 12.30 -> 2.70 |

Read volumes remain approximately 311 MB, 6.35 MB and 12.61 MB respectively.
Head and down become faster despite more instructions: operand availability,
not instruction count alone, was their limiting factor. Gate/up instructions
fall 56.2%, but remain about 1.88x the baseline's 588,288.

All five replays have **zero source-attributed excessive shared wavefronts**.
This does **not** mean every hardware aggregate bank-conflict counter is zero:

| Kernel | Hardware shared-load conflicts | Hardware shared-store conflicts |
| --- | ---: | ---: |
| QKV | 32 | 6175 |
| Output | 0 | 419 |
| Gate/up | 104 | 1675 |
| Down | 0 | 0 |
| Head | 0 | 0 |

Source/SASS derived and hardware aggregate counters have different accounting;
they are retained separately, not substituted for one another
([NVIDIA profiling guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)).

### Warm-weight same-process A/B

The paired-launch harness alternates old/new order across three timed graph
trials with repeated weights. This cache regime is deliberately reported apart
from the cold-cache library comparison.

| C8 kernel | Old LF us | New LF us | Speedup |
| --- | ---: | ---: | ---: |
| Head | 2474.658 | 1280.958 | 1.93x |
| Down | 43.905 | 17.155 | 2.56x |
| Gate/up | 50.042 | 13.240 | 3.78x |
| Output | 27.403 | 27.367 | 1.00x |
| QKV | 19.213 | 19.155 | 1.00x |

## Ordinary end-to-end throughput

Same model files, BF16, token-ID inputs, greedy decoding, ignored EOS, prefix
reuse disabled, one warmup and three measured synchronized finite bursts per
cell. No profiler or competing GPU workload during ordinary measurement.
vLLM 0.24.0 and SGLang 0.5.2 are the same-day baseline runs; SGLang's graph
maximum is 32, avoiding an unfair eager fallback at C16. These are finite-burst
output-token rates, not a saturation benchmark or quality-equivalence claim.

| Input / output tokens | C | Old LF tok/s | New LF tok/s | vLLM tok/s | SGLang tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 59 / 256 | 1 | 248.46 | 247.58 | 277.76 | 268.46 |
| 59 / 256 | 2 | 210.41 | 351.25 | 520.15 | 494.21 |
| 59 / 256 | 4 | 408.62 | 673.09 | 976.17 | 947.12 |
| 59 / 256 | 8 | 789.21 | 1280.80 | 1850.05 | 1746.46 |
| 59 / 256 | 16 | 1426.02 | 1978.49 | 3229.44 | 3117.26 |
| 128 / 128 | 1 | 241.06 | 240.91 | 268.72 | 258.59 |
| 128 / 128 | 2 | 206.84 | 340.88 | 496.45 | 476.20 |
| 128 / 128 | 4 | 397.93 | 646.20 | 933.17 | 896.25 |
| 128 / 128 | 8 | 766.28 | 1216.15 | 1744.47 | 1655.24 |
| 128 / 128 | 16 | 1352.11 | 1843.95 | 2968.13 | 2864.40 |
| 512 / 64 | 1 | 208.93 | 208.93 | 244.95 | 242.45 |
| 512 / 64 | 2 | 187.96 | 290.91 | 433.41 | 422.92 |
| 512 / 64 | 4 | 331.75 | 485.77 | 744.19 | 719.21 |
| 512 / 64 | 8 | 573.99 | 791.35 | 1183.36 | 1133.58 |
| 512 / 64 | 16 | 849.56 | 1017.56 | 1650.73 | 1604.18 |
| 1528 / 32 | 1 | 131.87 | 133.89 | 180.46 | 175.52 |
| 1528 / 32 | 2 | 126.74 | 166.67 | 268.54 | 255.70 |
| 1528 / 32 | 4 | 180.45 | 216.95 | 367.47 | 344.71 |
| 1528 / 32 | 8 | 238.29 | 267.78 | 444.70 | 419.90 |
| 1528 / 32 | 16 | 272.83 | 288.83 | 494.05 | 471.02 |

Single-request performance is essentially unchanged. Short C2 improves 66.9%
and C16 38.7%. Long C8 improves only 12.4%; vLLM/SGLang remain 1.66x/1.57x
faster there. The three fixes do not solve all long-prefill or output-projection
costs. No fresh full-model profiler decomposition is inferred from these
throughput values; the new NCU measurements concern the five selected kernels.

## Correctness and limits

- Clean Linux source: `moon info`, `moon fmt --check`, warning-denied native
  check and **2,974/2,974 tests passed**. Local affected compiler/backend tests:
  **105/105**; earlier affected engine/artifact tests also passed.
- **55/55** old/new actual-CUBIN cases agree bitwise, including untouched tails,
  finite observed outputs and deterministic resource release.
- Another **55/55** agree bitwise with varied signed BF16 exponents/mantissas,
  avoiding reliance on the initial simple, unusually exactly representable
  synthetic operands.
- Coverage for each of five families `(tokens, query rows)`:
  `(1,1), (2,2), (4,4), (8,8), (16,16), (17,8), (32,32), (33,8), (65,8),
  (512,8), (1024,32)`, using actual row variants and launch metadata.
- **75/75** paired sanitizer cases passed: memcheck with full leak check,
  racecheck and synccheck, five families at token bounds 1/8/16/17/1024.
- Ordinary generated sequences match the same named old request in
  **365/372 measured comparisons**. Seven C4 sequences differ (one 128/128,
  six 512/64); this must not be described as full end-to-end bitwise identity.
- Including warmups, **all 496 new sequences occur in the old version's
  same-input sequence pool**. The old version already emits different
  sequences across row/concurrency configurations, and sometimes within C4.
  This is consistent with existing batch-dependent numerical behavior, not a
  newly observed sequence type. It does not by itself prove a causal batch
  timeline or establish batch-invariant determinism. That limitation remains.

## Reproduction and preserved results

Local non-overwriting archive directory:
`benchmarks/qwen3_comparison/results/operand-supply-987db7a-20260910/`.
Archive: `lunaflux-operand-supply-987db7a-20260910-r1.tar.gz`.
SHA-256: `0921509e781d519263aa332749ae3d150586fe0cc1a53099837921cb8c12b924`.

It contains source archive, release build, generated sources/CUBINs, actual
launch plans, pair/sanitizer/diverse-input logs, raw NCU reports, ordinary
request SSE/timestamps, sequence comparisons, three-engine summaries, and
MoonBit `.mbtx` orchestration/analysis. Model weight copies are not duplicated
into the archive. The rejected weights-only head experiment is preserved
separately within it and is not the selected runtime.

Remote result roots are `/dev/shm/lunaflux-supply-final-{pair,sanitize,counters}-r4`,
`/dev/shm/lunaflux-supply-diverse-r1`, and
`/run/user/1000/lunaflux-concurrency-supply-benchmark-r4`.
The runtime is `/dev/shm/lunaflux-concurrency-supply-integrated-r4/runtime`.
The benchmark-owned process group was stopped and the GPU left idle. Only a
reproducible debug build cache was removed to reclaim RAM; results and release
binaries were preserved. No production cutover was performed.
