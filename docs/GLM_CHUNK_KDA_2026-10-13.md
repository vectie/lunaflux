# Parallel chunk delta attention

## Contract and experiment

Target: replace token-sequential KDA prefill with a generic chunk triangular
form, retaining the existing model, convolution, request-slot ownership and
explicit cache commit. Comparison remains outside the compiler. No hashing.

For normalized key k_i, query q_i (including query scale), vector decay D_i
and scalar beta_i, the real-valued recurrence is
S_i = D_i S_(i-1) + k_i u_i^T,
u_i = beta_i (v_i - k_i^T D_i S_(i-1)).
Define D_(j:i) as the componentwise product through the closed interval.
The strictly lower matrix L_ij = beta_i k_i^T D_(j+1:i) k_j and
R_i = beta_i (v_i - k_i^T D_(1:i) S_0) give
U = (I + L)^(-1) R. Outputs are the incoming-state projection plus causal
QK times U; only the final state is committed. Empty decay intervals are one.

The vLLM source in `../vlm/vllm/third_party/flash_linear_attention/ops/kda.py`
uses this chunk/WY structure, cumulative gate decays, a triangular solve,
transformed values/state and parallel output. Its Tensor Core, BF16 intermediate
and exp2 policies are not implicitly our numerical contract. Our first explicit
alternative uses F32 normalized operands, ordered F32 dots, cumulative F32
natural-log gates, strict exp, F32 triangular inversion and BF16 output only.
It is mathematically equivalent over real numbers, **not bitwise equivalent**
to sequential F32 updates, and is not yet an optimal Tensor Core lowering.

The semantic/numerical policy is distinct from the pure physical plan, which
describes bounded frame storage, independent value tiles, and three ordered
effects: prepare causal matrices, solve transformed values using old state,
then publish outputs and commit state. CUDA is the terminal implementation.
Decode and prefill are numerically explicit; unsupported resource geometries
reject at construction, never silently pick different arithmetic.

Budget: one implementation, up to five alternating paired microtrials, existing
0.003 independent-oracle absolute error gate (not a new relaxed threshold),
reset/continuation/reordering/tail/idle tests, memcheck/leak/race/sync checks,
one full-model matched six-prompt run, and selected-kernel hardware counters.
Reuse the earlier same-day vLLM timings. Capture actual source/launch propagation.
Keep 2 GiB/no-swap probes, 96 GiB full-model ceilings, 16 GiB host reserve and
serialized GPU work. Preserve failed/non-winning results and unrelated files.

## First measurements (not a selected production winner)

The three-stage source and owned workspace are wired through recurrent program,
dense/MoE recurrent layer, decoder slice, stage, AOT export and the checkpoint
CLI. Commands ending `-chunk-kda` explicitly select the experimental numerical
law; ordinary commands remain ordered. The matching benchmark/export helpers
accept `--chunk-kda`. Comparison and profiler logic stay outside the compiler.

Five alternating paired trials, retained roots
`/tmp/lunaflux-chunk-delta-20261013-v1` and `...-v2`, show:

| Frame | Ordered | Initial chunk | Local-factor chunk |
|---|---:|---:|---:|
| 16 rows, 64 heads, K/V128 | 0.0493 ms | 0.1095 ms | 0.0901 ms |
| 32 rows, 64 heads, K/V128 | 0.0898 ms | 0.2948 ms | 0.2001 ms |
| 1 row, 64 heads, K/V128 | 0.0103 ms | 0.0261 ms | 0.0261 ms |

All five geometries pass the independent F64 recurrence check, repeated-run
determinism, reset, continuation, reordered request slots, tails and idle frames.
Maximum observed BF16-output absolute difference is 0.0000610351562; F32-state
absolute difference is below 0.00000017. These are small synthetic probes,
not model correctness or bitwise equivalence. Sanitizers/full-model checks are
recorded separately after completion.

The first Nsight Compute capture (`v1/chunk.ncu-rep`) identifies 45.9% average
warp-latency contribution from global-load dependencies in preparation. Moving
reused key/gate factors and causal matrices to padded local storage reduces
time but does **not** beat the ordered implementation. The pure physical plan
selects local storage only when the complete tile fits the backend-supplied
byte budget. Large supported geometries retain frame storage, not an oversized
shared allocation. The F32 scalar causal dots are not vLLM's matrix lowering.

## Full-model attribution changes the optimization priority

Fresh whole-process Systems capture:
`/tmp/lunaflux-model-bench-glm-kda-before-profile-20261013-v2`.
The earlier `v1` fixed-delay capture missed inference; it is retained as a
diagnostic failure. `v2` contains both actual ranks, one 16-token prefill and
one decode step, 2,631.764 ms summed GPU kernel time. Loading copies are excluded
from these kernel sums, which are not wall-clock critical-path percentages.

| Kernel family | Summed ms | Share |
|---|---:|---:|
| KDA recurrence | 2.764 | 0.105% |
| KDA projections | 1,063.346 | 40.404% |
| Routed experts | 799.734 | 30.388% |
| Sparse attention chain | 380.319 | 14.451% |
| Dense FFN | 128.852 | 4.896% |
| Shared experts | 116.533 | 4.428% |
| Hyper connections | 29.142 | 1.107% |

Eliminating the recurrence entirely cannot remove a material fraction of this
measured workload's GPU time. It is incorrect to attribute the remaining
roughly tenfold full-model gap to sequential KDA recurrence alone. The selected
recurrent projection source still performs one scalar ordered K-loop per
output, with no tiled matrix reuse or tensor instruction lowering.

An external same-shape control, copied directly from the selected AOT source,
is retained at `/tmp/lunaflux-glm-projection-counters-20261013-v1`.
For BF16 `[16,4096] × [8192,4096]^T`, the native median is 6.725 ms versus
0.3002 ms for cuBLAS GEMM (22.4×). For one row, 0.629 versus 0.289 ms (2.18×).
Its bounded dyadic diagnostic inputs give identical BF16 outputs. This is a
CUDA-library hardware/control experiment, **not** an executed vLLM kernel or
model-output comparison. Hardware counters and the selected vLLM source must
be distinguished from this microbenchmark.

The same-shape native/vendor Compute reports are downloaded as
`native.ncu-rep` and `vendor.ncu-rep` in that root. Scoped administrator capture
did not change driver settings. The observed library kernel is
`cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128x2_tn_align8`.

| Metric, one 16-row invocation | Native selected source | Library control |
|---|---:|---:|
| Executed warp instructions | 106,037,248 | 3,781,120 |
| Global-load requests | 33,558,528 | 262,144 |
| L1/global-load sectors | 553,652,224 | 4,194,304 |
| Tensor pipe activity | 0% | 2.616% |
| Achieved occupancy | 79.45% | 10.06% |

The native projection issues narrow loads at `column*4096+inner`: adjacent
lanes read different weight rows, 8 KiB apart. The profiler reports only
2 useful bytes per 32-byte sector. Each input row repeats this traversal;
the generated schedule has no shared tiled matrix reuse. Its **LG instruction
queue throttle contributes 77.3% of average warp latency**, rather than a
barrier or bank-conflict bottleneck. Warp instructions are 28.04x and load
sectors 132.0x the library control. Sectors include cache hits: these are NOT
DRAM-byte measurements. Raw DRAM-byte counters were not present in this capture.
The faster control itself is 95.4% long-scoreboard-bound by average warp latency;
stall percentages and occupancy alone are not elapsed-time comparisons.

The inspected vLLM Kimi-GDN implementation merges Q/K/V plus narrow controls
in `_KimiGDNMergedColumnParallelLinear` and invokes its configured GEMM method;
its unquantized method dispatches GEMM, while our actual KDA owner still emits
separate scalar projections. It independently chooses chunk prefill and fused
recurrent decode. Neither this source inspection nor the cuBLAS control proves
which exact GEMM ran in the previously frozen vLLM full-model request.

## Full-model experiment: no improvement, do not promote

Implementation commit `7bed340c`, clean ARM build and fresh two-rank AOT:
`/tmp/lunaflux-checkpoint-build-glm-chunk-kda-20261013-v1`.
Export and execution both explicitly select `chunk-triangular-f32`; the AOT
contains prepare/solve/publish, and the prepared executor loads all three
functions with their launch metadata and owned scratch. There is no old delta
kernel fallback in this experimental bundle.

Unprofiled root `/tmp/lunaflux-model-bench-glm-chunk-kda-20261013-v1`,
`comparison.json` retains input/output token vectors and timing boundaries.
The same-day vLLM reference is reused, not rerun or retuned.

| Prompt | Input/output tokens | Ordered native s | Chunk native s | vLLM s |
|---|---:|---:|---:|---:|
| Fact | 22/4 | 4.350 | 4.376 | 0.384751 |
| Arithmetic | 23/4 | 4.504 | 4.536 | 0.388855 |
| Chinese | 22/8 | 5.717 | 5.770 | 0.653645 |
| JSON | 23/8 | 5.932 | 5.976 | 0.664909 |
| Code | 24/8 | 6.050 | 6.101 | 0.670385 |
| Retrieval | 36/3 | 6.428 | 6.468 | 0.398416 |
| Sum | 150/35 | 32.981 | 33.227 | 3.160961 |

The observed sum is **0.746% slower**, not a speedup (single sample; no
statistical significance claim). Completion time is still **10.512x vLLM**.
All six new native token vectors exactly equal the prior native vectors.
Four texts match vLLM after excluding the native stop marker; Chinese/JSON
still differ, and vLLM output IDs are null. This does not establish model parity.
Native PP2/TP1 uses BF16 expanded KV; vLLM TP2/PP1 uses FP8 compressed MLA KV.
Native CLI and buffered HTTP timing boundaries and warmup differ. These are
short prompts only, not a long-context conclusion.

Both ranks exit zero and explicitly close their CUDA contexts. Process swap
peaks are zero and observed host reserve stays at least **22.577 GiB**, above
the 16 GiB floor. Device/process ceilings remain 96 GiB. No payload hashes.

Decision: preserve the generic chunk algorithm, storage plan, effect ownership
and explicit diagnostic route, but keep it nondefault. Matching vLLM performance
is NOT achieved. The next performance implementation should lower the hot
projection families to tiled matrix execution and group the routed-expert work,
with explicit numerical laws and model-quality checks outside the compiler.
Adding more IR names, tuning the recurrence alone, or raising occupancy cannot
remove the measured major costs. The 40.4%/30.4% shares are our kernel-time
composition, not an additive decomposition of the vLLM wall-clock gap.

## Validation and retained limits

The final committed generator was exported/compiled again at
`/tmp/lunaflux-chunk-delta-20261013-v3`, after adding the generic local-storage
budget and large-geometry fallback. All five shapes pass the independent oracle
and all 15 memory/leak/race/sync campaigns, with zero errors/hazards/leaks,
explicit device release, 2 GiB caps and zero process swap. Final paired medians
are 0.04893/0.09225 ms ordered/chunk at 16 rows and 0.08959/0.20195 ms at
32 rows. These confirm the non-win; they do not replace the retained v1/v2 data.

Affected native tests pass (96 tests), the checkpoint target passes warning-
denied check with the existing migration-warning exclusions, and the clean
commit-pinned ARM build succeeds. `moon info` has 936 existing migration warnings
and no errors. Both unexcluded full native check and test were attempted and
stop on existing warning-79/related migration errors outside this change; there
is no new full-suite-pass claim or suppression. Unrelated toolchain/TLS work is
not staged. Comparison helpers, counter reports and prompt outputs are external
diagnostics; no payload hashing, runtime profiler dependencies or reference
substitutions were introduced.

Remaining chunk work is specifically matrix/tiled lowering of the causal
products and transformed-value/state contractions, plus measured dispatch
between prefill chunks and recurrent decode. The implemented triangular
algorithm should not be described as the complete optimized vLLM schedule.
