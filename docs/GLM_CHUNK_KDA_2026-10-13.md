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
