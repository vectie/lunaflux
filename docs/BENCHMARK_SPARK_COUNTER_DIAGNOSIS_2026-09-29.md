# Spark hardware-counter diagnosis — 2026-09-29

## Conclusion

The explicit full-route Qwen3-0.6B BF16 workload **4096 input / 64 output,
C16** is slower primarily because of the selected GPU schedules, not host
submission bubbles or exhaustion of unified memory. Attention and the complete
QKV → QK normalization → RoPE → KV-write chain account for approximately **81%
of the additional summed kernel time** in the captured batches.

The strongest source-level finding is that full ingress fusion constrains the
projection to a 16-row, per-head work decomposition. Reducing kernel launches
has not compensated for its much more expensive matrix implementation. This
is a physical scheduling/fusion-choice problem; more IR layers alone will not
make the selected schedule faster.

No production code or runtime deployment was changed by this diagnosis.

## Workload, versions and measurement boundaries

This follows [the architecture retest](BENCHMARK_SPARK_ARCHITECTURE_RETEST_2026-09-29.md).
LunaFlux code is committed `53deab59`, with the same isolated sm121 / CUDA
13.0.88 portability changes and **explicit full** bundle used there. It is not
the wrapper's accidentally unfused/residual-only default. The current dirty
working tree was not built or uploaded.

Hardware is GB10, 48 SMs, `spark-368c`, GPU
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, approximately 121 GiB unified RAM.
The pinned vLLM image is NVIDIA `26.01-py3`, vLLM
`0.13.0+faa43dbf.nv26.01`, not latest upstream. Both use the same model weights
and synthetic pre-tokenized requests as the preceding benchmark, greedy
generation with EOS ignored, prefix caching disabled. Frameworks run serially.

Three measurement types are deliberately separate:

1. The preceding five-batch E2E median: LunaFlux **6221 ms**, vLLM **4202 ms**,
   SGLang **4229 ms**. LunaFlux takes 48.0% / 47.1% more completion time.
2. Nsight Systems serving traces: LunaFlux measured batch **6178 ms**;
   vLLM trace rerun batches **4222 / 4219 ms**. They reproduce the gap.
3. Nsight Compute kernel replay: instruction, pipeline, occupancy and stall
   counters. Replay timings are **not** E2E performance measurements.

Nsight Compute 2025.3.1 runs inside isolated SYS_ADMIN-capable containers, with
clock control and cache flushing disabled. No global driver profiling policy
was changed. Host driver is 580.178.04; profiling containers use the image's
CUDA 13.1 compatibility userspace. This environment difference from native
serving and replay/cache effects limit direct interpretation of replay timings.

The temporary LunaFlux parent forwards profiling state to the unchanged
worker. Its diagnostic-only FD/exec changes are preserved separately; kernels,
worker executable, model and full bundle remain the frozen retest artifacts.

## Serving-time attribution

Nsight Systems' exact union of kernel intervals gives:

| Captured batch | Kernel span | GPU busy union | Busy/span | Sum of kernel durations |
| --- | ---: | ---: | ---: | ---: |
| LunaFlux, measured batch | 6161.38 ms | 6030.74 ms | 97.88% | 6032.33 ms |
| vLLM, complete first batch | 4185.76 ms | 4086.76 ms | 97.63% | 4087.76 ms |

Thus GPU idle gaps are not the dominant explanation for this approximately
two-second gap. Busy means some kernel is executing, not that the arithmetic
or memory pipelines are efficiently utilized.

| Operation chain, whole captured batch | LunaFlux | vLLM | Additional kernel time |
| --- | ---: | ---: | ---: |
| QKV + QK normalization + RoPE + KV write | 1224.25 ms | ~456.23 ms | ~768.02 ms |
| All attention, including partial/combine/merge kernels | 3328.88 ms | 2522.56 ms | 806.32 ms |
| Everything else | ~1479.20 ms | ~1108.97 ms | ~370.23 ms |
| Total | 6032.33 ms | 4087.76 ms | 1944.57 ms |

For vLLM, ingress includes both QKV `Kernel2` grids `(128,2,1)` and
`(8,32,1)`, `triton_red_fused_1/3`, `triton_poi_fused_2/4`, and
`reshape_and_cache_flash_kernel`. Trace ordering corroborates the role of the
small-row projection: it is followed by the QK reduction. The comparison does
**not** omit vLLM's separate epilogue and KV-write kernels.

The large-row portion alone is about **1138.85 ms** for LunaFlux, versus
**374.85 ms** for vLLM's projection plus normalization/RoPE/KV-write chain.
LunaFlux has 924 large-grid ingress launches versus 896 baseline large-row
chains: graph padding and mixed-step schedules differ, so these are complete
serving-workload costs, not perfectly paired individual invocations.

Attention phases must not be compared by name alone. vLLM's mixed attention
kernel can include work LunaFlux splits between prefill and decode kernels.
Consequently, the whole-attention chain is the primary aggregate comparison.
Neither these deltas nor stall fractions are additive predictions of speedup.

## Selected-kernel hardware counters

Instruction counts below are **executed warp instructions**, not scalar
thread instructions. Tensor activity is the Nsight tensor-pipeline activity
percentage, not achieved TFLOPS or the fraction of instructions that are MMA.

| Selected invocation | Replay duration | Warp instructions | Tensor activity | Achieved occupancy |
| --- | ---: | ---: | ---: | ---: |
| LunaFlux full ingress, grid 128×32, block 128 | 1298.43 µs | 98.11 M | 10.87% | 16.52% |
| vLLM QKV projection, grid 128×2, block 256 | 297.76 µs | 13.88 M | 48.72% | 16.64% |
| LunaFlux prefill attention, grid 63×16, block 128 | 736.96 µs | 48.23 M | 20.47% | 15.88% |
| vLLM first prefill attention, grid 32×1×16, block 128 | 337.25 µs | 19.56 M | 45.76% | 8.33% |
| LunaFlux exact-CUBIN decode probe, C16/history 4096 | 1305.86 µs | 121.35 M | 0% | 32.21% |
| vLLM serving decode, grid 1×16×8 | 1207.97 µs | 17.79 M | 12.12% | 8.33% |

The ingress rows do different amounts of epilogue work; the 7.1× instruction
ratio is **not** a same-operation redundancy ratio. The attention rows are
selected first configurations, not all-layer averages. Their query tiling,
mixed-step metadata and histories are not identical. The LunaFlux decode
probe uses synthetic tensors with the exact production CUBIN; the baseline
decode counters come from real serving. These qualifications matter: decode's
6.8× instruction difference does not imply a 6.8× duration gap.

### 1. Full ingress: too much movement/dependency work for the matrix work

At nearly identical occupancy, LunaFlux has much lower tensor activity.
Its average warp latency per issued instruction is 12.28 cycles, including:

- Long scoreboard: 3.31 cycles, approximately 27.0%.
- Short scoreboard: 3.23 cycles, approximately 26.3%.
- Fixed-latency wait: 2.44 cycles, approximately 19.9%.
- Barrier: 0.56 cycles, approximately 4.6%.

The selected kernel uses 45,056 static shared-memory bytes per block
(46,080 bytes including the reported allocation overhead) and permits two
resident blocks per SM. **That resource limit alone is not the explanation**:
vLLM's projection has practically the same achieved occupancy and much higher
tensor activity. Increasing occupancy or eliminating barriers in isolation is
not an adequate fix.

`kernels/luna_cuda_fused_parallel_aot/qwen_source.mbt` rejects tile dimensions
other than 16×16×16, uses a per-packed-head grid, and a `projected[16][128]`
shared epilogue. For the observed large launch it emits 4096 CTAs, versus 256
for vLLM's QKV projection. `source_ingress_projection.mbt` stages operands and
iterates WMMA fragments within that decomposition. This restricts reuse across
rows/heads and pays substantially more CTA/transfer/fragment bookkeeping.

The baseline trace selects a CUTLASS 256×128×32, three-stage projection, followed
by separate QK/RoPE/KV operations. The neighboring vLLM Qwen source likewise
expresses projection, QK normalization, rotary embedding and attention
separately. Its source is corroborating structure, not a claim that the local
checkout exactly matches the pinned container revision.

**Next experiment:** permit larger projection row/column tiles independently
of the epilogue ownership region. Compare the full chain against a larger
projection plus partial fusion. Preserve both semantics; choose by measured
chain cost rather than making maximum fusion mandatory.

### 2. Prefill attention: load dependencies and excessive instruction work

LunaFlux's average warp latency is 12.35 cycles. Long-scoreboard waits contribute
6.94 cycles (**56.2%**), versus 1.75/7.44 (**23.5%**) for the selected baseline
prefill launch. Barrier contribution is approximately 7.0% versus 3.0%.
The LunaFlux kernel has 209 registers/thread, but **no local spilling requests
were recorded** in this capture. The old hypothesis that this run is mainly
local-memory spilling is not supported.

The selected baseline executes roughly 2.5× fewer warp instructions and has
more than twice the tensor activity, despite half the occupancy. A subsequent
code-to-artifact review corrected the initial interpretation: this serving
bundle selects **synchronous c318**, not the available async c322 alternative.
Its measured SASS contains `LDG.E.128` followed by `STS.128`, not `LDGSTS`.
Thus these counters do not establish that the async prefill implementation
failed to overlap loads; it was not the selected implementation in this run.
The follow-up also found that the first capture takes the dense-current-token
path: its paged fallback has zero executed instructions. Page-table traversal
therefore cannot explain that invocation's long-scoreboard percentage.
See the [three-agent source review](BENCHMARK_SPARK_THREE_KERNEL_SOURCE_REVIEW_2026-09-29.md).

Long scoreboard means waiting on L1TEX-served dependencies; it is **not proof
of saturated DRAM bandwidth**. The selected LunaFlux L2 hit rate is 94.22%.

**Next experiment:** couple query/KV tile choice, operand ownership, async
prefetch distance and resource budget in the physical-plan search. Compare
complete mixed-attention steps, not merely an isolated early prefill tile.
Use PC-level operand-lifetime analysis to determine which remaining waits can
be moved off the consumer critical path; the present service capture does not
pin every prefill wait to a unique source instruction.

### 3. Decode: SIMT fold and synchronization remain expensive, but are not 6× slower

The exact LunaFlux C16/history-4096 decode CUBIN has average warp latency
16.23 cycles: long scoreboard 5.34 (**32.9%**), barrier 4.04 (**24.9%**), short
scoreboard 2.60 (**16.0%**), fixed wait 1.80 (**11.1%**).
Its source-correlated **excessive shared wavefront count is zero** in this
invocation. That does not establish zero hardware bank counters for every
kernel or shape. It does establish that returning to bank-conflict reduction
alone would miss this measured bottleneck.

The grouped decode lowering folds keys through subgroup shuffle dot products,
FP32 online-softmax state and explicit shared publication/synchronization.
`source_decode_pipeline.mbt` produces page-table-derived addresses and separate
K/V async copy groups; `source_grouped_split.mbt` consumes them in per-key folds.
SASS samples concentrate at the async consumer boundary and page-validity
comparisons. A sampled `BAR.SYNC.DEFER_BLOCKING` PC can reflect waiting for
earlier async work; it must not be mislabeled as a global-load instruction.

The standalone probe passed its CPU double-precision oracle with maximum
absolute error **0.00003051**, preserved KV contents and released resources.
Its ordinary event timing was **1243.71 µs**, compared with approximately
**1055.21 µs** for vLLM's predominant C16 decode configuration in the ordinary
trace. This is supporting context, not a controlled same-tensor microbenchmark.

**Next experiment:** reduce instructions and synchronization per KV tile using
a different reduction ownership/partition plan; compare a matrix-based grouped
decode alternative where profitable. Measure duration and traffic, not simply
occupancy or instruction count. Baseline decode is itself strongly
long-scoreboard-limited, explaining why fewer instructions alone do not yield
proportional speedup.

## Compiler implications and order of work

Keep model semantics and the functional compiler architecture. Change what the
physical plan is allowed to express and select:

1. Larger GEMM tiles with independently planned epilogue ownership; compare full
   and partial ingress fusion by end-to-end chain cost.
2. Joint attention tile/transfer/consumer-lifetime search with measured resource
   feedback. Include mixed prefill/decode steps in the objective.
3. Decode fold/partition alternatives that amortize page addressing and
   synchronization; retain numeric and KV-update semantics.
4. Then address output/down and other secondary kernel costs. Together the
   non-ingress/non-attention families account for only about 19% of this
   additional kernel time.

The concrete lesson is **semantic fusion must not lock the best GEMM schedule
out of the search space**. Pure immutable plans and explicit effect boundaries
are compatible with all of these changes; no Qwen-specific scheduler branching
is needed. These are proposed experiments, not measured gains or promises to
close the whole gap.

## Memory, cleanup and limitations

Runtime/container limits were 40 GiB for the native LunaFlux trace, 48 GiB for
LunaFlux service counters, 8 GiB for the exact-CUBIN decode probe, and 64 GiB for
vLLM. Containers had no extra swap allowance. vLLM cache fraction was 0.35 for
traces and 0.15 for counters, instead of the preceding E2E run's 0.5. The counter
configuration still advertised 146,448 KV tokens, above this workload's need.

Services were watched approximately every 200 ms with a 32 GiB available-memory
cutoff. Minimum observed available RAM was **72.02 GiB** across service campaigns;
the vLLM counter run minimum was **77.64 GiB**, LunaFlux counter run **83.58 GiB**.
Swap remained approximately 400 MiB, with no observed growth. No cutoff fired;
inspected containers report `OOMKilled=false`. Driver memory is not assumed to
be fully charged to cgroups, which is why system availability was watched too.

All diagnostic GPU services/containers are stopped; final GPU compute-process
list is empty. The vLLM counter container needed the timeout kill after
`stop_profile` and both client batches completed (container exit 137,
`OOMKilled=false`), so it is not labeled a clean graceful server shutdown.
Temporary HTTP helpers are also no longer running. Their logging/stop failures
are not kernel correctness failures. No existing production workload was stopped.

Important coverage limitations:

- The initial vLLM trace and the second batch's tail in the flush retry are
  incomplete. The timeline comparison uses the **complete first/warmup batch**
  of the retry, versus LunaFlux's complete measured batch. Their client times
  are close, but this is not a paired confidence-interval study.
- The LunaFlux service counter campaign was bounded after three selected launch
  configurations, not a completed service-correctness campaign. Decode has the
  separate successful numeric oracle described above.
- vLLM replay's first batch took 502,896 ms because of counter passes; it is
  excluded from speed comparisons. The subsequent batch after stopping counters
  completed in 4341 ms. Both returned the requested output lengths.
- No fresh SGLang counters, exhaustive shapes, or longer-context diagnosis were
  collected here. SGLang values above remain the prior E2E measurements.
- Frozen artifact hashes and numeric checks are startup/offline diagnostics;
  no checks were added to production token execution.

## Retained data

Compact [counter records](../benchmarks/spark_counter_diagnosis_20260929/counters.json)
contain 37 captures with metric units. [Memory minima](../benchmarks/spark_counter_diagnosis_20260929/memory-summary.json)
retain sample counts and the original campaign labels.

Remote root: `/home/wlc004s/lunaflux-counter-53deab59.eZThrxoV`.
Local root: `/tmp/lunaflux-counter-diagnosis.rUW2WBeZ`.
Downloaded archive: `diagnosis-archive.tar.gz`; local and remote SHA-256 match:
`3defb0bf17ea244bfc7ebfb4753a4d4ad46fa0188e2371e394d63a3245c82754`.
It includes Nsight reports/SQLite traces, raw counters/SASS, client bodies and
outputs, `.mbtx` orchestration, the diagnostic decode probe, commands and memory
samples. Model weights and duplicate build trees are excluded. The frozen base
runtime/CUBIN archive is retained by the preceding architecture retest.
