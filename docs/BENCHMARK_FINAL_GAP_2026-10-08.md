# Remaining framework performance gap on Spark

Fresh Qwen3-0.6B BF16 measurements on Spark .179 put the latest opt-in LunaFlux
runtime **3.95% above vLLM and 2.66% above SGLang in completion time** for
8192 input tokens, 64 output tokens and eight concurrent requests. The remaining
gap is not one missing compiler pass. Against vLLM, the largest measured
component differences are decode attention and prefill/mixed gate-up. Against
SGLang, most of this capture's net difference is GPU-inactive time.

No production implementation or selection policy was changed in this
investigation. The selected opt-in runtime includes the reference mixed-attention
owner and exact-live-row vendor output/down companions from the
[preceding integration](ORDERED_VENDOR_PROJECTION_2026-10-08.md).
It is not the old baseline or a claim that the default runtime has these rates.

## Fresh serving comparison

| Engine | Median completion ms | Output tok/s | Median request TTFT ms | Median request TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| LunaFlux latest opt-in | 4658.0 | 109.92 | 1385.0 | 48.08 |
| vLLM | 4481.0 | 114.26 | 1368.0 | 45.70 |
| SGLang | 4537.5 | 112.84 | 1162.5 | 53.23 |

Output throughput is 512 generated tokens divided by wave completion time;
prompt tokens are not included. TTFT/TPOT are medians over requests, so adding
them does not reconstruct the wave time. SGLang starts generation earlier but
has a higher median per-request TPOT in this workload.

Six fresh starts ran in order LunaFlux, vLLM, SGLang, SGLang, vLLM, LunaFlux.
Each start had one discarded warmup and three measured waves. The six waves
per engine are **two independent starts**, not six independent starts.

| Engine | First start wall times ms | Second start wall times ms |
| --- | --- | --- |
| LunaFlux | 4602, 4625, 4614 | 4691, 4707, 4710 |
| vLLM | 4483, 4491, 4506 | 4455, 4468, 4479 |
| SGLang | 4541, 4524, 4543 | 4534, 4533, 4556 |

LunaFlux's start medians differ by about 2%. That variability matters at the
remaining few-percent scale; these results do not establish a universal fixed
gap or a narrow confidence interval.

All 144 measured request bodies were checked for identical input token IDs
across engines and starts. Input/output vectors are `[8192 × 8]` / `[64 × 8]`.
Greedy generation, temperature zero, top-p one, ignored EOS, varied token-ID
prompts and disabled prefix reuse are held fixed. Every output has 64 token IDs
and 64 timing entries. This is a fixed-work performance test, **not output
sequence equality or model-quality parity**. Short prompts, 32K, heterogeneous
request lengths and other concurrency levels require their own measurements.

## Where the captured completion difference is spent

Separate Nsight Systems captures use the same selected worker and artifacts.
The analysis clips GPU activity to the measured client's epoch interval,
excludes warmup, and unions kernels, copies and sets to avoid double-counting
overlap. These are single profiled captures, not the ordinary medians above.

| Captured interval | LunaFlux ms | vLLM ms | SGLang ms |
| --- | ---: | ---: | ---: |
| Client window | 4694.166 | 4469.954 | 4540.033 |
| GPU activity union | 4562.694 | 4366.143 | 4510.391 |
| No recorded GPU activity | 131.472 | 103.812 | 29.641 |
| Idle inside first-to-last GPU activity | 106.029 | 81.686 | 20.207 |
| Outside that GPU envelope | 25.443 | 22.126 | 9.434 |
| Between step kernel envelopes | 104.323 | 31.330 | 0.018 |
| Observed steps | 96 | 96 | 72 |

The **224.212 ms** LunaFlux–vLLM capture difference consists of **196.552 ms
more GPU activity plus 27.660 ms more inactive time**. Its activity attribution is:

| Component | Extra LunaFlux ms |
| --- | ---: |
| Single-query decode attention chains | +92.988 |
| Gate-up chains, all phases | +44.685 |
| QKV and associated normalization/rotary/KV-write chains | +24.342 |
| Other normalization | +22.096 |
| Multi-query prefill/mixed attention | +18.980 |
| Down projection | +6.307 |
| Output projection | +0.453 |
| Vocabulary head | -6.935 |
| Sampling and other kernels | -5.876 |
| Copy/set and overlap correction | -0.488 |
| Additional inactive time | +27.660 |
| **Total captured difference** | **+224.212** |

Projection roles are assigned from source order and attention/activation
anchors, not by kernel-name substring alone. Vendor kernels can serve several
roles; C1 GEMV is included. vLLM's separate SiLU is charged to its gate-up chain.
LunaFlux and vLLM have no unresolved category steps in this classifier. SGLang
has four pre-marker kernels and a partial final step retained as unresolved;
its whole-window GPU union remains usable, but an exact SGLang operator
waterfall is not claimed.

These are **observed component-time differences**, not independently removable
causal savings. LunaFlux's phases are inferred from executed graph symbols;
its unchanged current worker lacks fresh logical row markers. Equal 96-step
counts do not establish equal live-row/history vectors, and launch buckets are
not live batch sizes. The older equal-work result is documented separately in
[the exact mixed-work study](BENCHMARK_EXACT_MIXED_WORK_2026-10-07.md).

The LunaFlux–SGLang capture difference is **154.134 ms: 52.303 ms GPU activity
and 101.831 ms inactive time**. SGLang's different chunking and 72-step schedule
make per-phase totals non-equivalent. Do not apply the vLLM waterfall to it.

## Decode attention retains a dependency-heavy physical schedule

The selected symbol remains
`lunaflux_attention_decode_tile_compiler_v1_owned8_blockwise_f32_v4`, not the
new mixed-attention kernel. Its module digest is verified against module 4
embedded in the actual serving bundle. The C8 replay uses the captured launch:
grid `8 × 8 × 1`, block 64, history 8192, head dimension 128.

The exact-module replay passed identical-module bitwise comparison and the
probe's sampled FP64 oracle (`maxabs=0.000243755`). Its synthetic operands are
not a substitute for full-serving data or quality validation.

| Fresh isolated decode counter | Value |
| --- | ---: |
| Warp instructions | 41,451,264 |
| Eligible warps per scheduler cycle | 0.12 |
| Active warps per scheduler cycle | 1.00 |
| Issue-active percentage | 12.01% |
| Achieved occupancy | 5.64% |
| Tensor activity | 0% |
| Registers per thread | 148 |
| Resident blocks limited by shared memory | 2 |
| Source-correlated excessive shared wavefronts | 0 |
| Spilling requests | 0 |

The per-issue-active stall ratios include long-scoreboard 2.83,
short-scoreboard 2.03, MIO throttle 0.90 and barrier 0.85. These ratios are not
percentages of end-to-end time. They show an instruction/dependency chain with
too few ready warps, **not a fresh shared-conflict or spilling diagnosis**.

The [blockwise renderer](../kernels/luna_cuda_attention_tile_source/source_blockwise_decode.mbt)
assigns one subgroup per query head and shares K/V across the GQA group.
The selected tile is 32 keys with two copy stages: an 8192-key context traverses
256 tiles. [Its fold](../kernels/luna_cuda_attention_tile_source/source_blockwise_fold.mbt)
uses SIMT dot products, shuffle reductions, shared F32 probabilities, strict
`expf`, and an ordered scalar F32 value accumulation. Independent K/V refill
and page-lookup hoisting are already consumed by
[the copy lowering](../kernels/luna_cuda_attention_tile_source/source_decode_independent_pipeline.mbt).
Calling those features missing would repeat an obsolete diagnosis.

The captured vLLM pure-decode alternative is FlashAttention-2 split-KV with
64-key tensor-core tiles, a common partial grid of `1 × 3 × 64`, and a separate
combine kernel. The serving totals include the combine. That is a different
work decomposition and numerical law, not merely the same kernel with another
pipeline-stage count. Fresh vLLM decode counters were **not** recovered from
the interrupted full-serving counter run, so no new matched stall-ratio
comparison is claimed.

The next useful alternatives must expose more independent work or reduce the
SIMT fold/ownership cost. Simply increasing theoretical occupancy has already
failed in the earlier exact-work experiment. A BF16 tensor-core probability
path requires an explicit error contract; it must not be presented as a
bitwise-preserving rewrite of the current F32 probability law. Likewise,
partitioning changes reduction association and must include its merge cost.

## Gate-up fusion still costs more than the separate reference chain

The dominant captured LunaFlux gate launch is **1536 CTAs × 512 threads**;
vLLM's corresponding large GEMM is **384 CTAs × 256 threads**, followed by
SiLU/multiply. The complete multi-query chains cost **461.072 vs 416.596 ms**.
Decode gate-up is nearly tied: **103.543 vs 103.334 ms**.

| Fresh counter | LunaFlux fused gate-up | vLLM gate-up GEMM only |
| --- | ---: | ---: |
| Warp instructions | 93,020,160 | 20,812,800 |
| Achieved occupancy | 77.40% | 16.56% |
| Tensor activity | 38.33% | 57.81% |
| Barrier stall per-issue-active ratio | 7.08 | 3.24 |
| MIO throttle per-issue-active ratio | 3.33 | 0.46 |
| Source-correlated excessive shared wavefronts | 4,717,724 | 1,966,236 |
| Spilling requests | 0 | 0 |

LunaFlux is an exact-selected-module synthetic replay at 2048 rows, including
sampled scalar correctness (`maxabs=0`). vLLM is a completed selected GEMM
capture from the interrupted serving profiler. Geometry matches the serving
trace, but operand values/cache state differ. The **4.47× instruction ratio is
not a whole-chain ratio**: it excludes vLLM's separate SiLU instructions.
Use the serving chain totals for the performance conclusion.

[Sibling lowering](../kernels/luna_cuda_projection_aot/source_sibling_reuse.mbt)
and [its resource policy](../kernels/luna_cuda_projection_aot/source_sibling_resources.mbt)
trade smaller per-thread resource use for a much larger launched workgroup
population. Reuse exists; the issue is how much staging, ownership transport
and synchronization the selected decomposition requires. Higher occupancy
does not compensate for that work. The generated source has both retained and
shared epilogue alternatives; this investigation does not assign the entire
stall count to one epilogue without instruction-level correlation.

A useful next test is a larger-row GEMM schedule with a separately costed
SiLU/epilogue ownership choice, compared as a whole chain. That belongs in the
generic product-fold schedule and materialization plan. It is not evidence
that fusion should always be disabled or that conflicts alone explain the gap.

## A smaller normalization penalty has a specific implementation cause

In pure decode, normalization totals **23.859 ms vs 5.816 ms**. Our common
residual RMSNorm invocation uses 128 threads, 17 registers and 1024 shared bytes;
its C8 mean is about **6.73 microseconds**. The selected vLLM fused normalization
entries use 32 threads, 65–80 registers, zero shared memory and approximately
1.5–1.7 microseconds for the common C8 entries.

[The selected residual reduction](../kernels/luna_cuda_fused_parallel_aot/source_residual_program.mbt)
publishes 128 partial sums, performs a seven-level shared-memory reduction with
a barrier after every level, and rereads the rounded residual from global
memory for the normalization output. Including initial publication, that is
eight block-wide barriers in the emitted program.

The current F32 association and BF16 rounding are explicit contracts. A
register-retained/subgroup-owned lowering should either preserve the exact
reduction tree through a different ownership map or declare and validate an
alternative numerical policy. The source cost is concrete, but a new
normalization kernel has not yet been measured in this investigation.

## SGLang exposes the remaining runtime overlap opportunity

SGLang's pinned `event_loop_overlap` launches the current batch, then processes
the previous batch's queued result. Its CPU forward marker precedes the prior
step's last GPU kernel in **71 of 71 transitions**. LunaFlux's next graph launch
follows the prior step's GPU end in **95 of 95 transitions**.

[LunaFlux execution](../engine/device_step/paged_executor_run.mbt) launches the
ordered graph, records completion, waits, and only then exposes the executed
state to the subsequent completion/publication path. CUDA Graph replay reduces
launch overhead **inside** the step; it does not by itself overlap this boundary.

The marker and graph-launch boundaries are different CPU points, so their
counts do not establish how much all scheduler work overlaps. The independent
GPU intervals are the stronger observation: **104.323 ms between LunaFlux step
kernel envelopes, versus 0.018 ms for SGLang**. Against vLLM, comparing only
inter-step gaps would overstate the net opportunity: vLLM has more idle time
inside its step envelopes, and the whole-window inactive difference is only
27.660 ms.

The architectural direction is an explicitly bounded in-flight execution state
with separate submit, completion and retirement effects. Immutable next-step
metadata and device-resident token dependencies can be prepared independently;
KV reuse, cancellation, output publication and resource release must remain
completion-ordered. No request-path JIT, global mutable runtime or model-name
special case is needed. Changing `wait_completion` alone is not a safe solution.

## Priorities and falsifiable tests

1. **Decode attention:** compare complete partial/merge chains at exact live
   row/history vectors. Measure issue eligibility and operand/score dependency
   hotspots, not just occupancy or total conflict counts. The current captured
   decode-attention difference is 93 ms, roughly 2% of the LunaFlux window;
   that is an attribution bound, not a promised speedup.
2. **Gate-up geometry and ownership:** hold GEMM arithmetic fixed, vary row
   reuse and fusion ownership, include any separate activation. Require a
   reduction in full-chain GPU time, not merely fewer barriers or registers.
3. **Completion overlap:** retain cancellation and KV ownership invariants,
   then verify submission and GPU gaps. SGLang demonstrates a larger opportunity
   here than the current vLLM comparison does.
4. **Residual reduction and remaining QKV cost:** validate numerical-law
   alternatives and fresh per-step work before treating aggregate differences
   as isolated-kernel inefficiency. Output/down and vocabulary head are no
   longer the first priorities for this cell.

These changes fit the existing semantic/numeric → schedule → ownership/effects
→ backend-lowering layers. The evidence does not call for more IR layers merely
to increase the layer count. It calls for better executable schedules, explicit
accuracy choices and real overlap across the runtime effect boundary.

## Evidence and limits

- GPU: GB10 sm121, 48 SMs, UUID
  `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, PCI `0000000F:01:00.0`.
- Worker SHA-256:
  `1c5cfa3cb2538d70b5cea0fc59d93e2b8b3f0f94a9cc76a32e24e63e64854c87`.
- Decode module SHA-256:
  `318d74ff5b8ebfb5e0a97ea6daea16501a98a5ef1b67116350cf8b737266c185`.
- Gate-up module SHA-256:
  `bf10e22f8213f89a8a99f4530e1928051111f670101b84b0785201918ab949fe`.
- Pinned vLLM image:
  `sha256:73e1b0f3230a9377a3109d20be7f8607963a41a715eb3000bc647f5f73c198ff`.
- Pinned SGLang image:
  `sha256:3a9d399434923689565e1a72f5c3fc409323ce039b65bdbe5c2d6ed14abad3a9`.
- Remote evidence: `/home/wlc004s/lunaflux-final-gap-20261008.RcjTl7Ez`.
- Downloaded and verified: `/tmp/lunaflux-final-gap-verified-20261008.RuPfWoFH`.
- Archive SHA-256:
  `abff63c13fc457d39185a9636fc85010513536b3991f04833092321672f265c5`.
  All 1238 internal file hashes verify, including failures and raw traces.

GPU workloads were serialized. Serving used 64 GiB/no-swap limits; controller
and replay jobs used at most 8 GiB/no swap, with a 32 GiB MemAvailable admission
floor. Lowest sampled ordinary-run MemAvailable was 65.09 GiB. No GPU workload
remained at sealing.

The initial reference trace attempt hit an existing container name; a unique
name recovered both captures without removing the old artifact. Full-serving
LunaFlux NCU failed during startup; empty worker stderr and the launcher failure
are retained, and no deeper cause is asserted. vLLM NCU hit the 180-second
client timeout after two completed GEMM records; no fresh decode record is
claimed. Those failed runs are not throughput samples. The successful decode
and gate replays are explicitly isolated diagnostics with synthetic inputs.

The archive contains commands, raw request/token vectors, pinned container
metadata, Nsight reports and SQLite, counter CSV/JSON, source excerpts and
failed-run receipts. Automation is MoonBit `.mbtx`; the existing CUDA probe and
pinned reference instrumentation are reused. No production Python dependency,
deployment change or full-model quality admission is introduced.
