# Operation-level diagnosis and split-decode dispatch fix

## Confirmed dispatch defect

Profiling the combined Qwen bundle (`07a3c33` runtime, head256 and MLP8/group2
artifacts) reveals that crossing the split-decode context threshold discards
the smaller execution-graph buckets. `prepare_paged_ordered_executors` previously
prepared only a maximum-envelope split-decode executor. Its selection bypassed
the ordinary decode bucket table.

The measured C1 short-vector trace contains 5,208 postprocess calls with grid
1024 × 32 and mean 42.62 µs, despite only one live query token. The same segment
uses QKV grid 2048 instead of the bounded grid 512. These are actual serving
launches, not isolated candidate measurements. The residual/norm kernel also
retains a 1024-block envelope; that independent issue is not fixed by this change.

The fix constructs split-decode bucket owners at startup. A pure transformation
matches `(operation ID, function index)` across the attention expansion and
reuses only unchanged kernels' bounded launch contracts. New partial/merge
attention functions retain their own exact dimensions. Dispatch is a fixed-array
lookup; numerical kernels, descriptor layout, and token-step allocations are
unchanged. The maximum-envelope executor remains the fallback.

Regressions cover identity remapping after inserted attention operations,
preservation of partial/merge geometry, C1/C8 owner selection, threshold/phase
exclusion, and the uncaptured-slot fallback. Commit `57e7417` passes warning-denied
native checks and 172 device-step tests on both macOS and Linux.

## Uninstrumented serving A/B

Same combined kernel bundle, 1024-token profile and 1024/1024 scheduler budget/
chunk; only the runtime changes from `07a3c33` to `57e7417`. One excluded warmup
and two timed trials per cell, control then fixed. These are arithmetic means,
not confidence intervals or an order-balanced campaign.

| Input/output | C | Control tok/s | Fixed tok/s | Change |
| --- | ---: | ---: | ---: | ---: |
| 59/256 | 1 | 200.94 | 240.49 | +19.68% |
| 59/256 | 8 | 1170.29 | 1169.62 | −0.06% |
| 128/128 | 1 | 183.52 | 230.63 | +25.67% |
| 128/128 | 8 | 1080.74 | 1081.88 | +0.11% |
| 512/64 | 1 | 155.53 | 186.60 | +19.97% |
| 512/64 | 8 | 571.75 | 574.64 | +0.50% |
| 1528/32 | 1 | 83.01 | 89.89 | +8.29% |
| 1528/32 | 8 | 138.90 | 139.40 | +0.35% |

C1 post-first-token spacing falls from 4.894/5.260/5.508/5.935 ms to
4.082/4.130/4.413/4.855 ms. C8 is essentially unchanged, consistent with this
bundle's small-batch split policy. Long-prefill TTFT is unchanged: the fix removes
decode overhead rather than accelerating prefill.

All 72 timed fixed-arm request sequences match their corresponding control
cell, with one unique sequence per cell and exact requested token counts.
This is control agreement, not an independent cross-engine numerical qualification.
No new kernel arithmetic or native ABI was introduced. Production deployment
was not changed.

## Fresh per-operation comparison

### Fixed-worker trace verification

The r3 trace runs the committed `57e7417` worker with the same kernels. For
59/256 C1, the dominant postprocess launch is now grid 1 × 32, averaging
4.016 µs, rather than the control's split segment grid 1024 × 32 at 42.62 µs.
Decode QKV remains grid 512; MLP up/down remain grid 192/64 after the split.
Across the entire first timed window:

| GPU operation group | Control ms | Fixed ms |
| --- | ---: | ---: |
| QKV auxiliary/postprocess | 230.06 | 29.01 |
| QKV projection | 175.93 | 151.91 |
| MLP | 383.71 | 350.69 |
| Attention | 112.69 | 113.76 |
| Residual/norm | 68.99 | 68.91 |

This independently confirms the dispatch mechanism behind the uninstrumented
gain. Attention and residual arithmetic were not accelerated by this fix.
The fixed short-C8 trace remains consistent with control: QKV 288.47 ms,
output projection 184.94 ms, MLP 459.20 ms, and head 367.88 ms. Fixed long-C8
attention is still 732.58 ms. The bottlenecks below therefore remain present
after the fix, rather than being artifacts of an obsolete executable.

### Cross-engine control profiles

The following **representative first timed trial** is from the fully flushed r2
profiles, using the pre-fix combined LunaFlux runtime. Both timed trials and all
eight vectors are retained. Units are cumulative GPU kernel milliseconds per
request window, not wall latency or isolated-kernel throughput. Different
schedulers may execute different counts of forwards for the same request vector.

### Short decode-heavy vector: 59 input / 256 output, C8

| Equivalent operation group | LunaFlux | vLLM | SGLang | LF/vLLM |
| --- | ---: | ---: | ---: | ---: |
| QKV projection | 289.86 | 160.63 | 162.18 | 1.80× |
| QK normalization / rotary / KV-write and adjacent auxiliary work | 39.34 | 52.87 | 94.89 | 0.74× |
| Attention, including merge kernels | 155.08 | 156.44 | 142.77 | 0.99× |
| Output projection | 184.82 | 91.86 | 92.96 | 2.01× |
| MLP projections + activation | 460.17 | 400.52 | 406.07 | 1.15× |
| Residual / other RMSNorm | 66.81 | 15.37 | 23.31 | 4.35× |
| LM head | 368.01 | 190.59 | 192.18 | 1.93× |
| Sampling / argmax kernels | 36.63 | 6.25 | 2.51 | 5.86× |
| Other kernels | 0.39 | 8.31 | 8.78 | — |

Baseline linear attribution checks the repeated 28-layer × four-linear + head
order (113 matrix launches per complete forward), validates head-sized terminal
launches, and keeps auxiliary kernels separate. Norm/rotary/copy work between
QKV and output projection is grouped with QKV auxiliary work; this is not a
comparison of a fused LunaFlux operation against a bare baseline GEMM. Baseline
sampling conversion/copy work outside identified sampler kernels remains in
`other`; the sampling row alone is not the full sampling-service cost.

### Long vector: 1528 input / 32 output, C8

| Operation group | LunaFlux | vLLM | LF/vLLM |
| --- | ---: | ---: | ---: |
| Attention | 737.90 | 188.23 | 3.92× |
| MLP + activation | 524.26 | 189.34 | 2.77× |
| QKV projection | 273.07 | 77.98 | 3.50× |
| QKV auxiliary work | 55.10 | 22.74 | 2.42× |
| Output projection | 137.21 | 41.57 | 3.30× |
| Residual / other RMSNorm | 19.15 | 8.70 | 2.20× |
| LM head | 56.57 | 28.18 | 2.01× |

SGLang attention is **175.97 ms**, making LunaFlux 4.19× slower for that group.
Its long-C8 windows end in a partial model forward (39 complete 113-matrix
groups followed by 112/81 matrix launches in trials 1/2); the automated full-
forward validation therefore fails. Do not present its positional linear
breakdown as fully validated. Attention is identified by kernel names and does
not depend on that positional attribution.

LunaFlux long-C8 GPU busy time is 1808.44 ms inside an 1846.95 ms kernel span
(97.9%). This instrumented trace strongly locates the remaining bottleneck in
GPU work, not merely HTTP, host scheduling, or a missing graph launch wrapper.

## Why individual operations remain slow

1. **Split-decode composition bug — fixed.** C1 starts with bounded projection/
   postprocess launches, then switches to full-envelope launches at the split
   threshold. The uninstrumented A/B isolates its 8–26% C1 cost above.
2. **Small-row output projection is underdistributed.** The dominant short-C8
   LunaFlux launch uses 8 CTAs × 256 threads, averaging 25.23 µs. The baseline
   uses 64 CTAs × 32 threads for its corresponding small output projection.
   Eight CTAs cannot occupy all 36 SMs. The generic compiler needs a smaller
   output-column grouping for this shape, not more model-specific branches.
3. **QKV and head schedules still differ materially from the baseline.** QKV
   averages 38.91 µs with 32 CTAs versus baseline ~22 µs with 256 CTAs. Head256
   uses 1187 CTAs × 256 threads, 40,960 shared bytes/CTA, and ~1.43 ms; the
   baseline uses 9496 CTAs × 32 threads, 16,896 shared bytes/CTA, and ~0.74 ms.
   LunaFlux's matrix reduction advances K by 16; the observed baseline small-
   matrix kernel names expose K128 schedules with one/two pipeline stages.
   These are measured schedule differences and source-level optimization
   targets, **not** measured stall-counter or occupancy attribution. Register
   counts alone do not prove spills or the dominant stall reason.
4. **Long-prefill attention remains expensive despite existing split work.**
   In the long-C8 trace, the partitioned partial kernel costs 318.58 ms (112
   calls), its merge 19.64 ms, and the wide c314 kernel 271.88 ms (224 calls).
   Together those prefill-named kernels account for 610.10 ms; ordinary decode
   kernels account for another 123.11 ms and split-decode partial/merge 4.69 ms.
   This is not 'no parallel attention': it is an expensive selected physical
   schedule plus different prefill/mixed forward granularity. The exporter
   still sets `supports_async_copy=false`, so the new overlapped transfer
   schedules are not installed. A universal async switch is not justified by
   the earlier shape-dependent microbenchmarks.
5. **Residual and sampling still have their own problems.** Fused residual/
   RMSNorm retains 1024 blocks even at C1/C8 (~4.6–4.8 µs per layer operation).
   The greedy kernel is ~142 µs per launch; its per-row vocabulary reduction
   needs comparison with segmented vocabulary reduction, not an HTTP change.
   These remain separate optimization tasks; they are not silently counted as
   fixed by the split-decode change.

The next compiler work should target backend-neutral work-distribution,
reduction tiling, transfer pipelines, and result-demand-aware graph composition.
CUDA-specific mapping stays in lowering. Adding more abstract compiler passes
without connecting these decisions to the selected executable cannot fix these
measured costs.

## Trace collection scope

All engines use Qwen3-0.6B BF16 on RTX 5060 Ti, identical token-ID vectors,
greedy fixed output lengths, no prefix cache, C1/C8, and excluded warmups.
Vectors are 59/256, 128/128, 512/64, and 1528/32 input/output tokens.

Nsight node-level tracing is diagnostic and its request throughput is not an
uninstrumented serving benchmark. A disposable supervisor forwards profiler
environment settings; the worker and kernel artifacts are unchanged.

The initial r1 traces lost final buffered events when process groups stopped.
Empty or partially covered long-vector windows must not be interpreted as zero
cost or compared. The r2 collection uses periodic CUDA buffer flushing and
explicit collection stop before process teardown. Raw files remain on the host
under `/dev/shm/lunaflux-op-profile-20260908-r1`, `-r2`, and `-r3`; no run
overwrites older traces. R2 contains all three engines; r3 verifies the fixed
LunaFlux worker. SGLang's partial long-C8 linear window limitation is documented
above, not silently treated as a validated complete forward.

Uninstrumented request results are under
`/dev/shm/lunaflux-baselines-20260906-r1-lunaflux-split-bucket-control-20260908-r1`
and the corresponding `split-bucket-fixed-20260908-r1` directory. Build logs,
launch descriptions, and the clean committed source are under
`/dev/shm/lunaflux-split-bucket-20260908-r1`.
Operation CSV tables are also downloaded without overwrite to
`/private/tmp/lunaflux-operation-results-20260908.q8vVnF` on the development host.
The owned profiling/benchmark servers were stopped; the final NVIDIA compute
process inventory was empty. No production service was deployed or modified.
