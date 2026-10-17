# GLM: executed work, handoff waits, and graph bubbles

## Scope and corrections

This investigation separates (1) mathematical/padded work, (2) executed
support instructions and device dependencies, (3) inter-kernel/rank gaps, and
(4) end-to-end time. None is a substitute for the others. Comparison stays
outside the compiler. No payload hashing, numerical waiver, checkpoint edit,
or new production numerical policy is involved.

Native implementation is `9e3b009e`, runtime
`/tmp/lunaflux-glm-expert-tp-shared-vector-20261017-v1`. The local vLLM checkout
is `../vlm` at `10f9b5d74fb4110adfc310278e029af4faea0565`; the installed
reference image reports `0.1.dev20051+g487ecf187`. They are **not the same
source version**. Installed source snapshots are retained in
`/tmp/lunaflux-glm-installed-trace-sources-20261017-v2`.

Three earlier interpretations need correction:

1. The selected native expert path already has packed operand tiles, bulk
   copies, two stages, separate producer/consumer roles, and persistent CTAs.
   Older scalar-copy probe counters do not describe it.
2. Long-scoreboard stalls are not automatically weight-load latency. The
   new instruction samples mostly attribute native expert stalls to the
   **barrier-status dependency at publication/retirement wait loops**.
3. A higher tensor-active percentage is not automatically more useful work.
   This CUTLASS component executes exactly eight times our tensor instructions
   because it computes 128-row rather than 16-row tiles for sparse expert groups.

## Unprofiled full-model result

Same six literal prompts, input lengths `[22,23,22,23,24,36]`, C1, one excluded
warmup, TP2, no prefix reuse/speculation. Native output lengths are
`[8,4,8,8,8,3]`; vLLM lengths are `[4,4,8,8,8,3]`. All native input/output ID
arrays remain unchanged across the old/new/graph ablations. Their agreement
does not establish model quality or agreement with the reference.

| Prompt | Old native repeat 1/2, ms | New native repeat 1/2, ms | New collective graph, ms | Retained vLLM, ms |
| --- | ---: | ---: | ---: | ---: |
| Fact | 775 / 797 | 780 / 777 | 817 | 387.121 |
| Arithmetic | 504 / 500 | 494 / 499 | 499 | 392.841 |
| Chinese | 784 / 783 | 771 / 774 | 781 | 657.336 |
| JSON | 812 / 810 | 784 / 792 | 799 | 670.561 |
| Code | 821 / 813 | 793 / 795 | 796 | 680.855 |
| Retrieval | 509 / 504 | 493 / 493 | 485 | 379.882 |

Mean native totals: **4206 → 4122.5 ms, 1.99% less time**. Mean TTFT sums:
1693 → 1636.5 ms; post-first-token sums: 2513 → 2486 ms. The shared BF16
transport component win primarily targets prefill; the small decode movement
must not be attributed to a changed decode kernel (its component was unchanged).

Retain the previously reported Chinese/code/retrieval cohort rather than
selecting a new favorable subset: 19 output tokens, native new totals
2057/2062 ms versus retained reference 1718.073 ms, **19.73–20.02% longer**.
That corresponds to 9.24/9.21 versus 11.06 output tokens/s including prefill.
These are matched input/output-count workloads, not identical generated-token
streams. Native timing is internal resident completion; reference timing is
buffered HTTP wall time. They are not identical timer boundaries.

Including **all five** equal-output-count prompts, not just the historical
three, gives 3335/3353 ms versus 2781.475 ms: **19.90–20.55% longer**.
The fact prompt is excluded only from that ratio because it emits twice as
many native output tokens; its raw result remains in the table.

The new collective-inclusive graph totals 4177 ms, versus new interval runs
4115/4130 ms. It is **not a demonstrated end-to-end winner**. There is only one
new whole-graph run; do not invent a confidence interval. No long-context or
concurrent-serving conclusion follows from this short-input cohort.

Raw native roots are under `/tmp/lunaflux-model-bench-`:
`glm-gap-interval-run-20261017-v1`, `glm-shared-vector-old-20261017-v2`,
`glm-shared-vector-next-20261017-v1`, `glm-shared-vector-next-20261017-v2`,
`glm-shared-vector-next-graph-20261017-v1`. Reference is
`/tmp/lunaflux-reference-bench-vllm-glm-bulk-tiles-20261017-v1`.
Reproducible ID/timing checks: `scripts/summarize-glm-replay-ablation.mbtx`;
saved analysis: `/tmp/lunaflux-glm-vector-ablation-20261017-v1/runs.jsonl`.

## Fresh selected native serving trace

Root `/tmp/lunaflux-model-bench-glm-selected-node-20261017-v1`, both ranks,
`--cuda-graph-trace=node` without all-API tracing. A literal 16-input/2-output
request follows an excluded warmup. Steps 3/4 are the measured prefill/decode;
steps 1/2 are warmup. This is **not** the six-prompt unprofiled timing run.
Both ranks report selected collective-inclusive replay and explicit release.

| Egress GPU accounting | Prefill step 3, ms | Decode step 4, ms |
| --- | ---: | ---: |
| Complete observed step span | 291.601 | 73.752 |
| Compute kernels, union | 189.666 | 68.544 |
| Collective kernels, union | 42.956 | 3.749 |
| No observed kernel running | 58.978 | 1.459 |
| Largest kernel-free interval | 2.056 | 0.009824 |
| Compute/collective overlap | 0 | 0 |
| Graph submissions | 91 intervals | 1 whole graph |

These unions do not add overlapping kernels twice. Distinct stream IDs exist,
but there is **no observed simultaneous kernel coverage** in either selected
window. GPU kernel coverage is not evidence that those kernels issue useful
instructions while resident.

Node tracing perturbs prefill: warmup prefill is 235.019 ms, measured prefill
291.601 ms; kernel-free time is 21.295 versus 58.978 ms. Earlier graph-level
interval tracing also inflated decode to 181.98 ms despite unprofiled decode
being around 74–80 ms. Do not advertise the profiled 59 ms prefill gap as a
removable production budget.

The fresh whole-graph decode has only 1.46 ms kernel-free time (~1.98% of its
span). Even eliminating all of it would not explain a ~20% completion gap.
That is a bound within this instrumented window, not a universal speedup bound.

Exclusive native family times on egress:

| Family | Prefill, ms | Decode, ms |
| --- | ---: | ---: |
| Routed experts including quantization/packing | 94.859 | 12.099 |
| Recurrent/KDA family | 34.005 | 23.877 |
| Sparse family including its projection/front-end | 21.743 | 11.149 |
| Shared experts | 19.725 | 6.560 |
| Text head/norm/sampling | 5.815 | 5.918 |
| Other hyper-connection family | 5.012 | 4.160 |
| Dense FFN | 3.900 | 2.048 |
| Router | 2.971 | 1.814 |
| MoE combine/collective packing | 1.637 | 0.920 |

More specific attribution:

- Prefill routed gate/up: **56.470 ms**, down: **26.800 ms** (42 calls each).
  They are the largest compute component, not necessarily the largest *gap*.
- Prefill activation-global-quantization reduction: **8.364 ms**, 42 one-CTA
  calls; input-global reduction another **1.739 ms**. Packing and grouping are
  smaller but are included, not hidden behind a GEMM-only number.
- Decode recurrent input projections: **16.212 ms**; recurrent output role
  `15`: **5.731 ms**. Their combined 21.944 ms is ~29.8% of the step span.
  The recurrent delta itself is only **0.600 ms**.
- Decode retained-score sparse attention itself: **0.219 ms**, whereas sparse
  output projection is **3.549 ms**. Calling the entire 11.149 ms sparse family
  an attention-core bottleneck would be wrong on this short history.
- All HC-containing leaves total 450 calls, 5.738/4.796 ms; this includes HC
  leaves classified inside sparse above. **Do not add this overlapping total.**

Ingress decode has 8.463 ms in collective kernels versus egress 3.749 ms.
Its first collective alone lasts 3.111 ms, while the longest egress decode
collective is 0.256 ms. These are rank-local elapsed durations, including
arrival/dependency waiting, not pure network transfer times. Cross-host clocks
have no calibrated error bound here; do not subtract absolute timestamps to
invent rank-arrival skew. The decoder head runs only on egress.

Analysis: `scripts/analyze-glm-selected-trace.mbtx` and
`scripts/analyze-gpu-overlap.mbtx`; roots
`/tmp/lunaflux-glm-selected-node-{,ingress-}analysis-20261017-v1` and
`/tmp/lunaflux-glm-selected-{prefill,decode}-overlap-20261017-v1`.

## Actual vLLM serving trace: the missing comparison is now collected

The successful run is
`/tmp/lunaflux-reference-bench-vllm-glm-egress-trace-20261017-v6`.
Only the egress rank is instrumented, avoiding the failed dual-profiler
startup footprint. Both ranks still execute the same model. TP2, literal
16-token input, two output tokens, excluded warmup, no prefix reuse or draft
model. Profiled HTTP time is 227.441 ms; this is **not** a replacement for the
six-prompt unprofiled acceptance numbers above. Different generated tokens
remain a limitation, especially for routed-expert distributions.

The selected serving trace contains the actual SM120 block-scaled CUTLASS
128x128x256, two-stage, 384-thread, 48-CTA grouped kernel, not just a similar
component. Its full instantiated symbol matches the mechanism examined above.
BF16 GEMV and WMMA projections, fused HC and auxiliary shared-expert execution
are also observed. The local newer checkout is still not the installed version.

GPU windows are bounded by the observed embedding and sampled-token tail:
prefill `[106340416,256886080]` ns, decode `[257052896,325154400]` ns.
CPU NVTX `execute_context` returns before GPU completion; using its end time
would silently truncate the model. The trace has one GPU context/process.

| Egress timeline, ms | LunaFlux prefill | vLLM prefill | LunaFlux decode | vLLM decode |
| --- | ---: | ---: | ---: | ---: |
| Observed model-step span | 291.601 | 150.546 | 73.752 | 68.102 |
| Compute union | 189.666 | 137.499 | 68.544 | 63.845 |
| Collective union | 42.956 | 5.549 | 3.749 | 2.108 |
| Copy union | 0 | 0.066 | 0 | 0.038 |
| Uncovered GPU interval | 58.978 | 7.431 | 1.459 | 2.111 |
| Shared-expert overlap with other kernels | 0 | 6.868 | 0 | 3.500 |
| Compute/NCCL overlap | 0 | 0 | 0 | 0 |

**Do not interpret the profiled prefill ratio as the production speed gap.**
The native warmup/measured discrepancy and enlarged rank-wait time establish
profiler perturbation. Both measured and negative captures are retained.
Decode is much more stable here, but is still one short-history instrumented
sample, not the six-request distribution or a long-context/C16 result.

### Complete accounting of the observed decode gap

Every kernel is attributed exactly once; HC leaves inside the native sparse
family are moved into HC rather than counted twice. The vLLM attribution uses
90 observed alternating attention/FFN HC boundaries, three dense FFNs, 42 MoEs,
34 KDA and 11 sparse layers, and the actual auxiliary stream per region.
Unsupported boundary counts are rejected rather than silently classified.

| Exclusive category, summed kernel ms | LunaFlux | vLLM | Native minus reference |
| --- | ---: | ---: | ---: |
| HC | 4.796 | 1.366 | +3.429 |
| Embedding/head/norm/sampling | 5.918 | 3.442 | +2.475 |
| Sparse chain, excluding HC | 10.513 | 8.847 | +1.666 |
| NCCL | 3.749 | 2.108 | +1.641 |
| Shared-expert compute | 6.560 | 6.340 | +0.220 |
| Dense FFN | 2.048 | 2.460 | -0.412 |
| Router/routed experts/packing/combine | 14.833 | 17.153 | -2.320 |
| Complete KDA chain | 23.877 | 27.994 | -4.117 |
| Sum before overlap correction | 72.294 | 69.711 | +2.583 |

The exact timeline identity is:

```text
  +2.583044 ms  difference in summed kernel durations
  +3.757757 ms  reference kernel-overlap correction (native overlap is zero)
  -0.037984 ms  reference-only copy coverage
  -0.652161 ms  native minus reference uncovered intervals
  =5.650656 ms  observed step-span difference (73.752160 - 68.101504)
```

This is an **accounting identity**, not independently removable speedups.
Of the 3.757757 ms overlap correction, 3.500093 ms is shared-expert work
coexisting with other streams. The remaining trace-node intersections are
also union-corrected, not labeled a new optimization. Removing concurrency
can change each kernel's time through contention. Do not add an overlap saving
to the same kernels' savings twice.

Important revisions to the earlier priorities:

- KDA projections alone are **21.944 vs 27.275 ms**. They consume much native
  time but are **not a demonstrated decode deficit**. The short-history KDA
  core is 0.600 ms native versus 0.460 ms reference; rewriting the recurrence
  cannot recover the multi-millisecond head/HC costs.
- Native routed gate/up/down total **10.871 vs 12.333 ms**. Their component
  publication waits are real, but they are not the observed largest decode
  gap. The full routing/packing/combine scope is faster too in this sample.
- vLLM shared compute totals 6.340 ms, of which **3.500 ms overlaps** and
  2.840 ms has no other kernel active. Ours serializes all 6.560 ms.
- HC is **450 native launches vs 181 reference launches** and 4.796 vs
  1.366 ms. This is measured boundary materialization/fusion cost, not simply
  CPU launch overhead: both are executing replayed device work.
- vLLM's selected BF16 vocabulary shard takes **3.396544 ms**, followed by a
  **0.059552 ms all-gather**. Native's unsharded head takes **5.652384 ms**.
  Head communication is already included in the NCCL table row.
- NCCL has 90 native F32 all-reduces versus 91 reference BF16 all-reduces
  plus one vocabulary gather. Reference has extra embedding communication,
  yet lower summed elapsed time. Neither trace shows compute hiding NCCL.
  Payload sizes were not exported in NVTX binary fields, so the source-derived
  payload ratio below is not relabeled as a measured byte counter.

There is even avoidable work in this installed reference: its model calls
`self.gate` externally and passes the same gate to the installed MoE runner,
which calls it again. Two projection/cast sequences are observed before each
top-k. This explains why it is unsafe to assume every reference region is
optimal, or to copy the newer checkout's behavior into the installed trace.

### Prefill targets differ from decode targets

| Summed compute, ms (profiled, not production budget) | LunaFlux | vLLM |
| --- | ---: | ---: |
| Shared experts | 19.725 | 7.527 |
| Sparse chain excluding HC | 21.017 | 9.508 |
| KDA chain | 34.005 | 26.701 |
| Router/routed/packing/combine | 99.467 | 93.740 |
| Routed gate/up + down subset | 83.271 | 87.521 |
| HC | 5.738 | 2.456 |

The routed GEMMs are not where this prefill comparison loses time either.
Native non-GEMM work around them costs **16.197 ms**, versus **6.219 ms**
reference (different fusion/rounding scopes, not identical individual ops).
Two one-CTA global-max scans account for **10.103 ms** natively, including
8.364 ms for activation scales. The selected source scans all
`rows * 8 * 1024` values in one 256-thread block before packing. A hierarchical
finite-absolute-max reduction can preserve the scale/error contract; replacing
it with the reference's checkpoint/global-scale convention would instead be a
numerical change and is not justified by timing alone.

The shared BF16 prefill source has a separate physical deficit from the low-bit
routed path. For 16 rows and one shared expert it has one 16-row group, hence
only **16 useful gate/up CTAs** for N1024 at N-tile64, on a 48-SM GB10. It
launches 1024 CTAs, but the other groups return. Gate/up folds K4096 in K32
chunks with two CTA barriers/chunk and synchronous vector staging. The
reference's actual first shared gate projection launches **128 CTAs**, with
the observed WMMA `16x16_128x2` symbol; the first native/reference calls take
0.362/0.152 ms, with different fused epilogues. The native down has 64 useful
CTAs (4096 launched) and the analogous K32 loop. Vectorizing its copies was a
real improvement, but did not change this available parallelism or staging.

The fused KDA input launch similarly reserves decode geometry (3144 CTAs),
while a 16-row prefill has 197 useful physical tiles; inactive row tiles return
before MMA. This is **launch-domain overhead**, not 16x excess matrix FLOPs.
Its prefill K32 two-slot pipeline waits/publishes every K32 chunk. Its decode
branch is a different SIMT schedule and already wins the matched projection
comparison. Tune these shapes separately instead of replacing both routes.

Analysis roots:
`/tmp/lunaflux-glm-reference-{prefill,decode}-overlap-20261017-v1`,
`/tmp/lunaflux-glm-reference-{prefill,decode}-attribution-20261017-v2`,
and `/tmp/lunaflux-glm-selected-node-analysis-20261017-v2`.
Reproduction uses `analyze-glm-reference-execution.mbtx` (full symbol and region
ledger), `analyze-glm-selected-trace.mbtx` (exclusive aligned categories), and
`analyze-gpu-overlap.mbtx` (interval unions, not summed concurrent durations).

## Actual selected expert bodies versus installed CUTLASS component

The new fixture includes GLM's clamp-10 epilogue. After symbol-prefix
normalization, gate/up's **13,020 source characters** and down's **8,984**
match the selected stage-1 AOT source exactly. This is direct source comparison,
not hashing. Launch geometry matches too: grid48, block288. It avoids profiling
the older scalar-copy implementation by mistake.

Component geometry: R22, H4096, local I1024, top8, 288 distinct expert banks,
identical route-index formula `(row*7+slot)%288`, 155 live groups. Both use the
installed GB10, one GPU at a time. Payload values are synthetic and differ;
these are not literal model-operand correctness comparisons. Native keeps F32
TP partials and fuses activation; the reference component has its own BF16
epilogue and surrounding chain. Full-chain semantics and launch scopes differ.

| NCU metric | Native gate/up | CUTLASS gate/up | Native down | CUTLASS down |
| --- | ---: | ---: | ---: | ---: |
| Profiled kernel ms | 3.161 | 3.270 | 1.610 | 1.739 |
| Warp instructions, million | 113.224 | 79.040 | 65.343 | 64.515 |
| Address-unit instructions, million | 28.820 | 3.668 | 19.376 | 3.571 |
| Control-unit instructions, million | 28.637 | 1.683 | 8.568 | 2.199 |
| LSU instructions, million | 8.623 | 14.443 | 6.270 | 9.689 |
| Tensor instructions, million | 2.540 | 20.316 | 1.270 | 10.158 |
| Uniform instructions, million | 0.246 | 15.304 | 0.367 | 10.192 |
| Eligible warps/scheduler/active cycle | 0.100 | 0.066 | 0.104 | 0.103 |
| Issue-active % | 9.50 | 5.09 | 9.10 | 7.96 |
| Active warps % | 18.75 | 22.92 | 18.75 | 22.93 |
| Long-scoreboard stall % | 36.79 | 4.06 | 56.04 | 10.61 |
| CTA-barrier stall % | 1.69 | 1.84 | 6.11 | 6.89 |
| Math-pipe throttle % | 0.33 | 6.21 | 0.39 | 6.16 |
| Tensor-active % | 2.71 | 20.97 | 2.67 | 19.71 |
| L2 read sectors, million | 24.382 | 45.790 | 12.936 | 22.959 |
| L2 hit % | 5.54 | 50.21 | 11.89 | 51.95 |
| Registers/thread | 64 | 168 | 56 | 168 |
| Static/dynamic shared bytes | 78368 / 0 | 0 / 89088 | 41504 / 0 | 0 / 89088 |

Both use 48 persistent CTAs and two stages; CUTLASS uses 384 threads. Its
128-row tile executes **8x** the tensor work of our 16-row tile here. Useful
row utilization from this routing is about 7.10% versus 0.887%. Our smaller
tile is an advantage for this distribution, but support/wait work consumes
much of that advantage. It would be incorrect to increase M merely to match
the reference's tensor-active percentage.

Native total instructions are 43.2% higher for gate/up **in this capture**, but
these are dynamic counters, including polling, not a fixed static instruction
multiplier. A separate SourceCounters pass reports native totals 34.581M and
27.415M, versus CUTLASS 76.951M and 63.617M. Instrumentation/cache state changes
the number of wait-loop iterations. Therefore neither counter total should be
promoted to a universal implementation ratio. Do not mix samples and aggregate
instructions from different passes as if collected simultaneously.

The address/control pipe totals include waiting-related support instructions;
they are not proof that address calculation alone consumes that much time.

The individual profiled kernel times are close; they do **not** demonstrate
that native expert matrix kernels explain the remaining end-to-end 20% gap.
NCU timing is not an unprofiled speed benchmark. DRAM metrics were unavailable
in this GB10 capture; L2 sectors/hit rate are not measured DRAM bandwidth.

Converting read sectors to bytes (32 bytes/sector) gives native/reference
gate reads of 780.235/1465.274 MB and down reads of 413.964/734.681 MB.
Dividing by this capture's kernel time gives L2 read traffic rates of about
246.8/448.0 GB/s and 257.2/422.4 GB/s. These are **L2 traffic**, not useful
weight bytes or DRAM rates. The hit-rate difference is substantial; the reported
aggregate hit rate is not read-miss sectors, so multiplying it by read bytes
would not establish external-memory traffic. Read-hit/read-miss counts remain
a needed discriminant.

The key counterfactual is already visible: CUTLASS performs 8x MMA yet takes
roughly the same profiled time. This is inconsistent with treating tensor
instruction count alone as the limiting cost here. It is consistent with
additional matrix work fitting behind data movement/other dependencies, but
that explanation still needs stage timing and unprofiled replay. Conversely,
replacing our tile16 with tile128 solely to increase tensor-active percentage
would remove an existing useful-work advantage without proving a latency gain.

Raw reports: `/tmp/lunaflux-glm-work-stalls-{native,reference}-20261017-v1`.
Fixture: `/tmp/lunaflux-glm-fp4-performance-selected-clamp-20261017-v1`;
reference: `/tmp/lunaflux-glm-cutlass-component-r22-20261017-v3`.
Native packed-payload/scale oracle and bitwise old/new complete chain pass,
zero nonfinite/device-error values. Reference replay success has no independent
expected-output oracle and is **not** a correctness pass.

The installed full-model loader also warns that `w1_weight_scale_2` must match
`w3_weight_scale_2` and that accuracy may be affected. The warning is retained;
we neither changed its checkpoint nor waived our numerical contracts. Reference
speed, reference correctness and native/reference token agreement are separate
questions throughout this report.

## The instruction PCs identify the wait mechanism

Separate bounded source captures:
`/tmp/lunaflux-glm-work-stalls-{native,reference}-source-20261017-v1`.

Native gate/up's two hottest wait sequences are:

```text
YIELD
SYNCS.PHASECHK.TRANS64.TRYWAIT predicate, barrier, parity
@!predicate BRA retry
```

The two retry branches account for **58,754/70,662 = 83.1%** of gate/up's
long-scoreboard samples. Down's five corresponding retry branches account for
**71,450/100,415 = 71.2%**. Following the branch targets and barrier-address
construction separates the two sides of the handoff:

| Native sampled retry path | Long-scoreboard samples | Share of kernel's long-scoreboard samples |
| --- | ---: | ---: |
| Gate/up consumers waiting for `bulk_ready` | 54,591 | 77.3% |
| Gate/up producer waiting for `bulk_retired` | 4,163 | 5.9% |
| Down consumers waiting for `bulk_ready`, four unrolled K epochs | 68,663 | 68.4% |
| Down producer waiting for `bulk_retired` | 2,787 | 2.8% |

Gate/up consumer branch `0x325b84320` enters retry `0x325b87810`; it is in
the `threadIdx.x >= 32` path and addresses shared base + `0x13200` (`ready`).
The producer path starts at `0x325b849f0`, checks epoch >= 2, and addresses
base + `0x13210` (`retired`), entering retry `0x325b87850`. Down's analogous
addresses are base + `0xa200` / `0xa210`. Its first four retry PCs are
`0x325b929c0`, `0x325b929f0`, `0x325b92a20`, `0x325b92a50`; producer retry is
`0x325b92a90`. These are identities in the retained capture, not stable ABI.
Consumer/producer initial-check counts also match the launched work: gate/up
158,720 = 155 groups × 8 column tiles × 8 consumer warps × 16 K epochs;
17,360 = 155 × 8 × 14 producer reuse epochs.

Thus the sampled waiting is predominantly **consumers awaiting publication**,
not producer buffer-reuse stalls. This still does not distinguish transfer
service time from late producer issue or barrier polling itself. Eight consumer
warps versus one producer warp also weights sample counts; these percentages
are not fractions of kernel elapsed time. A ready-stage latency/progress
ablation, not simply adding stages, is needed to resolve that final cause.

The reference's wait slow path includes `NANOSLEEP.SYNCS` and a blocking
`SYNCS.PHASECHK.TRANS64`, not only a tight TRYWAIT loop. Its hottest executed
instructions are fragment loads and MMA, rather than the native retry loops.
Some source instruction counters are zero while PC samples are nonzero
(different collection passes); zero there is not proof that an instruction
never executes. No percentage of sampled stalls is a percentage of wall time.

Native source-correlated excessive shared wavefronts also remain: 634,880 for
each captured kernel, concentrated in vector scale loads. They are visible,
but the wait-location evidence does not justify making conflict-count zero the
goal. The reference also has excessive wavefronts. Measure complete chain time.

## Why graph replay does not remove the gap

1. `CaptureDecodeCollectives` explicitly captures one-row decode; prefill
   remains interval replay (`compiler/parallel_ir/submission.mbt`). It cannot
   directly improve the prefill computation described above.
2. It replays the same ordered effects. Capturing 90 collective sites neither
   reduces their payload nor overlaps independent shared/routed branches.
3. Most interval submissions in the earlier queue trace were already ahead of
   the GPU (82/91 calls returned before GPU start). CPU submission can overlap
   GPU computation without overlapping two GPU kernels.
4. The current full-graph decode has little between-kernel empty time. Its
   dominant remaining time is inside compute and communication kernels.

The generic parallel IR supports dependency/effect reasoning and compute /
communication lanes. The selected GLM owner consumes an ordered submission
plan, not a general multi-compute-lane schedule. More graph capture is not the
same thing as changing that critical path.

## Source differences that still need serving-level causal budgets

| Difference | Evidence | Required discriminating measurement |
| --- | --- | --- |
| Publication/retirement wait strategy | Current AOT source and sampled SASS above | Separate ready/retired wait samples; bounded wait-policy ablation with unchanged numerical/ownership IR, then unprofiled whole chain |
| Expert physical row work | Actual tile16 vs CUTLASS128, exact8x MMA count | Real per-layer route distribution, live groups, padding and complete quantize/pack/activation/combine cost; synthetic R22 is not a substitute |
| Shared/routed branch ordering | Native no overlap; installed serving trace overlaps 6.868/3.500 ms prefill/decode | Native same-numeric DAG ablation; observed overlap is not automatically an equal speedup under different contention |
| Collective placement/payload | Native coalesces `(8*4096+4096)` F32 partials before expert combine; reference runner combines local outputs before late all-reduce on its ordinary TP path | Actual reference collective dtype/count and per-rank arrival; numeric contract must explicitly permit moving reduction/rounding |
| HC materialization | Installed model defers post and fuses post/pre across sublayers/layers; native parallel HC still materializes boundaries | Match complete HC region, count bytes and launches; no assumption all HC time is removable |
| KDA/sparse projections and representation | Native decode input/output projections dominate; reference merges KDA projections and uses compressed attention representation | Actual matched reference projection chains and cache policy, not a generic cuBLAS call or attention-only probe |
| Vocabulary work partition | Native full head 5.652 ms; actual reference BF16 shard 3.397 ms plus 0.060 ms gather | Native column-shard plus stable global selection ablation, with unchanged independent column arithmetic |

For the collective row widths, native currently transports 147456 bytes/row
at a coalesced MoE boundary. A 4096-wide BF16 late sum would be 8192 bytes/row,
an 18x arithmetic payload difference, **conditional on that reference route**.
This is not measured 18x network time and not automatically a legal numerical
rewrite. Native reduces each F32 expert partial before its publication/rounding;
moving a local sum across that boundary changes floating-point semantics.

Vocabulary is a distinct, more directly visible partitioning gap. The selected
native `glm_text_io_head` iterates all 154880 columns × 4096 BF16 weights on
egress only. That is 1,268,776,960 logical weight bytes per output row, with
5.652384 ms observed decode head time. `bind_text_io_weights` in
`integration/glm53_packed_expert_upload/text_io.mbt` binds the full table;
the installed model constructs `ParallelLMHead` at lines 934–939. The actual
trace confirms a BF16 GEMV shard (grid9680, block16x8, 164 registers/thread)
and subsequent all-gather. Passing quantization configuration alone would not
have proved that physical storage format; the executed instantiation does.

An ideal two-way split of the native head alone would save at most about
2.83 ms before communication and imbalance, about 3.8% of this decode span;
it cannot explain the entire completion gap. Unlike moving an expert sum
across a rounding boundary, splitting independent output columns need not
change their dot arithmetic. A generic physical output-domain partition and
stable `(score, global_token_id)` selection/join can express greedy execution
without gathering the full vocabulary. Other sampling contracts need their own
legal global combination. This is a proposed executable alternative, not a
measured speedup or an already implemented fix.

Installed shared-expert source sets the auxiliary dependency before routing but
submits shared computation after routed submission. Newer `../vlm` calls
`maybe_forward_async` before routing and joins via output events. Either may
overlap on the GPU; the new trace measures the installed version's overlap.
Do not credit it with a newer checkout's submission order.

Concrete source anchors, separate from observed execution:

- `../vlm/vllm/model_executor/layers/fused_moe/runner/moe_runner.py:848`
  enqueues the newer checkout's shared branch; `shared_experts.py:135` and
  `:156` describe input/output event ownership and the join.
- The installed snapshot `model_executor-layers-fused_moe-runner-moe_runner.py`
  has the different order at lines 583–617 and 853, and a local shared+routed
  sum before `_maybe_reduce_final_output` at lines 765–775.
- Installed `models-glm5next-nvidia-model.py:443` defers/fuses HC boundaries;
  `models-glm5next-nvidia-kda.py:202` merges six projection outputs, including
  two replicated TP shards. These are algorithms/schedules to evaluate, not
  justification to copy model-specific Python into the compiler.
- Native `integration/packed_expert_mlp/program.mbt:91` builds the coalesced
  F32 partial region; `integration/glm53_packed_expert_upload/tensor_boundary.mbt`
  specifies the F32 reduction effects. `compiler/parallel_ir/submission.mbt`
  owns the immutable ordered capture policy.

## Revised optimization order and evidence still missing

1. **Prefill BF16 shared/projection geometry and reduction staging:** distinguish
   useful CTA count from launched/early-exit CTAs; represent narrower output
   tiles and hierarchical max as physical-domain/reduction plans. Preserve
   accumulator and quantization laws. Measure complete shared chain, not only
   its fused gate leaf. This targets the demonstrated prefill losses without
   replacing faster decode routes.
2. **Dependency scheduling:** expose shared and routed pure branches with
   explicit disjoint scratch, completion tokens and a join. Lower the DAG to
   independent compute streams before capturing it. Replay alone cannot invent
   those dependencies or change an ordered plan into a parallel one.
3. **HC producer/consumer fusion:** fuse legal post/pre boundaries and retain
   numerical publication points; measure removed materialization/bytes, not
   just fewer API launches. Native's remaining cost is device work in replay.
4. **Vocabulary output-domain TP partition:** keep each column's reduction
   unchanged and join selection globally; account for gather/argmax and rank
   imbalance. This is independent from reducing expert communication payload.
5. **Collective placement:** the actual BF16/F32 path difference is now observed,
   but moving expert sums across rounding boundaries needs an explicit legal
   numerical alternative. Measure message counts/bytes and rank arrival as
   well as collective residency. Do not substitute the profile's perturbed
   prefill NCCL time for an unprofiled transfer budget.
6. **Expert ready-publication wait:** retain the smaller-row work advantage,
   isolate late producer issue versus transfer latency versus polling cost.
   The current component counters justify that experiment, not a claim that
   routed GEMMs are the largest observed full-model deficit.

These are schedule/layout/effect alternatives in the existing functional
multilayer IR, not a reason to move comparison into the compiler or add a
model-name tuning rule. No unproved policy was enabled during this tracing
phase. The tested vector transport improvement remains the earlier ~2% win.

Still missing: repeated **unprofiled** operator timing for these same complete
regions; measured read-miss/DRAM traffic; publication issue/completion latency;
full-model selected SASS counters rather than the geometry-matched component;
both reference ranks with calibrated clocks; long-context/concurrent traces.
Therefore the new exact short-decode accounting does **not** claim a complete
causal decomposition of the six-prompt ~20% production completion gap.

## Memory limits, negative attempts, restoration and validation

The new compact graph-level vLLM serving trace
`/tmp/lunaflux-reference-bench-vllm-glm-graph-trace-20261017-v5` loaded weights,
finished graph capture, then crossed the host reserve before serving became
ready: MemAvailable **16,716,632 KiB = 15.94 GiB**. The guard stopped both ranks.
Exit 137 after the requested stop is not proof of an OOM. No valid request
trace was collected. The previous node/eager attempts also remain retained.
We did not relax the 16 GiB reserve or repeat an unchanged failed attempt.
The v6 capture changed only instrumentation to egress-only and succeeded.
Benchmark containers stayed capped at 96 GiB with no swap; sampled minimum
MemAvailable was 18,422,284 / 17,772,120 KiB on .178/.179, both above 16 GiB.
The profiler's unsupported unified-memory tracing warning remains recorded;
no DRAM/unified-memory traffic claim is inferred from absent counters.

Both benchmark ranks exited successfully and were removed. Original
`glm53-exl3-worker` and `glm53-exl3-head` were restarted unchanged; the NFS
container was never stopped. Restore root
`/tmp/lunaflux-glm-egress-trace-restore-20261017-v1` records HTTP `/health` 200
and `/v1/models` returning `GLM-5.3-Flash-EXL3`, max_model_len850000.
This restored service has its original larger memory footprint (host available
about 9.0/11.1 GiB at the final snapshot); it is **not** inside the benchmark
envelope. No new test workload was launched alongside it.

The earlier source change passed 5187/5187 native tests and its targeted
correctness/sanitizer gates. The subsequent tracing/analysis additions are
external diagnostics only: selected/native analyzer1/1, reference analyzer1/1,
overlap1/1, replay ablation1/1, reference runner8/8, model runner4/4, stalls
runner1/1 and matrix-counter parser1/1. No new full-suite claim is made for an
unchanged production implementation; unrelated toolchain/TLS work remains
unstaged. No payload hashes were calculated.
