# GLM remaining execution gap

Historical pre-integration diagnosis. The subsequent selected shape-aware
implementation, bounded validation and repeated full-model results are in
[the shape-compiler report](GLM_SHAPE_COMPILER_2026-10-14.md). The old selected
serial kernels and timing figures below are not the current profile.

The measured six-prompt completion ratio remains 2.760–2.762x vLLM after
queued TP and graph replay. The remaining implementation is not uniformly
parallel: selected matrix kernels coexist with serial reference lowerings,
fixed small-row matrix schedules, duplicated projections, and extra collective
boundaries. More graph replay alone cannot remove their device work.

This is an external diagnosis, not a new compiler policy or performance fix.
Use the literal prompt/token vector and timing qualifications in
[the submission report](TP_SUBMISSION_AND_GRAPH_REPLAY_2026-10-13.md).
No payload hashing or integrity scans are part of this investigation.

## Selected native trace

Retained selected binary `aed93d05`, TP2/PP1, graph compute, queued collectives,
capacity 64. Source and trace are downloaded under
`/tmp/lunaflux-glm-gap-diagnosis-20261014-v1`. The source is the actual rank-1
AOT, not a newly generated prototype. Input is the retained 16-token prompt,
two output tokens, preceded by the same warmup. Steps 3/4 below are measured
prefill/decode; the trace includes the warmup as steps 1/2.

The external analyzer `scripts/analyze-glm-selected-trace.mbtx` ends each step
at the next embedding or final kernel. The previous submission analyzer ended
egress at the vocabulary head, omitting greedy sampling. Its 182.985 ms final
decode compute sum therefore omitted 10.518 ms; the complete sum is 193.504 ms.
This corrects trace accounting, not the previously measured completion times.

| Selected family | Prefill GPU sum ms | Decode GPU sum ms |
| --- | ---: | ---: |
| Routed experts, including quantization/grouping | 137.183 | 15.799 |
| KDA projections, convolution, recurrence, controls | 55.818 | 51.904 |
| Dense FFN | 40.881 | 5.265 |
| Shared experts | 34.369 | 32.256 |
| Router projection, score and selection | 32.436 | 31.233 |
| Sparse-attention envelope and projections | 25.396 | 22.446 |
| Embedding, final normalization, vocabulary head, greedy | 21.796 | 21.763 |
| Other hyper-connection envelopes | 17.333 | 12.730 |
| MoE final combination | .136 | .108 |
| NCCL, including peer wait | 23.577 | 23.208 |

Final decode kernel span is 266.696 ms, kernel-duration sum 216.711 ms,
including 23.208 ms NCCL. Positive kernel-free intervals sum 49.998 ms. Small
overlaps mean these are not an exact additive critical-path decomposition;
copies and host/peer waits require separate attribution. Profiled durations
are not unprofiled latency or recoverable speedup, and rank times must not be
added together.

## Concrete selected code gaps

### Fixed matrix geometry wastes small-row work

`compiler/grouped_matrix_ir/plan.mbt` fixes all matrix tiles to 16 rows,
64 output columns and 32 reduction columns. It carries capacity rather than
a family of live-row schedules. Private lowering
`kernels/tiled_matrix_cuda_source/source.mbt` stages operands synchronously,
with two CTA barriers per 32 K values, then publishes through shared memory.
For K=4096 that is 128 iterations and 256 loop barriers. It is not a
multistage asynchronous pipeline.

Selected KDA entries `_3`, `_5`, `_8` launch 8/4/8 CTAs for capacity 64, but
only 2/1/2 CTAs perform useful work for one live token. Each still computes
16 rows with 15 padded. Their aggregate decode times across 34 layers are
5.893/3.717/5.717 ms, total 15.327 ms. All nine KDA projections total
50.543 ms; the actual recurrent update totals only .689 ms. Optimizing the
recurrence again would target the wrong part of this short-prompt workload.

The installed reference `vllm/models/glm5next/nvidia/kda.py` merges
Q/K/V/beta/f-a/g-a into one column-parallel projection. It performs the two
small-input projections and output projection separately. This is not an
assumption based on a generic Kimi implementation: the GLM-specific installed
class was captured and inspected. Its decode also keeps QKV merged through
convolution and folds beta's sigmoid into the recurrent kernel.

### Scalar lowerings remain selected

* Router: one thread per expert logit, serial 4096-term dot; only two CTAs
  contain active threads at decode. Top-k is one active thread scanning 288
  experts repeatedly. All 42 routers cost 31.233 ms. Source is the actual
  `glm_router_*_project/choose` in the captured AOT, not obsolete code.
* Vocabulary: one thread per vocabulary output with a serial 4096-term dot.
  The entire 154880-logit argmax runs in **one thread**. Head and greedy cost
  11.022 and 10.518 ms. The installed reference uses `ParallelLMHead` and a
  parallel `argmax`; our egress owns the complete head.
* Dense FFN: first three layers still use ordered scalar gate/up/down folds.
  `integration/glm53_packed_expert_upload/dense_block.mbt` forwards the matrix
  law into the attention branch, but `DenseSwiGluPrecision` has no matrix-law
  alternative. Thus the 40.881 ms prefill cost is a missing legal executable
  alternative, not evidence that an existing optimized dense route was chosen.

These three families total 47.015 ms of decode compute, before the vocabulary
head itself. Graph capture preserves their implementation; it cannot make a
serial reduction parallel. Parallel sampling infrastructure exists elsewhere
in the repository but is not consumed by this numerical text owner.

### Shared and routed experts need separate treatment

Shared experts cost twice the routed experts in this decode trace. Their BF16
path uses the generic grouped 16-row matrix path for a single expert. The
selected gate/up body repeats the input-staging traversal for its two outputs;
one valid row still pads to 16. It is not the native low-bit routed kernel
fixed in the previous round.

The retained routed-FP4 counter experiment remains relevant only to that
specific selected leaf and workload: gate/down tensor activity was 2.06/1.80%,
long-scoreboard cycles per issued instruction 61.40/55.51, despite the operand
supply improvement. It used a one-row synthetic eight-expert probe, not a
matched vLLM leaf. Do not present those counts as a fresh cross-framework
counter comparison or extend its ~2% active-grid result to BF16 shared experts.

### Sparse attention is mostly projection work here

Of the 22.446 ms sparse family, output projection alone costs 5.289 ms, and
the eight other matrix projections total 12.460 ms. The actual retained-score
attention kernels total .815 ms; index and pool kernels add .732/.951 ms.
This is a short-history finding, not a claim about long-context attention.

### Communication boundaries still differ

Native explicitly emits attention, routed-expert and shared-expert reductions,
132 sites total. `tensor_boundary.mbt` exposes separate routed/shared sites.
The installed reference constructs shared experts with `reduce_results=False`
and its MoE runner can combine shared+routed partials before one final reduce.
Changing this requires an explicit publication/numerical law: summation and
BF16 rounding placement must not be silently reassociated.

Native reduction sizes already use **live tokens**, not capacity-64 payloads
(`device/tensor_collective_execute.mbt`). Capacity-wide network transfer is
therefore not the demonstrated cause. Whether a particular reference branch
actually reduces once is resolved by its executed trace, not factory presence.

## Compiler implications

There are two relevant planning paths, not one fully reused implementation.
`kernels/luna_projection_strategy` already represents `DecodeGemv`,
`SmallRowTiledGemm`, `LargeRowTiledGemm`, row buckets and resource/measured
selection. The selected GLM numerical path instead reaches
`compiler/grouped_matrix_ir` and `kernels/tiled_matrix_cuda_source`, whose
package imports and fixed constructor bypass that strategy. The same gap
exists between segmented greedy infrastructure and the selected text owner.
Presence of the mature planner is therefore not evidence of GLM coverage.

Keep the existing functional layers. The missing pieces are coverage and
composition, not more IR names:

1. Give dense FFN, router and text-head operators explicit legal parallel
   alternatives; retain their precise rounding, tie and nonfinite contracts.
2. Select vector/GEMV, narrow-column split-K, and matrix schedules from live
   shape buckets instead of always selecting 16x64x32. Reuse the existing
   projection planner rather than a second independent fixed-tile selector.
3. Represent compatible multi-output projections as one pure region with
   shared input ownership; lower the merged KDA projection and paired shared
   gate/up through the same reusable machinery.
4. Make reduction/publication regions capable of combining compatible local
   partials before a collective. Capture supported communication regions only
   after this topology is explicit, preserving cancellation and buffer leases.
5. Validate selected-source coverage per operator, then measure each change
   separately. Keep reference comparisons, profiler data and performance
   selection experiments outside semantic correctness claims.

No production source is changed by this investigation.

## Fresh reference capture and limits

One new API-scoped Nsight capture was attempted at
`/tmp/lunaflux-reference-bench-vllm-glm-gap-20261014-v1`, using the cached
installed vLLM image and the exact 16-token/two-output diagnostic request.
Each container had a 96 GiB memory ceiling with its swap ceiling equal to
memory, and the paired external monitor retained the 16 GiB host reserve.
Both model loads reported 88.47 GiB. During startup around graph capture,
rank 0 available memory reached **15.8969 GiB**, triggering the paired stop.
Both containers retired with exit 137 after the requested stop; this is not
evidence of an OOM kill. No HTTP benchmark request or API-scoped capture ran.
Readiness, launch configuration, memory observations and both journals are
retained. The initial configuration file contains the ordinary max-output
label 8 and a profile-output override 2; the literal request has max_tokens=2.
The driver now writes the literal profile maximum directly.

Consequently there is **no fresh per-operation vLLM timing/counter table** from
this attempt. The six-prompt 2.76x result remains the retained unprofiled
comparison; the table here measures actual native work and source gaps, not
an additive attribution of that exact cross-framework difference. In
particular, the reference's executed collective count and matched leaf timing
still need a successful bounded capture. Do not turn the plausible 42-site
reduction opportunity into a measured speedup.

Both Spark GPU-owner lists are empty after cleanup; available host memory is
again above 117 GiB. No reserve was lowered, no second GPU job overlapped, and
no payload hashes were computed. Seven focused external-script tests pass
(six reference-driver tests and one native-trace analyzer test). Only
diagnostic scripts and these reports changed; unrelated dirty migration files
are untouched. The bounded investigation stops after this preserved reference
capture failure rather than consuming more model reloads without changing the
measurement's memory footprint.
