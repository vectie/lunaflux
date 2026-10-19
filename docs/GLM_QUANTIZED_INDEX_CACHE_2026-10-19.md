# Quantized index and compressed-cache execution

This iteration starts from serving revision `98c629fd`: 7,973 input tokens and
one output take 41.679 seconds, versus the retained matched vLLM 17.642 seconds.
These observations are not an attribution of the entire gap to attention.

## Contracts and sequence

1. Introduce an explicit numerical program for row-scaled E4M3 index operands,
   F32 matrix accumulation, ordered head weighting and stable selection.
   Quantization is a different finite-precision program, not an exact rewrite.
2. Keep candidate tile ownership and score-publication effects in physical and
   parallel IR. Keep CUDA fragment ABI in private lowering. Persistent pooled
   keys retain page-generation ownership; quantized payload and scale publish
   together before selection.
3. Evaluate compressed-cache representation independently of index scoring.
   Representation, decode semantics, byte accounting and lifecycle must reach
   producers AND consumers. A compressed allocation with an expanded consumer
   is not an implementation. Lossy storage and latent MLA absorption are distinct
   numerical alternatives; do not silently conflate them.
4. Prove actual selected source and launch geometry, then run bounded GPU
   correctness, determinism, sanitizer/leak and alternating paired timing gates.
   Compare complete answers and fixed-output HTTP latency externally; neither
   vLLM agreement nor benchmark timing authorizes a compiler rewrite.

Budget: at most three candidate revisions per path before reprofiling or
reporting a blocked/non-winning result. Retain all failed trials. Use the
same prompt/token/concurrency vectors as the preceding report. No payload
hashes or integrity scans. GPU jobs are serialized; every Spark stays below
100 GiB, with at least 16 GiB host MemAvailable and no owned swap.

The existing GLM pair remains online during CPU implementation. Temporarily
stop it through its paired lifecycle controller for GPU qualification and
restore the validated `98c629fd` release after experiments. Do not replace
the public release with an unqualified numerical alternative.

## Reference implementation observations

The installed vLLM indexer (retained at
`/tmp/lunaflux-glm-long-indexer-vllm-20261019.py`) separates quantized matrix
logits from top-k. It stores one FP8 vector plus F32 scale per pool, applies
Hadamard preprocessing, folds query scale into head weights, and bypasses
scoring when all candidates are selected. Its selected MLA cache is compressed;
native currently retains expanded BF16 K/V. These differences are hypotheses
to test, not additive measured wall-time fractions or a correctness oracle.

## First index candidate (not production-qualified)

Actual-source probe `/tmp/lunaflux-long-paged-sparse-20261019-v18` compares the
retained serving implementation with the explicit E4M3 alternative, at 64 live
queries. Median complete index-chain GPU time is approximately 5.378 -> 1.007 ms
at 8K history, and 1.377 -> 0.170 ms at 2K. Attention itself is unchanged; do not
interpret this as a measured HTTP gain. The independent software E4M3/F64-score
oracle passes all nine history/live-row fixtures (two query rows per fixture).
Selected indices differ: at 8K the observed attention-output relative L2 delta
is 0.4531%. This is not a model-quality pass.

The initial 2 GiB racecheck exceeded its memory cap without swap. A separately
recorded 8 GiB run passes memcheck/full leak, racecheck and synccheck. Initcheck
then exposed an offline observer bug: the probe copied uninitialized unused
pooled-cache capacity to the host. It is corrected to read only complete valid
pool rows; v19 repeats the gates without suppressing the finding. The serving
kernel's cache validity contract is unchanged. GLM is temporarily stopped while
these exclusive GPU tests run; the validated release remains the restore target.

The next observer correction restricts score readback to each live row's
published candidates as well. v20 passes all four sanitizer tools, including
full leak checking, with zero swap. These were observer overreads of unused
capacity, not suppressed kernel findings. The index candidate is committed as
`35fd22a2`, with a pinned ARM build and both-rank AOT export/compilation.

### Selected counters (8K fixture, single profiled invocation)

| Metric | BF16 scalar score producer | E4M3 matrix score producer |
| --- | ---: | ---: |
| Profiled time | 4.7915 ms | 0.4730 ms |
| Warp instructions | 108,843,292 | 47,599,147 |
| LSU instructions | 33,183,313 | 7,700,688 |
| L2 read sectors | 34,084,865 | 1,445,851 |
| Eligible warps/active cycle | 0.163 | 0.439 |

Capture: v20 `counters-next.ncu-rep`, summarized in `selected-counters.txt`.
NCU replay timings are not interchangeable with alternating unprofiled medians.
The unchanged selection kernel executes 99.51M instructions and takes about
0.78 ms in this capture; it now dominates the index chain. Expanded attention
still executes 259.29M instructions, reads 37.42M L2 sectors and reports 57.67%
long-scoreboard warp stalls. Improving score generation does not fix that path.

## Latent cache program and ownership

`SparseCacheLaw::LatentBFloat16` explicitly selects
`Q -> Q Wk -> attention(normalized latent) -> context Wv -> output projection`.
Q absorption and value expansion use generic strided head-matrix plans. The
softmax scale remains tied to the original Q/K dimension, not the latent rank.
KV normalization is retained; expanded KV-B/split launches are omitted.
Persistent K and V ports borrow **one allocation and one deterministic release**.
Projection-frame aliases likewise have an explicit immutable owner map. No
weight repacking, vendor runtime, reference substitution or request-path JIT is
introduced. At 32 local heads, dimension 256 and latent rank 512, the retained
attention payload is 32 times smaller (index/gates/metadata are separate).

This is BF16 latent storage, not FP8 latent storage, and is not bitwise-equivalent
to expanded attention. It is independently selectable from E4M3 index scoring.
Actual-source probe `/tmp/lunaflux-glm-latent-probe-20261019-v2` exercises head
grouping, query absorption, shared latent attention, value expansion and inactive
row publication at 1/15/16/17 live rows. Independent F64 bounds and repeated
deterministic output pass; memcheck/full leak, racecheck, synccheck and initcheck
all pass at 2 GiB/no swap. v1's offline harness compile failure is retained.
Full-model answers and HTTP performance remain separate acceptance gates.

The third actual-source fixture adds zero live rows and a column-contiguous
producer for transposed head weights. CUDA stages these into a padded
column-major shared operand; this changes physical layout, not the declared
matrix arithmetic. v3 passes deterministic/F64-bound checks at 0/1/15/16/17
live rows and all four sanitizer tools, including full leak checking, at
2 GiB/no swap. The updated generic loader and external broad-answer evaluator
are committed in `cbffec1f`; the full native suite passes **5,319/5,319** and
the warning-denied native check passes with the unchanged migration exclusions.

## Index-only full-model measurements

Pinned `35fd22a2` service: E4M3 matrix index, expanded BF16 attention cache,
TP2, context 8192, 8 resident slots, 16 admission lanes, 64 physical prefill
rows and 513 history pages. Source export and execution both identify the
selected index program. This is a private experiment, not a production rollout.

| Actual input tokens / one output | Retained serving baseline (s) | Index-only steady repeat (s) | Retained matched vLLM (s) |
| --- | ---: | ---: | ---: |
| 165 / 1 | 1.234 | 1.278 | 0.808 |
| 2085 / 1 | 11.097 | 11.017 | 6.057 |
| 6181 / 1 | 31.770 | 29.795 | 15.018 |
| 7973 / 1 | 41.679 | 38.306 | 17.642 |

The near-8K gain is **8.1%** versus the retained serving baseline, not the
score kernel's roughly tenfold profiled improvement. Completion still takes
**2.17x** the retained vLLM time. Short-context timings do not demonstrate a
gain. These are individual observations, not confidence intervals.

The 128-output-budget retrieval campaign completes **64/64 exact answers**,
all with stop termination, over short/2K concurrency 1/4/8/16 and longer C1/C4
cases. This establishes those task answers only. Variable output lengths in
the answer campaign are not equal-work throughput. The fixed-one-token table
uses literal serialized-prompt and input/output-token checks instead.

Evidence roots:

- `/tmp/lunaflux-glm-service-benchmark-matrix-index-latency-20261019-v1`
  (first sweep retained) and `...-v2` (steady repeat above).
- `/tmp/lunaflux-glm-service-benchmark-matrix-index-answers-20261019-v1`.
- `/tmp/lunaflux-glm-matrix-index-latency-compare-20261019-v2.json` and
  `/tmp/lunaflux-glm-matrix-index-answers-compare-20261019-v1.json`.
- `/tmp/lunaflux-glm-http-matrix-index-20261019-v1/memory-watch.txt`:
  device peaks 96,877/96,905 MiB; minimum host available
  18,301,652/20,100,156 KiB; zero owned swap. Both workers stopped cleanly.

The subsequent latent-cache full-model run and refreshed vLLM comparison are
recorded below. They are separate from the small-shape kernel fixture.

### Paged cache qualification and refreshed reference

`/tmp/lunaflux-paged-sparse-latent-20261019-v1` exports the actual shared-cache
history writer. Its five mixed-request steps, noncontiguous physical pages,
recycled generations and same-generation overwrite match independent dense
per-request histories exactly. K/V ports alias one physical output allocation
in the GPU test. All four sanitizers and full leak checking pass at 2 GiB/no
swap. This extends the earlier attention-math fixture to the changed paged
write path; it is not a claim about model-answer quality.

The first full AOT compilation attempt used a 2 GiB CPU cap while vLLM loaded.
Both compilers reached that cap and were explicitly stopped to avoid sustained
memory pressure. Their v1 logs are retained; this is not a successful compile
or a measured kernel failure. The identical pinned source is being rebuilt
in a new v2 root at 4 GiB/no swap after the reference pair retired. Both v2
compiles completed; observed peaks were 2,589,188,096 / 3,832,655,872 bytes.

Fresh reference root:
`/tmp/lunaflux-reference-bench-glm-latent-cache-20261019-v1`.
The same installed vLLM build, NVFP4 checkpoint, TP2, context8192, sequence
limit8, 64-row chunking and 256 MiB KV budget are retained; prefix reuse and
speculation are disabled. Steady fixed-one-output measurements at
165/2085/6181/7973 input tokens are **0.788/6.047/14.349/18.033 seconds**.
Both sweeps are preserved under
`/tmp/lunaflux-glm-service-benchmark-vllm-latent-cache-latency-20261019-v{1,2}`.

The separate 15-prompt external corpus passes all 11 automatic task checks.
Manual inspection confirms the translation, Python function, SQL semantics
and bounded summary; SQL has a Markdown-fence formatting violation against
the literal instruction. Root:
`/tmp/lunaflux-glm-broad-vllm-latent-cache-20261019-v1`.
Framework agreement is not the acceptance criterion; the prompt's task is.

Reference device peaks are 94,544 MiB on both ranks; host available minima
are 17,626,708/18,521,100 KiB. Both GPU owners are released. The reference
reports 10,240 cache tokens, whereas native513 physical pages provide8,208
history rows. C1 requests fit both; concurrent results must retain this
capacity distinction instead of attributing all waiting to kernel speed.

## Latent-cache full-model result: not a performance winner

Pinned `cbffec1f`, root `/tmp/lunaflux-glm-http-latent-cache-20261019-v2`,
selects both matrix index scoring and BF16 latent sparse cache in export and
execution. All 64 retrieval requests finish normally with exact answers. The
15 broader prompts pass the 11 automated checks; manual inspection finds
correct translation, Python, SQL semantics and summary, but the SQL response
adds Markdown fences despite the formatting instruction. The vLLM broad run
has the same formatting issue; agreement is not the acceptance criterion.

| Actual input / one output | Index only, s | Latent cache repeat, s | Fresh vLLM repeat, s |
| --- | ---: | ---: | ---: |
| 165 / 1 | 1.278 | 1.226 | 0.788 |
| 2085 / 1 | 11.017 | 11.300 | 6.047 |
| 6181 / 1 | 29.795 | 31.964 | 14.349 |
| 7973 / 1 | 38.306 | 42.515 | 18.033 |

The near-8K latent path is **11.0% slower than index-only** and **2.358x the
fresh vLLM completion time**. The first latent sweep was
1.310/11.039/31.752/42.353 s and is retained. Reduced cache storage did not
demonstrate reduced total execution time. Do not promote this experimental
cache based on its correctness fixtures or memory savings.

Evidence: `/tmp/lunaflux-glm-service-benchmark-latent-cache-latency-20261019-v{1,2}`,
`/tmp/lunaflux-glm-latent-cache-latency-compare-20261019-v2.json`,
`/tmp/lunaflux-glm-service-benchmark-latent-cache-answers-20261019-v1`, and
`/tmp/lunaflux-glm-broad-latent-cache-20261019-v1`.
The service memory watcher records GPU peaks94,144/94,172MiB and host minima
21,076,060/22,847,440KiB, no owned swap. Both private workers retired cleanly.

The user requested deeper reference tracing before another kernel change.
The next phase therefore freezes implementation and traces the installed
selected vLLM build and exact native artifacts at the real long workload.
The old reference profiler silently substituted a short-context envelope;
explicit context/KV/sequence/chunk settings now apply to profiling as well.
Graph-level collection omits graph interiors and cannot by itself prove
GPU-busy time or operator attribution. Full-node profiling previously crossed
the host reserve; those failures remain visible and are not successful traces.

## Detailed selected vLLM trace, 19 October evening

This phase changes diagnostics only, not production kernels or selection. Its
budget is zero kernel candidates. It answers the user's request to inspect
what the reference actually executes before another optimization. No payload
hashing is performed. Source snapshots, trace SQL, comparisons and correctness
fixtures remain outside the compiler.

### Matched workload and evidence limits

Both engines use the same serialized 7,973-input/one-output prompt, C1, TP2 on
the two GB10 Sparks, 64-row prefill chunks and an 8,192-token context limit.
The one-token budget isolates prefill; it is **not** a complete-answer test.
Native `cbffec1f` is the experimental index+latent-cache HTTP artifact, not the
validated production `98c629fd`. The installed reference is
`v0.1.dev20051+g487ecf187`, from the existing Spark image. Local `../vlm` is
useful context but does not contain this exact installed GLM implementation.
Reference capacity is10,240 KV tokens versus native8,208; both fit this C1
request, but this is not a matched concurrency-capacity comparison. Their
attention cache arithmetic also differs: reference FP8, native BF16.

Reference root:
`/tmp/lunaflux-reference-bench-glm-long-egress-trace-20261019-v2`.
The refreshed client completes in17.703s under capture, versus18.646s in the
preceding uncaptured warmup and18.033s in the previous steady repeat. Rank1
has125 observed passes:124x64 rows and1x37 rows,333,954 kernel records. Long
prefill is eager here: zero opaque CUDA graph envelopes. Full kernel symbols,
launch geometry, API chronology, NVTX and GPU gaps are retained in
`/tmp/lunaflux-glm-vllm-long-trace-analysis-20261019-v1`; operation attribution
and per-pass/per-layer regions are in
`/tmp/lunaflux-glm-vllm-long-pass-analysis-20261019-v4`.

Nsight warns that not all CUDA/NVTX events might have been collected, including
for worker PID356; this warning must not be hidden. Other no-CUDA/no-NVTX
warnings belong to launcher PID79 and helper PID353. The actual GPU worker
has all125 expected passes,11,375 AllReduce and125 AllGather calls,5,250 each
routed GEMM role, and1,375 each sparse main/merge kernels. These checks support
internally consistent pass accounting, not a formal guarantee of lossless
instrumentation. Unified-memory fault tracing is unsupported in this driver.
Only rank1 is instrumented; absolute clocks across hosts are not calibrated.

### Whole-request vLLM GPU accounting

| Selected operation | Kernel sum, ms | Calls |
| --- | ---: | ---: |
| Routed gate/up CUTLASS | 4450.641 | 5250 |
| Routed down CUTLASS | 2435.784 | 5250 |
| KDA projections | 3265.719 | 17000 |
| Sparse projections/absorption/output | 1124.212 | 16500 |
| Shared experts | 900.777 | 15750 |
| KDA remaining work | 551.694 | 86375 |
| Sparse attention plus split merge | 568.207 | 2750 |
| HC control/normalization/publication | 464.227 | 33750 |
| Router | 474.526 | 42000 |
| Routed preparation/activation/finalization | 429.587 | 36750 |
| Dense FFN | 285.835 | 1125 |
| Index scores plus selection | 22.690 | 2046 |
| Other sparse preparation | 127.228 | 50523 |
| Embedding/head/sampling and boundary remainder | 448.220 | 7385 |
| NCCL | 1955.750 | 11500 |

These sums overlap. The exact captured interval identity is:

```
17,505.095 ms summed kernels
   -552.971 ms multiple-kernel overlap
    +22.261 ms copies outside kernel union
   +604.933 ms uncovered GPU intervals
 =17,579.319 ms GPU span
```

Compute union14,996.375ms; collective union1,955.750ms. No measured
compute/collective overlap on this rank. Shared-expert union899.779ms overlaps
other kernels410.745ms, leaving489.033ms exposed. The largest GPU gap is0.850ms:
many small gaps total0.605s, not a hidden multi-second idle hole. CPU API sums
are not additive critical-path time: e.g.6.454s of `cudaMemcpyAsync` CPU calls
can include device-dependent waiting while GPU work is active.

### Source-to-execution differences, not unselected feature lists

| Boundary | Actual selected vLLM | Actual selected native HTTP artifact |
| --- | --- | --- |
| Expert GEMM | SM120 block-scaled TMA collective;128x128x256 tile; grid1x48x1,384threads,168registers,89,088B dynamic shared | `PackedRows`;512CTAs,128threads;16x32 output tiles;16B `cp.async`, immediate wait and CTA barriers; separate gate/up operand traversals |
| Activation scale | CUTLASS NVFP4 uses loaded/calibrated global scale metadata and parallel conversion/activation path | Fresh full-input and full-routed-activation maximum scans, **one256-thread CTA each** |
| Recurrent work | Selected parallel chunk-KDA chain | HTTP plan still selects `OrderedTokenFloat32`; actual `_delta` entry |
| Projection staging | Selected vendor matrix kernels | HTTP plan sets both matrix staging fields `Synchronous` |
| HC | At64rows, separate pre/norm/post chain;33,750 launches | Deferred post/pre plus tail functions; two main entries per boundary, not the old serial HC implementation |
| Index | Cached completed pooled keys; skips scoring while all context fits topK; FP8 score matrix and parallel selection | FP8 matrix scoring is selected, but scoring entries still execute in early chunks |
| Sparse attention | Gather selected keys first, split-K, FP8 QK/PV, block softmax, bulk double-buffered copies | Full-history QK before masking; BF16 latent width512; global score/probability scratch; synchronous publication |
| Shared overlap | Auxiliary stream, observed partial overlap | Auxiliary stream already present; early shared work almost entirely hidden |
| Graphs | Actual64-row prefill eager, despite graphs enabled for small buckets | Constituent graphs used; service's full-step decode-hit counter does not describe them |

The native selection is explicit in pinned `cmd/glm_service/plan.mbt`:
`expert_transport=PackedRows`, `expert_scale_partitions=1`,
`recurrent_evaluation=OrderedTokenFloat32`, and synchronous staging. Exported
CUDA and measured launch resources agree. The bulk-tiled expert and distributed
maximum implementations exist, but **the service does not select them**. Their
earlier checkpoint-runner results cannot be called current HTTP results.
This is not proof that enabling all alternatives together is safe or faster;
earlier overlap/numerical investigations remain preserved.

Installed source snapshots:
`/tmp/lunaflux-glm-selected-vllm-source-20261019-v3`,
`/tmp/lunaflux-glm-selected-vllm-experts-20261019-v2`, and
`/tmp/lunaflux-glm-selected-flashinfer-source-20261019-v2`.
The selected expert is CUTLASS, **not** the also-installed fused B12x backend.
The expert snapshot v1's nonexistent `nvfp4.py` read failed; corrected v2 reads
the actual `modelopt.py`. No model or GPU workload is loaded by this source read.

Sparse dispatch is specifically `sparse_mla_decode_dsv3_2_kernel<2,32,2048,64>`:
64x2x32 nominal grid,288threads,96registers,98,608B shared. Its16 heads/CTA share
selected KV, with one IO warp and eight math warps, two64-key buffers and a
separate merge. Inactive splits return, so nominal CTAs are not active work.
Native one-head/CTA instead shares keys across16 queries: it is incorrect to
infer16x worse reuse just from head ownership. The demonstrated work-domain
difference is full-history QK versus selected-key QK (near8K/top2K, roughly4x
pairs before other costs), not a proved4x end-to-end penalty. Reference executes
index scoring on93 late passes x11layers=1023; the early32 passes bypass it.

### Native trace limitations and matched early operation windows

The node capture root
`/tmp/lunaflux-glm-http-latent-egress-trace-20261019-v2` exceeded host headroom
after collection began: .179 MemAvailable fell to12,991,620KiB. The paired
watcher stopped both workers. Device use remained94,144/94,181MiB and owned
swap was zero, but the host-floor breach is a **failed profiling attempt**.
The interrupted request is not a benchmark. Its earlier uncaptured warmup
completed in42.161s. The v1 attempt failed before GPU work because the trace
directory did not match the artifact manifest; launcher correction preserves
the original artifact root, with no recompilation or configuration changes.

The partial node report has13 embedding boundaries; the external analyzer
excludes the last pass and reports12 completed boundary intervals. Each has
42 gate/up and42 down calls. It does **not** establish complete collective
coverage: only149 NCCL records appear, fewer than the expected full per-pass
sequence. No complete native busy/idle or communication attribution is claimed.
`/tmp/lunaflux-glm-native-partial-long-analysis-20261019-v2` retains the exact
classified records and per-pass sums.

| Same first12 prefill chunks, selected kernel sums | Native ms | vLLM ms |
| --- | ---: | ---: |
| Routed preparation/activation/finalization | 574.332 | 43.027 |
| Routed gate/up | 1194.572 | 872.654 |
| Routed down | 562.369 | 474.231 |
| KDA projections | 458.602 | 313.308 |
| KDA remaining | 113.124 | 53.062 |
| Sparse projections | 251.729 | 107.894 |
| Sparse attention/merge | 96.710 | 20.773 |
| Dense FFN | 73.554 | 27.381 |

These are instrumented early-window observations, not a decomposition of the
entire25-second HTTP gap. Instrumentation modes differ and the native capture
was interrupted, so these ratios are not deployment speedup predictions.
History changes and routing work may vary during the request: reference
gate/up averages1.731ms in these first12 passes versus
0.848ms over all125. Extrapolating one sample across all chunks is invalid.
Native shared-expert union762.320ms overlaps757.400ms with other kernels in
this window; only4.921ms remains exposed. Removing all shared work would not
save its762ms sum. Concurrent resource competition still requires a causal
control; near-complete overlap does not prove free execution.

Actual13-pass native leaf totals identify the preparation cost: activation
global-scale scan427.030ms and input global-scale scan88.047ms, each546 calls
at grid1/block256. At64rows the former serially distributes524,288 BF16 values
over one CTA. The grouping/packing kernels are separately recorded, not
mistaken for GEMMs. The same trace records1,037,004 `cuCtxSetCurrent` calls
(44.148ms summed API time). This is a real polling/API-volume issue, but that
44ms does not demonstrate a multi-second latency cause.

The subsequent graph-level capture at
`/tmp/lunaflux-glm-http-latent-graph-trace-20261019-v2` was proactively stopped
at20GiB profiler headroom, before the16GiB service floor. The model request
continued normally and completed in42.673s; its uncaptured warmup was42.312s.
Only1.182s of graph activity was captured:940.379ms opaque graph envelopes,
169.100ms NCCL and72.465ms uncovered. Opaque envelopes are **not** individual
kernel work or proof of GPU busy time. Full native tracing is still incomplete;
do not repeat unbounded resident captures or mislabel this as a full trace.
CUDA13's flush interval zero already saves full buffers; changing the flush
interval was not demonstrated to solve memory overhead.

### Why latent storage regressed: exact-AOT counter control

`/tmp/lunaflux-glm-latent-chain-20261019-v2` directly loads the existing old
index-only35fd and new latentcbff cubins. A compiler-owner regression verifies
the launch grids. The one-hot weight bank makes both representations solve
the same problem; these are synthetic operands, **not captured GLM activations**.
Projection/attention stages are timed separately, with11 alternating trials
and the first2 discarded; indexing/history copy/rest-of-model are excluded.
All524,288 outputs match in all four cases. The8K/stride4 case passes full
memcheck/leak, racecheck, synccheck and initcheck with zero reported errors.
The process is capped at8GiB/no swap, admitted only with24GiB host headroom.

| History / selected-key stride | Expanded attention, ms | Latent attention, ms |
| --- | ---: | ---: |
| 2048 / 1 | 1.802 | 3.113 |
| 2048 / 4 | 1.792 | 3.009 |
| 8192 / 1 | 4.077 | 8.628 |
| 8192 / 4 | 7.626 | 18.835 |

At8K/stride4 the old projection+split is0.142ms; new query absorption+value
expansion is0.191ms. The dominant regression is inside attention, not merely
two extra projection launches. All timed component medians are retained; sums
of medians are not independently measured whole-chain medians.

| Actual attention NCU counters, 8K/stride4 | Expanded | Latent |
| --- | ---: | ---: |
| Warp instructions | 312,229,536 | 541,799,584 |
| L2 read sectors | 45,571,269 | 70,115,825 |
| Tensor-pipe warp instructions | 4,188,160 | 8,376,320 |
| Registers/thread | 56 | 64 |
| Allocated shared memory incl. overhead | 27,672 B | 35,864 B |
| Eligible warps/cycle active | 0.115545 | 0.085378 |
| Long-scoreboard stall contribution | 65.914% | 65.425% |
| Barrier stall contribution | 14.392% | 8.881% |

The attention dimensions double256->512, tensor instructions double, support
instructions and read traffic rise, and eligibility declines. Barrier
percentage falls while execution slows: another direct counterexample to
optimizing one stall percentage. Dispersed selections also force more16-key
PV tiles than contiguous selections. Clock/cache controls are disabled and
NCU records its warnings; use the unprofiled event times above for speed.
This explains the controlled latent regression, **not** all model slowdown,
nor a matched counter comparison against reference FP8 attention.

### Priorities supported by this evidence

1. Unify service/checkpoint **selection plumbing** through the existing pure
   plan: qualify distributed maximum and bulk expert alternatives at actual
  64-row HTTP/paged/concurrent shapes. Do not add another IR merely because
   two callers choose different policies. Prove selected symbols after binding.
2. For compressed attention, make selected-key domain, head/query reuse,
   precision, split reduction and producer/consumer ownership a coherent
   physical alternative. Storage compression alone is not an execution plan.
3. Keep full request accounting and component counter tests separate. Preserve
   routing distribution and early/middle/late context; compare exposed union
   time, not kernel sums. Native full collective/idle attribution remains open.
4. Benchmark each qualified change separately with unchanged public answer
   checks; reference agreement is not correctness. No promotion of latent
   regression or unsupported arithmetic just because vLLM selects it.

This tracing round demonstrates selection gaps and specific excess work. It
does not claim a speedup, a complete attribution of the remaining gap, or that
existing unselected alternatives will automatically match vLLM. Production
restoration uses validated98c629fd, never the experimental latent artifact.

Restoration completed: both persistent units are active on98c629fd; private
HTTP health is healthy and `17+25` returns `42`, finish reason`stop`,26 input /
3 output tokens. Observed GPU use96,881/96,909MiB; host available18,557,784 /
20,203,656KiB; both owned swap counters zero. This is a restoration smoke,
not a rerun of the full customer-facing quality campaign. No new artifact was
promoted. Diagnostic tests17/17, affected projection tests3/3 and native check
pass with unchanged existing warning exclusions. Targeted`moon info` completes
with569 existing migration warnings and zero errors. No full-suite rerun is
claimed for this diagnostics-only phase; unrelated worktree changes remain.
