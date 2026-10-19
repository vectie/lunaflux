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
