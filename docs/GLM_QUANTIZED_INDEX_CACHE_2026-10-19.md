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
