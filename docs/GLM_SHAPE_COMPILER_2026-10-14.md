# GLM shape-aware execution integration

The selected GLM path must consume the existing projection strategy, not infer
one fixed matrix schedule from its maximum row capacity. The numerical program
continues to precede scheduling: ordered scalar, matrix, and explicitly permitted
shape-dependent matrix/partitioned reductions are distinct programs. Timing or
agreement with another framework does not authorize reassociation.

Implementation order:

1. Connect reusable projection lowering to `luna_projection_strategy` and the
   existing scalar fold plan. Derive decode work from live rows; retain matrix
   prefill and exact output/publication boundaries.
2. Propagate that explicit arithmetic choice through recurrent/sparse projections,
   dense FFN, shared experts and the vocabulary head. Parallelize deterministic
   selection without changing tie, invalid-value or selected-weight semantics.
3. Check source/launch propagation, targeted native tests and bounded GPU numerical,
   sanitizer and timing fixtures, then run the identical full-model prompt vector.

Bounded measurement budget: one old/new component matrix, at most two revised
schedules per changed family if a failure or regression is diagnosed, two native
full-model repetitions and one unprofiled vLLM cohort. A failed memory guard is a
stop, not permission to reduce the reserve. Both Sparks remain at or below
100 GiB per task with zero task swap and at least 16 GiB available host memory.
No payload hashing or reference comparison enters the compiler or runtime.

Implementation now propagates an explicit `ShapeAwareFloat32` alternative through
GLM KDA/sparse projections, dense FFN, BF16 shared experts and text head. Decode
uses the existing projection classifier and immutable scalar-fold plan; multiple
rows retain tiled matrix execution. Shared prefill stages gate/up input together.
Router projection uses the existing independent grouped-dot law; group selection
and greedy selection parallelize stable maxima while preserving ordered sums,
ties and nonfinite behavior. The model does not own warp sizes or CUDA source.
The existing low-bit routed-expert program and ordinary defaults are unchanged.

This is not yet reuse of every mature prefill pipeline or autotune record. The
existing simple matrix prefill remains; the integration must not be described as
complete multistage/autotune coverage.

Actual generated-source GPU fixtures completed at
`/tmp/lunaflux-glm-shape-probe-20261014-v2`. Projection dyadic mathematical oracles,
dense FFN boundaries, head, selection, BF16/NVFP4 expert fixtures, invalid counts,
tails, ties and nonfinite selection controls passed. Both harnesses pass
memcheck/full leak, racecheck and synccheck (zero reports), each limited to 2 GiB
with zero task swap. The retained v1 failed only to compile the grouped harness's
stale unqualified fixture launch names; the component harness had already passed.
These bounded fixtures are not full-model numerical quality or speed evidence.

Full-model benchmarking is still pending. Comparison stays outside the compiler;
neither agreement nor speed permits an undeclared numerical rewrite.

Implementation `0633d85a` is pushed. The warning-denied native check passes with
the repository's existing migration exclusions, and the full native suite passes
**5,150/5,150**. Targeted `moon info` reports 1,084 existing migration warnings and
zero errors. A clean commit-pinned ARM build completed at
`/tmp/lunaflux-checkpoint-build-glm-shape-20261014-v1`; both actual AOT stages at
`/tmp/lunaflux-glm-expert-tp-shape-aware-20261014-v1` compile for sm121a. Export
selects the new numerical alternative with 98,252,360,500/99,521,773,364 planned
bytes, below the independent 100 GiB ceiling. GPU benchmark admission additionally
retains the 16 GiB host reserve; the plan's 2 GiB device overhead reserve is not a
replacement for that host floor.

The installed vLLM implementation remains an external implementation reference:
`reference-glm-kda.py` in the retained diagnosis root merges Q/K/V/beta/f_a/g_a
into one projection; `reference-model.py` uses merged gate/up and a parallel
language head. This change removes selected serial/padded work but does not yet
merge those six KDA operators or replace every matrix prefill schedule. No
complete performance-parity claim follows from connecting the shape classifier.
