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

After the repeated cohort, one bounded two-token selected-kernel timeline is
permitted to attribute the remaining gap. It is not another timing cohort or a
retry of the previous vLLM profiling attempt that reached the memory reserve.

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

The bounded full-model benchmark is complete. Comparison stays outside the
compiler; neither agreement nor speed permits an undeclared numerical rewrite.

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

## Matched full-model measurements

Both native repetitions selected the actual `0633d85a` ARM executable and both
shape-aware AOT stages above, TP2/PP1 with graph replay and queued collectives.
The frozen previous native run used the preceding matrix/graph path. The fresh
vLLM run used the cached GLM Spark image
`ghcr.nju.edu.cn/miaai-lab/glm-5.3-flash-2x-dgx-sparks:exl3-e3-20260907`, version
`0.1.dev20051+g487ecf187`, TP2/PP1, FP8 MLA KV, no prefix reuse or speculation,
temperature zero and eight maximum output tokens. Native KV is BF16. Literal
input IDs are checked directly; no payload hashes are computed.

One resident warmup request is excluded in each cohort. These are serial,
short-input **completion latencies including prefill**, not decode-only token
throughput. Native timing includes CLI diagnostic output/retirement; vLLM uses
buffered HTTP including client/server overhead. They are not identical timing
boundaries or numerical programs. Two native samples and one reference cohort
do not establish a confidence interval or long-context/continuous-batch behavior.

| Prompt | Input tokens | New / vLLM output tokens | Previous native s | New run 1 s | New run 2 s | Fresh vLLM s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Fact | 22 | 4 / 4 | 1.063 | .737 | .630 | .388772 |
| Arithmetic | 23 | 3 / 4 | 1.078 | .771 | .551 | .391046 |
| Chinese | 22 | 8 / 8 | 1.840 | 1.378 | .972 | .652115 |
| JSON | 23 | 8 / 8 | 1.870 | 1.110 | 1.001 | .668753 |
| Code | 24 | 8 / 8 | 1.869 | 1.016 | 1.024 | .687002 |
| Retrieval | 36 | 3 / 3 | 1.011 | .674 | .687 | .372406 |

All-six sums are 8.731 s previous native, 5.686/4.865 s new native and
3.160094 s vLLM. **Those sums contain unequal generated work:** the arithmetic
case shortened by one token. The five equal-output-count cases total
7.653 s previous, **4.915/4.314 s new**, and **2.769048 s vLLM**. That is
**35.8–43.6% less native time**, with **1.775/1.558x** remaining completion-time
ratios. Even the narrower three-case subset whose complete native token IDs
did not change falls from 3.943 s to 2.427/2.341 s (38.4–40.6% less time).

Native first-token vectors in seconds are
`[.354,.508,.497,.475,.386,.478]` and
`[.349,.366,.343,.378,.390,.487]`. Every request used one prefill chunk.
All-six output-token rates including prefill are 5.98/6.99 tok/s native versus
11.08 tok/s vLLM, but the native/reference output totals are 34/35. These rates
must not be described as decode-only performance or equal-token throughput.

Both new repetitions return identical native token IDs. Fact, code and retrieval
also retain their previous native IDs. Chinese now matches the displayed vLLM
text; arithmetic still answers 42 but omits the earlier repeated reasoning
token, and JSON emits a different truncated prefix. Four of six displayed
texts match vLLM after removing the native stop marker. Eight-token truncation
cannot assess full JSON/code answers. **This is not a quality/parity pass**, and
the new numerical alternative remains explicitly selected, not the ordinary
serving default. Agreement with vLLM is not the compiler's correctness rule.

Native minima across the two host-memory watchers were 22.818/22.779 GiB;
fresh vLLM minimum was 17.667 GiB. Both native ranks exited zero and explicitly
closed their CUDA contexts, with zero task swap. Both tasks were capped at
100 GiB; cgroup memory figures alone are not total unified GPU allocation.
The vLLM containers were capped at 96 GiB with no task swap. After all six HTTP
200 responses, the requested paired stop ended rank 0 normally; rank 1 reached
the five-second stop timeout and exited 137. The watcher recorded that terminal
failure **after the intentional stop**, not a measured reserve breach. Both
containers/GPU owners were released. The failure report is retained.

Raw roots:

* Previous: `/tmp/lunaflux-model-bench-glm-tp-graphs-20261013-v1`.
* New run 1: `/tmp/lunaflux-model-bench-glm-shape-20261014-v2`.
* New run 2: `/tmp/lunaflux-model-bench-glm-shape-repeat-20261014-v1`.
* Fresh reference: `/tmp/lunaflux-reference-bench-vllm-glm-shape-20261014-v1`.

Run 1's initial `comparison.json` inherited the old report's selection string
despite reading the new timing. This external-report metadata defect is fixed;
`comparison-corrected-selection.json` records the actual new `selected.txt` and
preserves the original report without overwrite. Run 2's comparison uses the
corrected writer. Both export/execution journals independently record selection.

## Selected component control

The actual KDA `glm_recurrent_25_0` projection, N=K=4096, capacity 64, was
extracted from the retained old and new AOT sources with their actual launch
geometry. At one live row the median GPU duration falls from .208822399 ms to
.136889607 ms (**34.45% less**, 1.526x speedup). At 16 rows it is
.215916798/.214643195 ms, effectively unchanged; matrix prefill was not replaced.
Dyadic mathematical checks have zero maximum absolute error. The bounded
2-GiB/no-swap root is `/tmp/lunaflux-glm-shape-leaf-20261014-v1`.
The diagnostic cuBLAS controls in that harness are not vLLM's selected kernels.

## Actual selected timeline and remaining work

One two-token, 16-input-token diagnostic after an identical warmup completed at
`/tmp/lunaflux-model-bench-glm-shape-trace-20261014-v1`. Both rank traces and
SQLite exports are retained locally. `scripts/analyze-glm-selected-trace.mbtx`
uses the full step through greedy sampling, not an endpoint at the head. The
table compares rank-1 measured steps 3/4 with the retained previous selected
trace in `/tmp/lunaflux-glm-gap-diagnosis-20261014-v1`. Do not add ranks together.

| Selected family | Previous prefill ms | New prefill ms | Previous decode ms | New decode ms |
| --- | ---: | ---: | ---: | ---: |
| Routed experts | 137.183 | 137.602 | 15.799 | 16.022 |
| KDA envelope | 55.818 | 55.597 | 51.904 | 24.913 |
| Dense FFN | 40.881 | 3.899 | 5.265 | 2.037 |
| Shared experts | 34.369 | 30.588 | 32.256 | 7.135 |
| Router | 32.436 | 2.883 | 31.233 | 1.780 |
| Sparse envelope | 25.396 | 25.022 | 22.446 | 12.462 |
| Text/head/greedy | 21.796 | 5.784 | 21.763 | 5.803 |
| Other hyper-connection envelopes | 17.333 | 17.370 | 12.730 | 12.812 |
| MoE combine | .136 | .131 | .108 | .103 |

Complete decode **compute-kernel sum** falls from 193.504 to 83.067 ms
(57.1% less). Greedy specifically falls from 10.518 to .034304 ms; head from
11.022 to 5.547872 ms. New dense, router, shared and head launches have graph
node IDs and selected shape-aware source; these are executed changes, not
unused candidate files. Routed low-bit arithmetic was intentionally unchanged;
its prefill/decode costs are effectively unchanged. KDA/sparse matrix prefill
also does not improve merely from the decode branch.

This is **not a 57% end-to-end gain**. Detailed all-API/graph-node tracing reports
272.302 ms final decode span and 146.463 ms positive kernel-free intervals on
rank 1, versus about 89–100 ms/token after prefill in most unprofiled repeated
requests. NCCL sums differ by rank (42.788/112.266 ms on egress/ingress), because
they include peer wait and are not pure network transfer time. The capture
records 72,448 egress `cuStreamQuery` calls over four steps and heavy graph/API
instrumentation. These waits cannot be treated as unchanged production bubbles
or converted directly to recoverable speedup. This run is a selected-kernel
timeline, not fresh instruction/stall-counter evidence or a new vLLM trace.
The short profiling prompt also differs from the six-prompt timing cohort;
instrumentation is not established as the sole cause of their latency difference.
All four egress steps still have zero `cuStreamQuery` calls before their final
collective enqueue; the previous whole-stream wait has not been reintroduced.

Remaining demonstrated implementation coverage:

1. **KDA still performs separate projections.** Nine projection roles account
   for 23.562 of the new 24.913 ms decode envelope; recurrence itself is .693 ms.
   The installed vLLM source merges Q/K/V/beta/f-a/g-a and keeps QKV merged through
   convolution. That remains a reusable multi-output-region opportunity, not
   justification for changing recurrence again.
2. **Routed-expert prefill is now the largest compute family:** 137.602 ms.
   It did not change in this patch. Shape-aware decoded BF16 execution must not
   be mistaken for a replacement of the native low-bit grouped program.
3. **Matrix prefill is still the simpler synchronous tile path.** This task
   connects the existing classifier/fold, not every mature multistage pipeline,
   row bucket or measured autotune record.
4. **Collective composition is unchanged:** 132 sites and 133 compute graphs per
   rank. Shared/routed partial combination and communication-region capture
   require explicit publication laws; the profiler's rank waits alone do not
   prove their unprofiled savings.
5. Small ordered normalization and publication folds remain where their numerical
   program requires them. The expensive serial router/head/dense/greedy paths
   were replaced in the selected alternative; this does not mean every scalar
   loop in every supported kernel has been removed.

Both timeline ranks exit zero and release CUDA. The lowest observed available
host memory is 19.690 GiB, still above the 16-GiB reserve, with zero task swap.
No failed reference-memory capture was retried and no payload hashing was added.
