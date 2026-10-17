# GLM HC fusion and vocabulary tensor partition experiment

Status: isolated kernels and first full-model ablations validated; repeated
full-model and refreshed reference measurements are in progress. No claim
that the remaining end-to-end gap has closed.

The starting point is commit `214ad93f`, with shared-column 32 and explicit
shared/routed overlap. The matched five-prompt cohort took 3218.5 ms versus
2774.733 ms for installed vLLM (15.99% more completion time). These are short
C1 prompts, not a long-context result; native and reference timer boundaries
and generated content are not identical. See the preceding prefill/overlap
report for the full workload and negative controls.

## Hypotheses and independent experiments

1. Fuse adjacent HC residual publication and the successor's square reduction.
   The intermediate BF16 round remains explicit. The physical plan may remove
   a launch and a reload, not alter arithmetic or move across a collective.
   Check the loss of producer parallelism: one row-owned fused workgroup could
   be slower than the separate, widely distributed publication. Keep that
   negative result if observed. Control/Sinkhorn fusion is a separate variant.
2. Partition the independent vocabulary output columns across TP ranks.
   Preserve every dot product and logit round. Join the greedy result using
   global token IDs and the original tie/nonfinite rules. Other sampling
   modes must retain their existing implementation, not silently use greedy.
3. Measure baseline, HC only, vocabulary only and their combination separately
   against the same prompt corpus and installed vLLM. No additive speedup
   assumption; trace actual selected kernels and collective boundaries.

Numerical and ownership contracts belong to immutable compiler plans; CUDA
and communication execution remain private lowering/owners. Numerical
comparison, hardware counters and performance evidence remain outside the
compiler. At most three measured variants per target before revisiting the
hypothesis. No payload hashes or integrity scans.

Stop only the authorized original GLM head/worker for the exclusive physical
campaign; retain NFS. Enforce 100 GiB per Spark, no swap and 16 GiB available
host memory during tests. Restore the original containers/configuration and
verify health afterward. The restored deployment has a larger pre-existing
footprint than this benchmark envelope; do not silently reconfigure it.

## Isolated results and rejected variants

All roots below begin `/tmp/lunaflux-glm-hc-vocab-` and retain raw paired
CUDA-event trials and failure logs. No model checkpoint was changed.

| Variant | GLM live-row result | Decision |
| --- | --- | --- |
| `postsquare-20261018-v1` | decode about 10.3–10.9 to 12.2 us; live64 also regressed | Unselected diagnostic; loss of publication parallelism |
| `tail-20261018-v1` | control/Sinkhorn fusion 26.74 to 26.65 us at live1 | Numerically exact but negligible decode benefit |
| `tail-20261018-v2` | subgroup-owned matrix, about 26.7 to 12.3 us at live1; live64 about 28.6 to 13.45 us | Candidate for full-model test |
| `vocabulary-20261018-v3` | full table about 5.44 ms; independent half tables about 2.73 ms each | Candidate; this is one-device leaf timing, not TP completion time |

The successful HC alternative keeps projection distributed, fuses the control
tail and Sinkhorn, and gives each matrix cell its own subgroup lane. Ordered
or butterfly sums still follow the numerical IR. A supplied subgroup
capability determines legality; larger matrices retain workgroup-local storage.
The CUDA backend supplies its subgroup width, not the model or scheduler.
`ControlTailFusion` and `AdjacentFusion` express the schedule and effect-boundary
rewrite; `-fused-controls` explicitly selects it. The post/square alternative
is not silently enabled.

HC compares post controls, initial/final matrices, rounded collapse and final
normalization bitwise at four geometries, live rows 0/1/tail/full/over-capacity.
Memory/full-leak, race and synchronization checks pass all four geometries.
Vocabulary compares every logit and greedy ID, including uneven partitions,
cross-rank ties, nonfinite logits and inactive outputs. Full-size memory/leak
checks pass; race/synchronization checks cover the small boundary case only.
The full-size vocabulary fixture uploads bounded chunks, rather than keeping
another full host table. These probes have a 2 GiB process limit, zero swap,
and check the 16 GiB host reserve before launching.

The vocabulary v1 preparation raced a changing local compile and failed;
its premature run found no executable. V2 exposed a missing newline in the
multi-source test exporter. V3 repairs the exporter and runs actual generated
source. Those failures are preserved; neither was a GPU numerical mismatch.
Initial full-suite compilation also overlapped edits and failed on a stale
function signature; it is not reported as a completed full-suite result.
The current affected suite passes 89/89; selected `moon info` reports existing
migration warnings (1063), zero errors. Warning exclusions are unchanged.

## Counter-level explanation

Scoped offline Nsight captures in the successful probe roots use the actual
generated entries and the same operands. They are not model-layer traces.
HC selects the live1 invocation after the empty-row control; vocabulary selects
the actual154880/4096 head and its two independent column partitions. Counter
containers are bounded to4GiB with no swap; no persistent driver settings are
changed. Raw `counters.csv` and `counters.ncu-rep` are retained.

HC old function+Sinkhorn executes12964+7027=19991 warp instructions; the fused
subgroup version executes13791 (**31.0% fewer**). L2 read sectors decrease
3951+2337=6288 to4471 (**28.9% fewer**). The old Sinkhorn has one active lane,
low active-warps2.72%, and47.53% long-scoreboard contribution. The fused entry
retains38 registers/thread, the same as the old function, and avoids the separate
serial global-memory matrix loop. Its long-scoreboard percentage still53% is
not a failure: percentages are not elapsed milliseconds, and less total work
finishes sooner. NCU replay times are not used in place of the paired timings.

Vocabulary instructions are exactly132267520 full-table versus66133760 for
each shard. L2 reads40.290M sectors become20.152M/20.143M per shard. Occupancy
remains about94–95%, registers37/thread, and long-scoreboard contribution stays
about52%. This improvement is **partitioning the memory work**, not hiding each
individual load better or increasing occupancy. The two-rank owner must still
demonstrate that candidate gather and rank arrival do not consume the saving.

Implementation pins: `ef6f2c4e` introduces explicit HC/vocabulary alternatives;
`74f1212e` adds subgroup-owned HC controls. Both are pushed to main. The latter
clean ARM release build passes in37seconds, peak700.3MiB, zero swap. The complete
local native suite subsequently passes **5199/5199**, using the repository's
unchanged warning exclusions; unrelated dirty migration files are preserved.

## Independent full-model ablations

These are source-pinned `74f1212e` native executions, not sums of isolated
kernel predictions. The same six literal prompts, one excluded warmup, TP2,
64-row capacity, low-bit experts, shared-column32/branch overlap and interval
graph policy are retained. Native input and output token IDs are identical in
all four runs. The post/square diagnostic remains disabled.

| First run | Six requests, ms | Sum of native first-token times, ms | After first tokens, ms | Five equal-output-count requests, ms |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 3995 | 1575 | 2420 | 3242 |
| HC controls only | 3902 | 1555 | 2347 | 3169 |
| Vocabulary TP only | 3842 | 1546 | 2296 | 3117 |
| Both | 3813 | 1533 | 2280 | 3093 |

Root prefix `/tmp/lunaflux-model-bench-glm-hc-vocab-`; suffixes
`base-20261018-v1`, `hc-only-20261018-v1`, `vocab-only-20261018-v1` and
`both-20261018-v1`. Raw journals, outputs, memory observations and
`comparison.json` are retained. The reference in those initial JSON files is
the frozen earlier comparison, not the fresh reference being collected now.

Single-run six-request completion reductions are 2.3%, 3.8%, and 4.6%.
They are not additive; repetition is needed before attributing a small
interaction to scheduling. Native output counts are `[8,4,8,8,8,3]`, whereas
the frozen vLLM counts are `[4,4,8,8,8,3]`. Therefore the five-request cohort
indices1–5 (31 output tokens) remains the primary cross-engine timing
comparison. Equal counts do not establish identical generated content or
quality. Native resident generation timing and vLLM buffered HTTP timing
remain distinct boundaries. The six-request comparison measures changes
within LunaFlux, where actual token work is unchanged.

Both ranks log `fused_hc_controls=true` and `vocabulary_tp=true` for the
combined run, with92 prepared interval graphs rather than91. Each vocabulary
rank plans100373434160bytes (93.48GiB) under100GiB. All four runs terminate
normally and explicitly close CUDA; swap is zero and minimum observed
host-available memory is21.76GiB, above16GiB. Selected-kernel trace verification
is separate from these unprofiled timings.

The reverse-order repeat (combined then baseline) gives3837/3988ms, retaining
every native ID. Thus baseline six-request mean3991.5ms versus combined3825ms
is a4.17% reduction; the equal-count cohort means3238/3098ms improve4.32%.
Baseline first-token sum mean1570ms becomes1543.5ms (1.69% less); after-first
time mean2421.5ms becomes2281.5ms (5.78% less). There are33 after-first tokens
in these six native requests, giving13.63 versus14.46tokens/s over that
after-first interval. This is not saturated multi-request serving throughput.

Repeating both individual ablations gives the following complete native
sample set. No prompt output IDs change, and all eight runs exit0 with
explicit CUDA release, zero swap and host reserve above21.75GiB.

| Variant | Six-request run1 / run2, ms | Six-request mean, ms | Mean time reduction | Equal-count five-request mean, ms |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 3995 / 3988 | 3991.5 | — | 3238 |
| HC controls only | 3902 / 3948 | 3925 | 1.67% | 3186.5 |
| Vocabulary TP only | 3842 / 3889 | 3865.5 | 3.16% | 3131.5 |
| Both | 3813 / 3837 | 3825 | 4.17% | 3098 |

Run2 roots use the same suffix pattern ending `-v2`. Execution order is
baseline1, HC1, vocabulary1, both1, both2, baseline2, separate instrumented
combined capture, HC2, vocabulary2. The combined mean is40.5ms (1.05%) below
the vocabulary-only mean. This is a modest incremental whole-model HC benefit,
not the isolated52–54% component saving applied to the whole model. The
combined166.5ms saving is less than the192.5ms sum of individual mean savings;
with two repetitions we cannot attribute that26ms interaction to a particular
cache/clock/scheduling cause or claim a statistical confidence interval.

## Selected full-model execution, not just emitted source

The combined trace root is
`/tmp/lunaflux-model-bench-glm-hc-vocab-both-node-20261018-v1`.
Both rank reports/SQLite databases are retained. Analysis roots are
`/tmp/lunaflux-glm-hc-vocab-{ingress,egress}-trace-20261018-v1`, produced by
`scripts/analyze-glm-selected-trace.mbtx`. The literal16-input/2-output probe
follows a warmup; steps3/4 are the measured prefill/decode. This is a different
workload and a heavily instrumented execution, not the unprofiled cohort.

Both ranks execute the19360-CTA half-head, local greedy, candidate all-gather
and final greedy join. Egress prefill/decode head times are2.842880/2.814784ms;
ingress2.819584/2.788480ms. Egress gather takes0.107168/0.061568ms, and join
0.003936/0.002240ms. Head, local greedy and join have nonzero graph-node IDs.
The head is no longer egress-only or a full38720-CTA table. Ninety ordinary
layer collectives plus one candidate gather are observed per step, with92
interval graph launches. No stream query occurs before the final collective
enqueue in the analyzed window.

HC executes360 rather than450 calls per step, including90 actual
`*_hyper_tail_function_tree_tiles_v1` entries and zero standalone Sinkhorn
entries. The fused tails sum1.571008/1.568448ms prefill/decode, with graph-node
IDs present. This verifies source → owner → rewritten execution plan → graph
propagation. The arithmetic/numerical probes remain separate correctness
evidence; selection counts alone do not prove correctness.

| Egress summed kernel scope | Previous baseline prefill / decode, ms | Combined prefill / decode, ms |
| --- | ---: | ---: |
| HC, exclusive attribution | 5.797280 / 4.755360 | 4.728288 / 3.561024 |
| Embedding/head/norm/greedy, excluding collectives | 5.864224 / 5.786080 | 3.095712 / 3.058912 |
| New candidate gather | absent | 0.107168 / 0.061568 |

The previous trace is the same node/all-API mode at
`/tmp/lunaflux-model-bench-glm-columns32-overlap-node-20261018-v1`.
Shared/routed overlap is retained: union intersections66.246144/11.980608ms
on egress and66.420640/11.649312ms on ingress. Overlapping kernel sums must
not be added to those overlap durations as an extra speedup.

All-API instrumentation still inflates the complete decode span to177.25ms
and egress collective residency to48.72ms. Those numbers are **not** native
production latency or pure network-transfer time. They cannot be subtracted
from the older vLLM trace to manufacture a remaining-gap budget. Unprofiled
repetitions supply the acceptance time; this trace establishes execution and
the changed local work.

The installed vLLM source snapshot's model.py lines443 onward defer HC post
into the following pre, and lines934–939 construct `ParallelLMHead`.
Snapshot root `/tmp/lunaflux-glm-installed-trace-sources-20261017-v2` corresponds
to the installed reference, unlike the newer local `../vlm` checkout.
The prior actual vLLM trace has181 HC calls and a3.396544ms vocabulary shard.
We have removed the full-vocabulary work deficit, but360 HC calls are still
not vLLM's181-call deferred post/pre path. The rejected post/square fusion
shows why deleting a boundary without retaining parallelism is not enough.
This comparison motivates further work; it is not a claim that all HC
rounding/effect constraints are interchangeable between the implementations.
