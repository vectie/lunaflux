# GLM HC fusion and vocabulary tensor partition experiment

Status: isolated kernels validated; full-model ablations pending. No claim
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
