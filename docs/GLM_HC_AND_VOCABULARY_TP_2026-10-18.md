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
