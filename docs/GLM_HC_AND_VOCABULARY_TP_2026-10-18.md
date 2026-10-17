# GLM HC fusion and vocabulary tensor partition experiment

Status: implementation and validation in progress; no speedup claimed.

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
