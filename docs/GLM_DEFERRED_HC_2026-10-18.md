# Deferred HC post/pre: preserved arithmetic, measured work distribution

## Scope and hypothesis

Continue `GLM_HC_AND_VOCABULARY_TP_2026-10-18.md` from main `ed83800e`.
The previous five equal-output-count prompts total 3098 ms native versus
2765.851 ms vLLM: 12.01% more completion time. This is not an attribution of
that whole difference to HC. The previous heavily instrumented trace is not
an additive unprofiled latency budget.

Use pure numerical laws -> legal physical ownership -> bounded row dispatch
-> effect-preserving launch rewrite -> private CUDA lowering. Comparisons and
measurements remain outside compiler correctness. No payload hashing, changed
tolerances, checkpoint edits, or production replacement. Preserve unrelated
toolchain/TLS migration changes.

Initial finite experiment: three control widths (1, 2, 4), holding arithmetic
fixed. Reprofile after the prefill regressions, then test operand reuse and a
bounded grid. Final model measurement is required before enabling the path.

## Installed reference, not just the newer checkout

The installed reference's
`vllm/model_executor/kernels/mhc/tilelang.py` (inside `glm53-exl3-head`)
confirms that <=16-token work uses fused post/pre FMA, tile width 2 below 8
tokens or 3 otherwise, and split-K 8 below 8 tokens for hidden<=4096, else 4.
Larger work uses separate post plus prenorm GEMM. It is not one universal
fusion geometry. The wrapper carries post/combination/residual across layers.

Its control projection takes F32 weights and normalization after projection.
Our current GLM numerical contract rounds controls/products and residual
publication to BF16 and normalizes operands before projection. Reassociation
is not a bitwise-preserving scheduling pass. The candidate retains all our
existing rounding, reduction trees and output ownership; it does not claim
the reference is a correctness oracle.

## Isolated actual-generated-source measurements

Common workload: streams=4, hidden=4096, capacity=64; actual rows vary.
Alternating seven trials, each CUDA graph contains 100 complete boundaries.
Times below are mean microseconds per boundary, not full-model token latency.
Correctness compares every residual, inverse, and control projection value.

| Actual rows | Initial baseline | Initial best fused | Revised baseline | Row-dispatched candidate |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 11.601 | 10.426 (width 1) | 11.424 | 6.822 |
| 22 | 27.691 | 30.046 (width 4) | 28.539 | 29.466 |
| 36 | 38.822 | 47.064 (width 4) | 40.838 | 28.650 |
| 64 | 62.552 | 76.963 (width 4) | 63.433 | 55.552 |

The final row records choose width1 for row1, width2 through row32, width4
above32, with a128-workgroup budget. These are an explicit experimental CUDA
catalog candidate, not model semantics or universal optimum. The immutable
IR validates supplied buckets, coverage and supported arithmetic. Every group
strides through its live domain; maximum capacity no longer requires all
capacity-sized CTAs. Private component ownership reuses branch/source operands
across destination streams, then performs the original flattened square fold.

Initial counters at root v2 show why fusion alone is insufficient: original
post/square/projection grids are4096/64/1536 even for a one-row decode. The
fused width1 grid remains1536, with95 registers versus24/31/38 for the old
three stages; its L2 read sectors are63501 versus a summed57245. Width4 cuts
reads to41059 but has only6 productive CTAs at one row and96 registers.
These are instrumented counters, not the graph timing above. Inactive CTA
launch work and repeated producer work must both be addressed. Kernel sums
and percent stalls must not be mistaken for full-engine gap attribution.

Evidence roots (local and Spark178):

- `/tmp/lunaflux-glm-hc-vocab-deferred-20261018-v1`: failed diagnostic header
  concatenation (missing newline); no GPU execution, retained.
- `...-v2`: original three-width probe and six-kernel NCU capture.
- `...-v3`: operand reuse + bounded grid, separate widths.
- `...-v4`: failed MoonBit exporter syntax, retained; no GPU execution.
- `...-v5`: row-dispatched source, graph timings and all safety gates.

Four geometries: (4,64,3), (4,320,4), (4,4096,64), (16,64,2).
Live counts0,1,capacity-1,capacity,capacity+1 retain baseline bytes (including
untouched inactive output). Memcheck/full leak, racecheck and synccheck all
report zero errors/hazards,2GiB/no swap, explicit CUDA release. The22-row loss
is retained, not hidden. These gates do not prove full-model quality.

## Queue integration and pending acceptance

The explicit `-deferred-hc` mode replaces adjacent post/square/project with
one row-dispatched launch. Existing control-tail fusion remains independent.
The pure contiguous-span rewrite rejects a collective or fork boundary inside
the removed span, and remaps all remaining effects. Both frame owners retain
their borrowed allocations/functions until the execution queue is closed.
Unselected serving behavior stays unchanged.

Full-model selection, paired unprofiled runs and comparison against vLLM are
pending at this commit. Do not infer a closed12% gap from the isolated result.

Implementation `213647b8` passed warning-denied native check with the unchanged
warning exclusions `-20-25-29-35-79-92`,88 affected tests, and the full native
suite5202/5202. Targeted `moon info` has1086 existing migration warnings and
zero errors. Clean commit-pinned ARM build passed in38 seconds with822.3MiB
peak memory and zero swap. Both AOT stages compiled within4GiB limits; existing
unused-variable/unsigned-comparison warnings are retained, not suppressed.

Fresh current-path baseline root
`/tmp/lunaflux-model-bench-glm-deferred-current-base-20261018-v1` completes with
all original output IDs unchanged: [714,445,714,730,734,460]ms for the six
prompts; equal-output five-prompt sum3083ms. Its requests finish at10:47:28;
the new CPU-only AOT export starts at10:48:02, so preparation did not overlap
these timings. Candidate model measurements remain pending.

The optional external `LUNAFLUX_REFERENCE_STREAM_DIAGNOSTIC=1` client adds a
separate warm six-prompt streaming replay after the ordinary buffered cohort.
It timestamps localhost receive events with nonempty, complete SSE content;
chunks are not claimed to be individual tokens. Raw curl trace/body are kept.
This provides first-content and following-content arrival times without GPU
profiling and without adding measurement code to the compiler/runtime.

## First model result and actual selected work

`glm-deferred-selected-20261018-v1` completes both ranks, preserves every
native output ID, and records at least21.956GiB host availability. Six timings
are697/462/710/735/740/462ms; equal-count sum3109ms versus fresh baseline3083ms.
This single observation does **not** establish an engine gain. The optional
CLI selection receipt is independently checked against the actual graph.

Trace `glm-deferred-selected-node-20261018-v1` (same16-input/2-output diagnostic
as the retained prior trace) executes89 deferred kernels and90 fused control
tails per step, with three initial/final HC launches:182 instead of360.
All182 are graph nodes; vocabulary remains sharded and91 collective sites
remain. Egress warmed prefill/decode HC sums are4.386496/3.247424ms versus
prior4.728288/3.561024ms. Thus decode saves about0.314ms in this instrumented
scope, not12% of a request. The profiler still substantially inflates whole
step/collective gaps; these sums are not an additive unprofiled budget.

The production dispatched kernel uses108 registers and7168 shared bytes,
not the80 registers of the separately profiled width-one entry. Its89 decode
calls sum1.614624ms; the remaining90 control tails sum1.612576ms. This exposes
two weaknesses of inferring engine performance from the hot repeated probe:
the complete bucket wrapper has a different resource envelope, and half of
remaining HC time is outside the replaced post/square/project chain.

Source inspection of the executed control tail finds independent sigmoid and
initial softmax cells still initialized by worker0. One additional bounded
candidate distributes those cells via existing `ControlTailFusion` ownership,
while retaining each row's declared maximum/sum order, strict exponential,
rounding and final publication. The immutable plan checks subgroup capacity;
larger geometries retain their existing implementation. This candidate is not
accepted on source inspection alone: paired original-fused/new-fused probes,
frontier correctness, sanitizers and full-model tests are required.

Separately, current prefill still contains42 one-CTA activation-scale scans
totalling9.301024ms (plus1.738496ms for input scales). Their256-thread grid
strides over all live selected rows in `quantize_fp4_rows.mbt`; the existing
two-level maximum tree is intra-CTA, not a distributed multi-CTA reduction.
These are observed residual work, not a claim of an equivalent removable
request delay. No quantization convention has been changed in this experiment.

## Maintenance and limits

Direct LAN SSH was unavailable; access recovered with the configured
`lunanexa-access` jump host and persistent foreground SSH masters. Original
head/worker were stopped using their exact previously recorded IDs, worker
first, at `/tmp/lunaflux-glm-deferred-maintenance-stop-20261018-v1`. NFS stays
running. Both nodes became GPU-idle with ample host reserve. Benchmark limits
remain100GiB/device and process, zero benchmark swap and16GiB host reserve.
The unchanged original EXL3/DFlash service must be restored afterward; its
pre-existing footprint does not satisfy the benchmark host-reserve policy.
