# GLM whole-path repair

This work follows the full inventory in
[packed expert transport](GLM_PACKED_EXPERT_TRANSPORT_2026-10-17.md), not just
the failed producer/consumer experiment. Completion requires executable
selection and whole-model evidence, not new IR types alone.

The implementation boundaries remain numerical program → legal regions →
physical domains/layout/lifetimes → execution effects → private device lowering.
Comparison and performance observations remain external. No payload hashing.

## Work ledger and bounded experiments

| Region | Required repair | Acceptance boundary |
|---|---|---|
| Expert operands | Preserve valid-row masks through packing; combine bulk transfer, consumer geometry and fragment lookahead | Exact full chain, padding/tails, sanitizer, five alternating paired trials |
| Quantization/epilogues | Parallel global reduction and avoid redundant materialization without moving BF16 publication | Exact payload/scales and complete chain timing |
| Non-expert projections | Buffered operand transport through physical plans; shared-input region reuse | Ordered MMA fold preserved, odd tails, full selected matrix chain |
| KDA | Shared-input projections, joint convolution, efficient chunk implementation | State/chunk equivalence and short/long timing separately |
| Sparse attention | Propagate producer's written prefix; share selection work; compressed-history alternative | Malformed-list behavior retained for unconstrained inputs; causal/history tests |
| Hyper-connections | Fuse post/pre/norm across semantic region boundaries | Preserve each declared publication/rounding and state lifetime |
| Shared experts | Independent branch execution with explicit join | No alias hazards, cancellation and complete critical-path timing |
| Tensor parallel | Smaller reduction domain under explicit numerical law; collective capture | Numerical program change disclosed; rank agreement, cancellation and bytes/timeline |
| Head | Shard vocabulary with global deterministic argmax | Tie semantics and rank-tail timing including collective |
| Evaluation | Fixed-work controls, broader rows/history and full-model ablations | Timing/token vectors, actual selected source and memory observations |

First campaign: exact physical-domain repairs and buffered matrix transport;
at most three schedule variants, five alternating component trials per cell.
Only successful candidates proceed to combined full-model testing. Changes to
scale/reduction order are separate numerical alternatives, never justified by
agreement with vLLM alone. Failures and non-wins remain recorded.

The original head and worker containers stay running during source work. Stop
only those two for GPU measurements, retain the NFS container, and restart the
same containers afterward. Task limits: 100 GiB per Spark, independently
observed 16 GiB MemAvailable reserve, no task swap and no overlapping GPU jobs.
The original deployment is restored unchanged even though its original memory
configuration does not meet this benchmark reserve.

No implementation or performance completion is claimed by this ledger.

## First physical repairs and measured alternatives

The sparse index producer now exposes its maximum written prefix through its
immutable physical plan. Only consumers bound to that producer shorten their
traversal; generic arbitrary-list validation still scans the complete extent.
The selected-list storage stride, stable ordering and arithmetic are unchanged.
At history 64 the old consumer unnecessarily traversed 2051 slots instead of
the producer's maximum 67. Actual generated-source paired trials at 22 queries
improve median .227632 to .062645 ms; single-query .073776 to .022582 ms.
History 4096 is unchanged (.77828/.77849 ms). This is a sparse kernel result,
not whole-model throughput. Root: `lunaflux-glm-sparse-retained-prefix-20261017-v1`.
Malformed lists, odd dimensions, finite/tail checks and all three sanitizers
pass with full leak checking, bounded 2 GiB/no task swap.

`MatrixStaging` separates physical two-slot publication/retirement from the
ascending K16 numerical fold. Explicit GLM `-buffered-matrices` propagates the
same choice through export, runtime reconstruction and recurrent projections.
Standalone generated-source probes include unaligned and odd-stride operands;
all outputs match the retained implementation bitwise and all sanitizers pass.
Three K schedules were tested, not an unbounded search:

| Shared K | R1 median ms old/new | R22 median ms old/new | R64 median ms old/new |
|---|---:|---:|---:|
| 128 | .11458/.12089 | .18219/.15521 | .20131/.33128 |
| 64 | .110323/.112876 | .178185/.111222 | .195513/.213170 |
| 32 | .105600/.107102 | .194826/.146491 | .195793/.148196 |

The K128 and K64 schedules are not blanket winners. K32 goes to explicit
whole-model testing, not default promotion. Roots:
`lunaflux-glm-buffered-matrix-20261017-v{1,2,3}`. Actual K32 Nsight replay at
R22 shows instructions **5.973M to 8.418M**, static shared memory 9.216 to
14.336 KB, long-scoreboard 63.11% to 58.61%, barriers 20.26% to 20.86%, and
profiled duration 329.344 to 305.024 us. A second pair is 326.272/296.448 us.
Thus copy overlap helps, but does not remove barrier or address-instruction
cost; the profiled gain is smaller than the unprofiled component result.

The indexed expert alternative uses cooperative double buffering without
packed-row materialization or producer polling. Its source has the same eight
launches and activation allocation as indexed transport. At full 288-bank
R22/capacity the independent-input variant changes median 6.786 to 6.532 ms;
R64 13.152 to 12.734 ms. R3 remains noisy and is not a demonstrated win.
R1/R3/R22/R64 capacity and trimmed chains are bitwise, with poisoned padding;
R3 full-bank capacity memcheck/leak/race/sync pass under 8 GiB/no task swap.
Root: `lunaflux-glm-fp4-performance-indexed-buffered-20261017-v2`.
This is still individual asynchronous copies, not CUTLASS TMA or its wider
consumer geometry. The earlier separate-role experiment remains a non-win.

The finite nonnegative max tree removes seven CTA-wide barriers from the
global quantizer without changing payload/scales. Its isolated R22 expert
chain median 6.781 to 6.748 ms is within observed variation, not a claimed
throughput improvement. Root: `lunaflux-glm-fp4-performance-max-tree-20261017-v2`.
The earlier v1 also changed gate/up input sharing and is explicitly confounded;
do not attribute its result to the max tree. v2 uses the actual selected
independent gate/up staging. No numerical reduction/scale law changes.

These repairs do **not** complete the remaining cross-boundary HC fusion,
shared-expert overlap, collective capture/reduction-domain, head sharding,
shared-input KDA transport, compressed sparse-history and expert TMA/fragment
lookahead items above. Full-model results must be recorded separately below.
