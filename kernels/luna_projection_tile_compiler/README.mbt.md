# Functional LunaTile projection compiler

This package is the pure, backend-neutral middle end for projection workloads.
It turns an immutable shape-selected dispatch plan into phase-typed Selected,
Semantic, Optimized, and Scheduled values. The semantic value graph makes
query-row selection explicit; the optimizer records demand pruning, selected
input-tile hoisting, and cross-output input-tile reuse; the schedule expresses
parallel row/output maps and the ordered reduction fold without vendor terms.

Offline fold choices independently refine row and output reuse multiplicity.
`row_tiles` bounds the parallel row map (including partial evaluation for small
domains), while `output_tiles` changes matrix-column sharing within the admitted
consumer envelope. Neither changes a dot's reduction ordering or rounding.
`ProjectionProducerAddressPlan` separates each worker's immutable row/vector
coordinate from the varying reduction offset; CUDA lowering may retain these
addresses across ordered stages without moving operand reads across effects.
The same choices propagate through bounded row executables, so a tuned primary
does not silently revert to ordinary geometry when a graph bucket is selected.
Larger reuse is an executable alternative, not an unmeasured default winner.

Full projection/ingress packing is represented separately from numerical head
ownership. `with_ingress_head_tiles` expands only the producer's independent
column domain; it leaves the semantic graph, ordered dot fold, and per-head
BF16 rounding law unchanged. `AttentionIngressPacking` resolves Q/K/V operand
rows across segment boundaries and returns no authority for padded heads.
Its explicit prepare/consume/retire effects give rotary pairs token-row
lifetime across complete-head consumers. Resource policies decide the retained
column window; the terminal backend is responsible for realizing that lifetime,
not for inventing a model-specific fusion or numeric reassociation.

Aligned split-half normalization/rotary pairs now retain their original
strided component owner through `SplitPairRetention`. The physical refinement
loads first-half components followed by second-half components, preserving the
original square-sum order and both BF16 rounding boundaries. CUDA can retain
the normalized pair in registers rather than publish/reload it through shared
scratch. Cross-owner halves keep their existing explicit publication path;
alignment is a legality decision, not an unmeasured performance claim.

The ingress numerical plan can group independent complete head folds into a
workgroup. `AttentionIngressHeadMap` distributes only the product's head axis;
the lane count, ordered reduction, rotary basis, and BF16 rounding boundaries
do not change. Tail subgroups own no head or KV write. CUDA lowers this map to
subgroup-local storage and synchronization rather than whole-block barriers.

Single-row dot scheduling also supports an explicitly authorized deterministic
tree. `PreserveDotOrder` remains the default. `AllowDeterministicDotTree` permits
a `StridedPairwiseDot(lanes)` schedule only when the selected decode strategy
supports a power-of-two subgroup; otherwise it retains `OrderedDot`. Each lane
folds its strided products in increasing index order, then a descending-distance
pairwise tree combines partial sums. Products and sums round separately. This
is not equivalent to an ordered FP32 fold and must be checked against the
caller's numerical tolerance. Purity does not make floating-point addition
associative; normalization/CSE never grants this permission.

The permission is bound into the semantic program and the chosen topology into
the v7 schedule (v6 for ordered requests). These versions replace the obsolete
always-false `reuse_input_tile` toggle with an explicit scalar/masked-matrix
row-tail policy. Operand lifetime and reuse belong to the fold plans. The CUDA fused attention
ingress is the first consumer: all four subgroups share output columns and
cooperatively load contiguous reduction elements for a single token. Its
single-token epilogue remains unchanged. This is a reusable
projection schedule, not a model-specific rewriting rule or an autotuner.

### Physical column-fold refinement

`ProjectionTileSchedule::column_fold_plan` now refines the scheduled output
map into an immutable `ProjectionColumnFoldPlan`. The backend supplies consumer
group count and a live-accumulator budget. The pure plan partitions independent
columns into bounded windows and specifies `window -> ordered reduction ->
column map`: the input fragment is evaluated once per reduction step per
consumer and reused across that window's columns. Tail consumers remain in
workgroup effects even when they own no column. No reassociation is permitted.

For ingress, the projection domain is one head's component axis, not the
concatenation of every Q/K/V head. The CUDA full-ingress lowering consumes this
plan, retains at most two column accumulators per row fragment per consumer, and no longer
repeats input loads and padded-tail barriers separately for every column round.
The physical plan identity is included in the AOT recipe. Single-token dot
order, projection BF16 materialization, Q/K normalization and KV commits are
unchanged. Two live fragments are a static backend envelope, **not** a measured
optimum; register pressure and runtime still need physical evaluation.

Ingress requests can independently choose the CTA row-tile multiplicity.
The semantic program and ordered reduction stay unchanged; the schedule binds
the wider row extent. Physical column-window ownership assigns every row to
its column consumer (not the ordinary two-axis product distribution). Fragment
rows are derived from that ownership, not overridden by the CUDA emitter.
The CUDA backend enumerates row factors 1/2/4/8 subject to actual shared-memory
bounds; source-bound offline resource selection chooses among them. The
unmeasured default remains factor 1. Larger factors are not presumed faster.

Ingress also refines its head domain into the common `ProjectionFoldPipeline`.
The CUDA consumer uses the common operand transfer plan and finite-ring effect
renderer. Offline matrix-fold choices select transfer width, operand stage
count and fragment lookahead; ordered accumulation is retained. Backend static
shared-memory limits constrain these choices. None is a measured winner merely
because it is the default.

This fills physical-loop and lifetime boundaries, not a new parallel compiler
stack. Existing Semantic/Optimized/Scheduled types remain authoritative.
The semantic ingress cut lives in `luna_fusion_plan`: it records which values
remain materialized and whether ingress owns KV writes. CUDA instruction and
layout realization stays in the backend.

Before reuse analysis, normalization removes unreachable pure bindings and
performs typed, exact common-subexpression elimination in topological order.
Operand substitution exposes cascading duplicates; dense value numbering
normalizes alpha-renamed IDs. Stores and KV commits are preserved in order and
end each CSE region. Reuse and ingress-elision analysis use the same conservative
effect-region boundary: a producer can fuse into its terminating effect, but
not cross an unrelated store or KV commit. No floating-point reassociation or
approximate matching is performed. This is backend-neutral graph rewriting,
not a Qwen-specific rule.

Reuse and ingress-elision decisions are derived by pure use-def analysis over
the topological value graph, not just by inspecting the semantic-family tag.
A reverse demand fold treats output stores and KV commits as observable roots;
dead sibling computations do not manufacture reuse opportunities. Ingress
round-trip elimination requires exclusive live use of each projected,
normalized, and positioned intermediate. Scheduling consumes the resulting
rewrite decisions rather than reconstructing them from family labels.
The analysis consumes the normalized graph, so merged dots cannot manufacture
sibling-reuse opportunities. `program()` returns that graph; `source_program()`
retains the input graph, and `eliminated_bindings()` reports CSE/DCE counts.
Source-dependent counts are separate from compiled-code identity: alpha
renaming, dead insertions, and exact duplication that normalize to the same
graph produce the same schedule and compilation digest.

This is not arbitrary-DAG CUDA lowering or a complete effect/alias system.
Existing backend templates still implement the supported elaborated families;
normalization does not expand their supported graphs. Those current graphs
are already normalization fixed points, so this change does not claim a GPU
speedup or change their floating-point evaluation order.

Gated MLP is represented as a pure value graph rather than an opaque kernel:
two sibling dot products consume one input value, SiLU gating produces one
intermediate value, and the down-projection fold consumes that intermediate.
The consumer map has its own output-tile distribution, independent of the
producer's workgroup. This policy is part of schedule identity; CUDA lowers it
to separate launch dimensions without changing the ordered reduction or BF16
materialization boundary. See [companion launch results](../../docs/COMPILER_COMPANION_LAUNCH.md).
The middle end therefore records sibling-traversal fusion and intermediate-tile
reuse once. A device backend lowers those sharing decisions to its own local
memory and synchronization primitives; model code never names them.

The materialized intermediate fold also carries an immutable two-stage block
pipeline: a bounded row map (up to 64 rows), 64-element consumer transfers, and unchanged 16-element ordered
reductions. Full blocks and masked row tails share the same fold; single-row
execution factors one input across four independent output folds. CUDA is the
first lowering. MLP sibling input transfers use 64 elements for a bounded
single row tile with at most eight consumer groups, and 32 for wider products
to retain the existing storage envelope; complete admitted
MLP matrix extents start at 256. QKV, output, and head matrix pipelines select
16/32/64-element transfer groups from their reduction extent and storage plan.
See [current coverage](../../docs/UNIFIED_COMPILER_COVERAGE_2026-09-09.md).

Row-domain partial evaluation happens before operand storage and consumer
distribution: a maximum of 16 rows retains one padded matrix row microtile,
not two QKV/output tiles or four sibling/down tiles. This removes unobserved
parallel-map members, without reordering the reduction of any live result.
The AOT exporter compares the resulting pipeline plans, not merely strategy
names, so these bounded executables survive even without new autotune records.
An unmeasured bounded version retains the installed primary distribution;
only a measured record may replace that distribution. The full-profile
allocation and single-row numerical contracts remain unchanged.

Attention ingress is also a composable projection epilogue. Its immutable value
graph is `QKV dot -> per-head Q/K RMSNorm -> positioned rotary -> output store +
paged KV commit`. The optimizer records that the projected QKV round trip can be
elided. It also records two pure rotary rewrites: hoisting the
position-independent inverse-frequency basis and evaluating each paired rotary
component together. The schedule exposes those decisions plus head and
head-component parallel maps. A backend chooses constant storage and a paired
trigonometric intrinsic when available. The same semantic epilogue can
therefore be lowered by CUDA, HIP, Metal, or CPU backends without placing warp
width, page addressing instructions, vendor types, or vendor intrinsics in the
compiler middle end.

`QueryRowEnds` is legal only for a language-model head. It represents the
general decoder-serving rule that a row-wise next-token consumer observes only
the final token of each packed query row. Full-logit callers retain
`AllTokenRows`. This distinction lets the compiler remove unobserved vocabulary
projections without changing model-family semantics.

For a single-output-tile matrix strategy with streaming selected rows, the
compiler binds selected input directly to the consumer's register fragments.
The row-selection map is loop-invariant, and each result element has one
consumer owner, so the intermediate shared input and result tiles disappear.
The 16-element matrix folds and final BF16 rounding remain ordered; no
floating-point reassociation is granted. This replaces the former 9,216-byte
reduction-strip arena with zero local-memory storage. Eligibility requires
complete 16-element reduction and output tiles; explicit resident and
multi-consumer strategies retain their declared storage. CUDA lowers this
placement through direct packed fragment loads and unique result stores. The
existing single-row reduction is unchanged. Offline records still select the
strategy; this source change alone does not claim a measured speedup.

The selected-row register fold now carries a bounded lookahead window of up to
eight immutable operand fragments. It primes the window, evaluates each future
operand before consuming its current slot, then consumes slots in the original
reduction order. Small reductions shrink the window; a partial final window
performs neither an out-of-range load nor an extra zero-product reduction.
The window is independent of total K and is part of schedule identity. This
changes evaluation timing, not floating-point association or memory layout.
Actual load/compute overlap and speed still require device measurement.

The CUDA materialized consumer issues the next slot's asynchronous copies
before consuming the current slot, including small workgroups. The bounded
sibling product lowers its three immutable operand domains separately, so
source selection disappears before rendering; its matrix consumer uses packed
fragment loads. Neither transformation changes the ordered fold or epilogue
rounding. Counter and end-to-end measurements determine the performance result.
See the [measured operand-supply optimization](../../docs/OPERAND_SUPPLY_OPTIMIZATION_2026-09-10.md)
for the selected head/down/sibling changes, full concurrency/token vectors,
numerical coverage, and remaining baseline gaps.

Before source emission, `ScheduledProjectionTileCompilation::refine_physical`
retains a checked `PhysicalProjectionTileCompilation`. It binds scalar laws,
operand ownership and layouts, transfer segments, fragment rings, selected-row
windows, weight selection, and effect lifetimes together. A required refinement
failure rejects the compilation; it cannot silently switch to another route.
CUDA emitters consume this retained value, including transport mode, rather
than reconstructing independent producer and consumer plans. Backend binding
checks instruction vector and subgroup geometry separately from generic IR.

Sibling pointwise materialization is now a separate immutable ownership plan,
not an implicit consequence of a CUDA renderer. `SplitSiblingPlanes` retains
the existing measured two-plane exchange. `RetainSiblingProducer` keeps one
ordered fold in its consumer registers and publishes only its peer through a
retired operand ring. `CoownedSiblingValues` gives the same consumer both ordered
folds; their pointwise map requires no result plane or workgroup exchange.
The latter removes the split consumer topology during physical refinement.
These are generic two-fold ownership choices, not model-specific fusion rules.
All three retain the same reduction and strict pointwise numerical contract.
Physical plans expose distribution, result-plane count, and explicit
publication/retirement effects. CUDA only realizes the chosen plan.

Ownership can be combined with a finite sibling consumer-group choice and
independent row/output geometry. This jointly refines accumulator ownership,
operand storage, schedule identity and launch topology; it is not a diagnostic
block-size override. The down fold keeps its own geometry. Reduced consumer
counts preserve the scalar fallback's output coverage even in small bounded
variants. More register ownership can reduce resident workgroups and can be
slower despite removing shared result planes, so these choices are not promoted
without complete-chain measurements and selected-kernel counters.

Alternatives are exportable and selectable through source-bound v2 offline
fold observations, including all bounded-row executables. No measurement keeps
the baseline; an alternative must improve the complete declared chain, not
merely its epilogue or one GEMM, before selection. A source-bound record is a
selection input, not a replacement for physical correctness and sanitizer tests.

The compiler performs no I/O, device probing, benchmarking, or runtime
allocation. CUDA, HIP, Metal, and CPU backends may lower the same scheduled
value differently. Subgroup width arrives as an abstract capability; device
instruction names, fixed vendor widths, and launch geometry remain outside
this package.

Bounded fold choices refine independent row/column products and consumer
cooperation before schedule construction. The narrowest applicable token-row
domain wins, with an unbounded measured fallback. Every row variant receives
its own resolved physical plan; its executable source, primary launch and
companion launch are derived together. This avoids changing only a diagnostic
block size or preserving a scalar grid floor on a matrix row variant. Ordered
reduction and pointwise numeric contracts remain unchanged.
