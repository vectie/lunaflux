# Functional LunaTile projection compiler

This package is the pure, backend-neutral middle end for projection workloads.
It turns an immutable shape-selected dispatch plan into phase-typed Selected,
Semantic, Optimized, and Scheduled values. The semantic value graph makes
query-row selection explicit; the optimizer records demand pruning, selected
input-tile hoisting, and cross-output input-tile reuse; the schedule expresses
parallel row/output maps and the ordered reduction fold without vendor terms.

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
multi-token matrix path and epilogue remain unchanged. This is a reusable
projection schedule, not a model-specific rewriting rule or an autotuner.

Before reuse analysis, normalization removes unreachable pure bindings and
performs typed, exact common-subexpression elimination in topological order.
Operand substitution exposes cascading duplicates; dense value numbering
normalizes alpha-renamed IDs. Stores and KV commits are preserved in order and
end each CSE region. No floating-point reassociation or approximate matching is
performed. This is backend-neutral graph rewriting, not a Qwen-specific rule.

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

The compiler performs no I/O, device probing, benchmarking, or runtime
allocation. CUDA, HIP, Metal, and CPU backends may lower the same scheduled
value differently. Subgroup width arrives as an abstract capability; device
instruction names, fixed vendor widths, and launch geometry remain outside
this package.
