# Functional LunaTile projection compiler

This package is the pure, backend-neutral middle end for projection workloads.
It turns an immutable shape-selected dispatch plan into phase-typed Selected,
Semantic, Optimized, and Scheduled values. The semantic value graph makes
query-row selection explicit; the optimizer records demand pruning, selected
input-tile hoisting, and cross-output input-tile reuse; the schedule expresses
parallel row/output maps and the ordered reduction fold without vendor terms.

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
The middle end therefore records sibling-traversal fusion and intermediate-tile
reuse once. A device backend lowers those sharing decisions to its own local
memory and synchronization primitives; model code never names them.

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

The compiler performs no I/O, device probing, benchmarking, or runtime
allocation. CUDA, HIP, Metal, and CPU backends may lower the same scheduled
value differently. Device instruction names, subgroup widths, and launch
geometry remain outside this package.
