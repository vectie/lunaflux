# Functional LunaTile projection compiler

This package is the pure, backend-neutral middle end for projection workloads.
It turns an immutable shape-selected dispatch plan into phase-typed Selected,
Semantic, Optimized, and Scheduled values. The semantic value graph makes
query-row selection explicit; the optimizer records demand pruning, selected
input-tile hoisting, and cross-output input-tile reuse; the schedule expresses
parallel row/output maps and the ordered reduction fold without vendor terms.

Gated MLP is represented as a pure value graph rather than an opaque kernel:
two sibling dot products consume one input value, SiLU gating produces one
intermediate value, and the down-projection fold consumes that intermediate.
The middle end therefore records sibling-traversal fusion and intermediate-tile
reuse once. A device backend lowers those sharing decisions to its own local
memory and synchronization primitives; model code never names them.

Attention ingress is also a composable projection epilogue. Its immutable value
graph is `QKV dot -> per-head Q/K RMSNorm -> positioned rotary -> output store +
paged KV commit`. The optimizer records that the projected QKV round trip can be
elided, while the schedule exposes head and head-component parallel maps. The
same semantic epilogue can therefore be lowered by CUDA, HIP, Metal, or CPU
backends without placing warp width, page addressing instructions, or vendor
types in the compiler middle end.

`QueryRowEnds` is legal only for a language-model head. It represents the
general decoder-serving rule that a row-wise next-token consumer observes only
the final token of each packed query row. Full-logit callers retain
`AllTokenRows`. This distinction lets the compiler remove unobserved vocabulary
projections without changing model-family semantics.

The compiler performs no I/O, device probing, benchmarking, or runtime
allocation. CUDA, HIP, Metal, and CPU backends may lower the same scheduled
value differently. Device instruction names, subgroup widths, and launch
geometry remain outside this package.
