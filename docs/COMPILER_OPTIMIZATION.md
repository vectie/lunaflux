# Functional compiler optimization

LunaFlux evolves its existing production kernels through pure, typed compiler
transformations. Functional programming is a design discipline, not a claim
that a trace entry alone implements a general optimizer. Device mutation stays
explicit in lowering and execution; compilation and search stay offline.

## Semantic schedule metadata

The reusable runtime bundle wire schema v4 carries optional `query_tile_rows`
for each module. The candidate exporter emits the selected compiler tile into
the recipe; the bundle exporter retains it independently of launch grid size.
Partition partial and merge entries must agree. The runtime uses these values
for launch caps and partition dispatch instead of reconstructing them from
`grid_x` or assuming a fixed tile.

Schema v3 remains readable, but its missing metadata is represented as absent,
not guessed. Compiler phase-specific prefill and partitioned variants without
metadata must be regenerated before use with the updated runtime. Legacy
mixed attention remains supported. Existing build-script filenames ending in
`.v3` are retained as path compatibility; the content declares schema v4.
No production deployment is performed by this change.

## Projection dataflow analysis

The projection optimizer now derives live values with a pure reverse fold of
topological bindings. Stores and paged KV commits seed observable demand even
when their results are not the selected root. Shared dot inputs, gated
intermediates, and exclusive attention-ingress chains are recognized from
dependencies rather than a model-family tag. Dead consumers do not affect use
counts; additional live consumers prevent ingress round-trip elimination.
The schedule consumes those decisions instead of independently reconstructing
them from the epilogue or family. Existing generated graphs retain their
floating-point evaluation order.

## Projection graph normalization

The next slice adds real IR transformations before reuse analysis and
scheduling: demand-rooted DCE, exact typed CSE with operand substitution, and
dense SSA value numbering. Identical expressions merge only within the same
effect-free region, with the same value kind and ordered operands. No
commutative rewriting, floating-point reassociation, or approximate matching
is enabled. All stores and paged KV commits remain observable and ordered;
each ends the CSE region, including reuse of input and weight reads, until an
explicit alias model can justify a less conservative rule.

The original semantic graph is retained separately from the normalized graph.
Scheduling and compiled identity consume the latter. Diagnostic elimination
counts do not enter the code identity: source graphs differing only by
alpha-renamed IDs, dead insertions, or exact duplicates can converge to the
same compiled result. This does not promise a canonical form for arbitrary
algebraic equivalence or permutations of independent bindings.

Tests cover cascading CSE, DCE, source immutability, normalization idempotence,
identity convergence, typed/operand-order distinctions, read/effect barriers,
and ordered symbolic observations across the norm/rotary/KV chain. The current
elaborated dense, QKV, head, MLP, and ingress graphs are fixed points: this
increment does not change their generated kernel computation or claim a new
GPU speedup. Arbitrary-graph CUDA lowering and a full effect/alias type system
remain follow-on work.

## Follow-on work

1. Extend dependency-derived reuse and fusion legality to attention and share
   the analysis infrastructure between the two dialects.
2. Generalize variant collections and compile bucket-to-variant tables at
   startup; remove the duplicated base/deep partition runtime representation.
3. Share analysis primitives across attention and projection dialects; extend
   projection's exact CSE/DCE with explicit numeric/effect contracts and
   broader lowering coverage.
4. Separate semantic optimization from schedule exploration, then incorporate
   actual resource feedback and invalidate autotune records by final schedule
   and code-generation identity.

Validation distinguishes structural/unit tests from physical correctness and
end-to-end performance. New dispatch metadata requires regenerated artifacts
and a focused GPU campaign before a speed or serving-readiness claim.

## Validation of this increment

Warning-denied native check and the affected compiler, exporter, bundle,
device-step, device-worker, and fused-AOT tests pass. The whole native suite
does not pass: the FP8 projection test `FP8 dense and head identities cannot
replay across operation or profile` aborts in its operand fixture. A separate
projection-AOT test also has a pinned recipe digest mismatch (13/14 pass).
Both failures reproduce unchanged in a clean archive of pre-change commit
`2f93ab7`; neither fixture is relabeled or updated by this increment. No new
physical benchmark result is claimed.

The graph-normalization increment passes warning-denied native check and all
19 projection-compiler tests (including eight new normalization regressions).
The affected fused-AOT and Qwen-exporter tests pass 28/28; projection-AOT
lowering, contract, and non-physical fixture tests pass 13/13. The known pinned
physical-fixture failure above is not included in that targeted 13-test count.
The symbolic observation tests verify graph substitution and effect ordering,
not CUDA numerical accuracy or end-to-end throughput.
