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

## Follow-on work

1. Derive reuse and fusion legality from live value dependencies instead of
   semantic-family labels, preserving observable stores and KV commits.
2. Generalize variant collections and compile bucket-to-variant tables at
   startup; remove the duplicated base/deep partition runtime representation.
3. Share analysis primitives across attention and projection dialects; add
   semantics-preserving CSE/DCE and explicit numeric/effect contracts.
4. Separate semantic optimization from schedule exploration, then incorporate
   actual resource feedback and invalidate autotune records by final schedule
   and code-generation identity.

Validation distinguishes structural/unit tests from physical correctness and
end-to-end performance. New dispatch metadata requires regenerated artifacts
and a focused GPU campaign before a speed or serving-readiness claim.
