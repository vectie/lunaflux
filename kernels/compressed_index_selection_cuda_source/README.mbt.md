# Learned compressed index scores and selection

Consumes transformed BF16 query heads, already-scaled BF16 head weights and
published learned compressed-cache rows. Unlike the older GLM pool-expanded
index, it does not expand selected IDs into raw token positions or apply a
second dimension/head scale. Dot, weighted product and final head reduction
preserve separate BF16 publication boundaries.

Two effects compute scores once, then select causal compressed row IDs with a
deterministic lower-row-ID tie break. Outputs carry one selected count per
query. Invalid append metadata, cache length, position or offset produces an
empty sentinel row. Zero-capacity plans allocate inaccessible backing only.
This is a correctness-first full-head path, not a distributed head reduction
or a demonstrated high-throughput long-context selector.
