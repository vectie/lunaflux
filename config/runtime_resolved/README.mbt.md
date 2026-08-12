# Resolved runtime capacity

This package performs startup-only cross-component capacity resolution. It
combines already validated scheduler, cache, model-shape, and worker-protocol
records, rejects incompatible envelopes, and materializes immutable host-side
page-allocator and request block-table limits.

The submitted-row ceiling is
`min(max_active_requests, step_token_budget)`. Active requests may exceed a
single model batch: only the rows in one submitted step must fit the model and
worker. Both prefill and decode row limits must independently hold the full
ceiling because a legal step may contain only one row kind. Worker page-table
storage conservatively holds a complete context-sized table for every row.
The explicitly configured output-event capacity is the scheduler-owned
generated-token publication ring. It must hold at least one complete submitted
row set so completion retirement can preflight all token publications before
mutating request state. Terminal notices use a separate total-request-slot
ring, and downstream transport queues are outside this capacity.
The worker capability-cell limit is preserved in the resolved value. Exact
capability cells per row are validated separately when provenance-bound
`RowCapabilityRecipe` values are attached to the loaded model generation;
kernel or backend guessing in this startup package would make the capacity
report less truthful.

Block-table capacity is the active-request capacity; waiting requests do not
own KV mappings. Page and block-table generations use the maximum defaults
owned by their respective KV packages, whose allocators retire terminal slots
instead of wrapping identities.

Resolution does not allocate device KV memory and does not prove a paged-KV
layout, paged-attention kernel, device executor, or serving path. Those remain
Phase 3 integration and physical-hardware gates.
