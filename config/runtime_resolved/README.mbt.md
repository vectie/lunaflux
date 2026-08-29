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
Recompute-only preemption stores generated token IDs in one dense startup
array. Its checked cell count is
`total_request_slots * (model_max_sequence_tokens - 1)`; overflow is rejected
before scheduler allocation.
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

Block-table capacity is the active-request capacity. Waiting requests retain
only logical prefix-match evidence and acquire a physical table, pages, and an
entry reference transactionally at activation. Resolution therefore bounds
shared-page and per-entry active references by the active-request capacity,
while request slots and recompute history still cover active plus waiting
requests. Resolution also materializes the exact compressed-radix
entry/node/token/page, scope-byte, active-reference, and layout envelopes. The
per-entry token ceiling must cover at least one complete page, and the aggregate
page-cell arena must cover every full page permitted by that ceiling; unusable
prefix envelopes are rejected before scheduler allocation. Page and block-table
generations use the maximum defaults owned by their respective KV packages,
whose allocators retire terminal slots instead of wrapping identities.

Resolution does not allocate device KV memory and does not prove physical CUDA
cached-vs-uncached numerical equivalence, a paged-attention kernel, or a
hardware reuse benchmark. Those remain explicit physical-hardware gates.

The package also owns a separate startup-only graph-memory join. It sums the
weight, activation/workspace, and persistent-KV arenas exactly once, then adds
an authenticated positive graph-memory declared upper bound only when a
positive independently pinned v2 descriptor ceiling contains it. V1 is
eager-only and therefore projects `Absent` even when inert authenticated
metadata carries a future capture bound. A v2 ceiling with absent or
capture-unsupported evidence fails closed. This is capacity accounting, not an
observation of CUDA driver allocation behavior.
