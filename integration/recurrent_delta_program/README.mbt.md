# Request-owned recurrent delta execution

The pure precision IR specifies normalized causal delta arithmetic, BF16 frame
operands, F32 persistent state and its memory ceiling. CUDA lowering assigns a
sequence/head to a CTA and independent value columns to lanes, retaining the
ordered key fold and token recurrence. State is read/written once per frame;
the token loop has no global state publication.

The Program owns one layer's cache, AOT functions and reusable completion queue.
The caller owns frame buffers and CSR sequence metadata. Explicit request slot
IDs—not batch ordinals—address persistent state, so compaction and reordering
do not change which request resumes. Slots must be exclusive within a frame;
reset flags discard old state when a slot is reused for a new request.

No request-path compilation, allocations, tensor validation scans or host state
readback are added. Abort drains the queue and terminally discards its cache;
partially updated cancelled state must not resume. Initialization is startup-only
with bounded 64 KiB startup host scratch. Close releases queue, functions,
intermediates and cache.

`with_convolution` now composes request-owned Q/K/V short convolutions followed
by recurrent update, with one completion event and no warmed-path allocation.
It requires matching capacities and symmetric Q/K/V geometry; the standalone
constructor retains asymmetric recurrent geometry. Supply Q/K/V weights in that
order at preparation. `state_bytes` covers all four persistent caches;
`workspace_bytes` additionally includes three convolved frame buffers. Both
startup budgets are explicit. `source_bytes` includes both startup AOT symbols.

This is still not a complete GLM attention block. Projection, decay/beta controls,
output gating and model/worker composition remain separate responsibilities and
must be wired before whole-model claims.
