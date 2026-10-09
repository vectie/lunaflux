# Request-owned recurrent delta execution

The pure precision IR specifies normalized causal delta arithmetic, BF16 frame
operands, F32 persistent state and its memory ceiling. CUDA lowering assigns a
sequence/head to a CTA and independent value columns to lanes, retaining the
ordered key fold and token recurrence. State is read/written once per frame;
the token loop has no global state publication.

The Program owns one layer's cache, AOT function and reusable completion queue.
The caller owns frame buffers and CSR sequence metadata. Explicit request slot
IDs—not batch ordinals—address persistent state, so compaction and reordering
do not change which request resumes. Slots must be exclusive within a frame;
reset flags discard old state when a slot is reused for a new request.

No request-path compilation, allocations, tensor validation scans or host state
readback are added. Abort drains the queue and terminally discards its cache;
partially updated cancelled state must not resume. Initialization is startup-only
with 64 KiB reusable host scratch. Close releases queue, function, then cache.

This is the recurrent transition, not a complete GLM attention block. Projection,
decay controls, short convolution, output gating and model/worker composition
remain separate responsibilities and must be wired before whole-model claims.
