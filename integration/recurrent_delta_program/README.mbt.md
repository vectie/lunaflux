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

`with_block` adds the complete hidden-to-hidden recurrent attention branch:
Q/K/V projections, low-rank forget projection, beta projection and sigmoid,
bounded F32 decay, low-rank output gate, Q/K/V convolution, recurrent update,
sigmoid-gated per-head RMSNorm and final dense projection. All 16 launches share
one queue and completion event. `prepare_block` borrows named checkpoint weights,
hidden/output and sequence metadata; the Program owns all intermediate frames.
No warmed-path buffers or plans are constructed between stages.

`RecurrentBlockPrecision` owns numerical stage boundaries and shape accounting;
the model adapter selects geometry and epsilon, and CUDA lowering owns execution.
Projection sums retain ordered F32 arithmetic and explicit BF16 stage rounds.
Unlike the old single-thread composite oracle, output elements run in parallel.
These correctness-first projections are not tensor-core performance kernels.

The complete branch passes a small GPU oracle and chunked-prefill/decode test,
with native ownership and zero-allocation queue tests. Actual checkpoint binding,
surrounding mHC/residual/MLP composition, DSA layers and model/worker integration
remain unfinished. This does not claim complete GLM inference or serving speed.
