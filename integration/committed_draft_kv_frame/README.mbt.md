# Committed-main/draft execution frame

One owner retains the main BF16 ring, frontier/error descriptor and draft
output. Both startup-prepared phase tables share that storage. Contiguous priming
has reserve/copy/publication; prediction additionally evaluates attention after
main publication. Main and draft counts are separate borrowed ports. Already
normalized, rotary-transformed and precision-simulated KV/query inputs are
borrowed; this frame does not replace those preceding arithmetic stages.

The enclosing request owns ordered submission, completion and cancellation.
Drain both phase queues before closing this frame. No token-step allocation,
source generation, host readback or independent completion is introduced.

Priming accepts later prompt chunks and accepted-prefix replay at the current
committed frontier, not only position zero. Prediction still publishes exactly
one main row before evaluating drafts. `persistent_state` exposes only the ring
and frontier/error pair for a startup-prepared device undo transaction; exclude
the recomputed descriptor and output. Drain the enclosing queues before undo.
