# Committed-main/draft execution frame

One owner retains the main BF16 ring, frontier/error descriptor and draft
output. Both startup-prepared phase tables share that storage. Initial priming
has reserve/copy/publication; prediction additionally evaluates attention after
main publication. Main and draft counts are separate borrowed ports. Already
normalized, rotary-transformed and precision-simulated KV/query inputs are
borrowed; this frame does not replace those preceding arithmetic stages.

The enclosing request owns ordered submission, completion and cancellation.
Drain both phase queues before closing this frame. No token-step allocation,
source generation, host readback or independent completion is introduced.
