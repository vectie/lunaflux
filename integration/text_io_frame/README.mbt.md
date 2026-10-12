# Prepared text frame

`TextIoFrame` owns normalized/logit buffers and four AOT functions. Its borrowed
prefix and suffix launch views surround any compatible decoder frame in one
caller-owned queue, with one completion per step. Preparation binds all pointers
and checks workspace capacity; warmed launch execution allocates no heap memory.

Token IDs, input counts, selected row IDs/output count, residual input/output,
sampled token output and weights belong to the caller. Distinct residual input
and output allocations must outlive the borrowing queue. Validate request ports
before submission and reject sampled `-1`/decoder error ports before publishing.
Close/abort the queue before closing this frame or releasing borrowed buffers.
Partial preparation is releasable; repeated close is harmless.

`LearnedTextIoFrame` adds a normalized F32 learned-control reduction before
the same final normalization/head/greedy consumer. It owns the collapsed BF16
rows, checks all borrowed spans before allocation, and exposes four prepared
suffix launches. Controls and multi-stream residual input remain caller-owned;
the vocabulary consumer receives exactly one stream. It has no embedding
prefix, model-family branching or per-step compilation. Invalid live controls
produce NaN reduction rows and rejected (`-1`) samples, never stale values.

Both suffixes now consume effect-derived execution order and typed numerical
programs. Their physical launches and arithmetic are unchanged. `projection_suffix`
omits greedy sampling for teacher-forced evaluation. `replay_buffers` borrows
normalized/logit buffers and caller-supplied input counts, output count and selected
local rows; the learned frame additionally exposes collapsed rows. Calling it
does not synchronize, copy data, evaluate a metric or perform hashing. The offline
caller must retire the producing queue, validate live counts and selection, map
local rows to absolute token positions, and capture before reuse or closure.
Fresh, failed, closed and ingress/interior-only frames reject these egress ports.
