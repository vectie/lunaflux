# Worker wire protocol

This package owns bounded, canonical frames for crossing the isolated-worker
process boundary. It does not expose MoonBit heap-owner capabilities such as
`SubmittedSchedulePlan`; the parent encodes those capabilities into fixed byte
storage, and the worker validates an untrusted copy before reading scalar
tables and row descriptors.

The first format is plan frame v1. Its 64-byte header carries an exact length,
plan sequence, model-plan generation, token budget, table counts, and an FNV-1a
checksum. Tokens, full generational page identities, ordered capability IDs,
prefill rows, and decode rows follow in canonical contiguous order. Integer and
floating-point fields use little-endian fixed-width representations. Sampling
seed and output index are preserved exactly.

`PlanFrameBuffer` allocates its worst-case byte capacity once from validated
worker limits. Successful encode, load, copy, and scalar access reuse that
storage and do not grow collections. Frame views and row views authenticate an
owner epoch on every access. A rejected load leaves the prior frame and epoch
unchanged. Error values may allocate on rejected paths.

The checksum detects accidental corruption only. It is not a MAC and provides
no peer authentication. The eventual process transport must authenticate the
worker endpoint and enforce its own framing and resource limits. The receiver
still validates every count, range, identity, token, page generation,
capability, sampling field, request uniqueness, completion slot, and canonical
table cursor after checksum verification.

This package is transport metadata only. Completion frames, process lifecycle,
timeouts, worker-death recovery, and device execution from received plans are
separate slices and are not claimed here.
