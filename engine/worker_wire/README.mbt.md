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

Completion frame v1 carries the exact plan sequence and model generation plus
a canonical table of slot, request identity/generation, outcome kind,
processed-token count, token ID, and bounded worker-failure category. The
receiver builds a fixed slot-to-entry index after validation, so exact-slot
lookup is O(1) without a map or collection growth.

The parent authenticates every received completion frame against the retained
exact `SubmittedSchedulePlan` before acquiring the scheduler's paired
completion writer. Sequence, model generation, row count, slot, request
identity/generation, row kind, and processed prompt length must all match.
The converted `SubmittedCompletion` remains retryable if scheduler output or
terminal publication is backpressured; frame acceptance itself does not retire
the plan or mutate request/KV state.

The isolated side does not need a scheduler completion owner. An exclusive
`CompletionFrameWriter` is bound to one exact validated plan-frame owner and
epoch. It copies canonical request, generation, slot, and prompt-length fields
from those authenticated rows, accepts only bounded outcome scalars, enforces
prefill-before-decode row order, and either submits a complete checksummed
frame or aborts to a new stale epoch. Partial, foreign-owner, duplicate-writer,
and terminal-epoch paths fail without publishing a frame.

Untrusted-frame request and slot uniqueness checks use startup-owned scratch
and deterministic in-place heapsort. They are O(n log n), avoiding quadratic
work at the configured maximum row count while retaining allocation-free
steady-state validation.

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

This package is transport metadata only. Process lifecycle and I/O, endpoint
authentication, timeouts, worker-death recovery, and device execution from
received plans are separate slices and are not claimed here.
