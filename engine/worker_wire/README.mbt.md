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

Before plan traffic, startup frame v4 performs an exact `Configure`/`Ready`
exchange. Its canonical 408-byte body binds protocol version, full model
identity, lowercase SHA-256 identities of the admitted worker bootstrap and
the independently admitted bootstrap source, the exact ordinal within the
worker process's visible device set, model-plan generation, predecessor
sequence, worker limits, and inference limits under the same bounded checksum
and reserved-field rules. A
worker is not protocol-ready until it returns the identical validated
contract. This proves configuration agreement, not model loading or CUDA
readiness; a production device worker must establish those facts before it
emits `Ready`.

Bootstrap-source v1 is a separate inert transport value for the production
device-worker handshake. It owns five strict filesystem names below two
lexically validated absolute root locators, independent SHA-256 identities for
model configuration and the full-graph execution manifest, exact device target
and KV page geometry, and focused file/arena ceilings. The canonical little-endian
body has a fixed 184-byte header, variable UTF-8 path section bounded to five
4096-byte fields, and a trailing raw SHA-256 of every preceding byte. Its
maximum is 20,696 bytes. Absolute artifact locators, traversal, aliases,
backslashes, empty path segments, reserved flags, noncanonical lengths, and
unbounded ceilings fail closed. Lexical admission and decode do not prove a root
exists, is trusted, or is stable and read-only. A deployment boundary must
independently supply that authority before any filesystem use. Model and kernel
roots may intentionally be the same deployment root; their role-specific
locators and digests remain independently encoded.

`EncodedBootstrapSource` is an immutable owned snapshot: it exposes only its
digest, focused scalar/path records, exact length, and bounded copying. It never
contains model weights, model configuration bytes, manifests, CUDA modules, or
native handles. Startup v4 now binds this source digest into the independent
`Configure` contract, but the process handshake still sends only `Configure ->
Ready`; it does not send bootstrap-source bytes yet. A later slice will send
`Configure -> BootstrapSource -> Ready` and retain the exact encoded source
across replacement workers. The trailing self-SHA provides internal integrity
and canonical content identity only; it is not peer authentication. Future
source delivery must compare the received bytes with the already independent
`Configure` source digest over the private child channel.

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

The frame exposes an opaque owner identity plus scalar epoch proof for the
worker's device-execution owner. This identity grants no storage or mutation
authority; it only prevents logits from one staged frame being published as
the completion of another buffer or superseded epoch.

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
no peer authentication. The private process transport authenticates the local
child endpoint by construction and enforces its own framing and resource
limits. The receiver
still validates every count, range, identity, token, page generation,
capability, sampling field, request uniqueness, completion slot, and canonical
table cursor after checksum verification.

This package is transport metadata only. Process lifecycle and I/O are owned by
the private process and worker-supervisor packages. Worker-death recovery and
device execution from received plans remain separate slices and are not
claimed here.
