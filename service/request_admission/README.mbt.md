# Online request admission

This package is the synchronous, transport-neutral bridge from one canonical
framed `GenerateRequest` to the scheduler's opaque Luna stop-token contract. The opaque
`IncrementalRequestReceiver` preallocates its framed reader and scalar receipt
state. Empty or invalid caller ranges read no clock; the first valid nonempty
append samples monotonic time exactly once before any byte/count mutation, and
later fragments cannot resample it. Frame poison closes the receiver, while an
incomplete take remains appendable and a successful take publishes one opaque
`ReceivedRequest`. The whole-frame compatibility `receive` preserves the same
clock-before-authoritative-load ordering on its caller-owned buffer.

Binding copies the immutable request out of the exact validated frame and
derives its absolute admission deadline once; neither the frame epoch, receipt
time, deadline scalar, nor a detachable receipt capability is public.
`prepare_luna_request` binds the exact selected model and an independently
expected tokenizer digest, tokenizes text with special-token and overflow
rejection, samples the clock again after tokenization, and preserves that
retained deadline without rebasing it.

The live cooperative path is `LunaRequestPreparationPool`. It allocates a hard
maximum of 1024 fixed lanes at startup, after checked aggregate `Int`, `Byte`,
and reference-cell accounting. Every lane permanently owns one Luna framed
scanner workspace, tokenizer worker, token buffer storage, request-semantic
storage, and incremental-output workspace. The mandatory `framed_limits`
binding must carry the exact pool inference limits. Scanner frame bytes and
stop-table integer cells, plus its preallocated Work/View authority slots, are
included in the pool envelope. `try_submit`
returns Saturated or Draining without consuming its `ReceivedRequest`; an
admitted request is pinned to its lane and exact generation until explicit
discard or claim release. Once a free lane is selected, `try_submit` consumes
the received authority before fallible lane start; a rejected start resets and
releases that lane but intentionally does not restore the invalid submission's
receipt capability.

Only the pool advances work. Its fixed active ring grants one FIFO lane a
configured quantum per `progress` call. One charged preparation unit is one
received frame byte, one framed validation unit, one model-digest byte
comparison, one input-byte copy into the fixed tokenizer backing, one tokenizer
state-machine unit, one token scalar read/write, one semantic
token/UTF-8/cache import or validation unit, one incremental-output setup unit,
or one constant-size assembly transition. Work is checked against both a per-call
quantum and an exact total-work ceiling. The monotonic deadline is checked
before each lane quantum, immediately before Ready publication, and again
before Prepared and Claimed transfer. Cancellation is observed by the central
owner; Ready and Failed results remain pinned rather than being silently
evicted.

`try_begin_luna_framed` is the direct transport-neutral path. Saturated and
Draining dispositions consume zero bytes and sample no clock. An admitted
nonempty range captures its receipt timestamp before byte one, reports the
exact consumed prefix through `consumed_bytes`, and exposes further bounded
offers only on the same generation-authenticated preparation Work. Incomplete
receipts use the overflow-checked hard deadline `receipt +
inference.max_deadline_millis`; once the validated View exposes the client
budget, its exact deadline is derived from that same original receipt and is
never rebased. `luna_framed_receipt_complete` authenticates that exact Work and
becomes true immediately after the scanner consumes the declared final frame
byte, before canonical validation or later import must finish. It remains true
through Ready, returns false for live object-form submissions, and rejects
stale, Failed, CancelRequested, or already-transferred Work authority.
Incomplete receipts remain in the FIFO ring at zero charged work so deadlines,
cancellation, and drain remain observable. After deadline and drain precedence,
`progress` reports `LunaRequestPreparationPoolAwaitingInput` only when the
authenticated dequeued direct lane still needs bytes; scanner validation and
object-form work remain `Advanced`. An authenticated incomplete
direct-frame Work can report only its remaining
hard-receipt interval, never its receipt or absolute deadline. Zero means a
transport must not begin another blocking operation; completed, object-form,
and stale Work reject.

Compatibility protocols that must parse or render before canonical framed
bytes exist use `LunaRequestReceiptWorkspace`. It captures the OS monotonic
receipt before the first transport byte and returns a generation-bound
`LunaRequestReceipt` exposing only remaining time and abort. It has no timestamp
or budget getter. `try_begin_luna_framed_with_receipt` atomically consumes the
receipt only on admission; draining, saturation, invalid source ranges,
cross-envelope budgets, clock failure, and expiry leave it live. The later
frame must carry exactly the workspace's fixed relative DeadlineBudget, so
parse, render, encode, and preparation all consume the original interval.

Drain promptly retires incomplete receipts; a fully received scanner or later
import may finish. The validated
framed View stays private to its lane while expected content and plan digests
are compared bytewise, then while text/token input and stop/cache semantics are
copied into their fixed owners. It is released before tokenization or semantic
validation continues.

Ready assembly may allocate only constant-size scheduler, prepared, and claim
records. Text bytes remain canonical and immutable while BPE and token copying
are cooperative. TokenIds reuse their already-validated `TokenBuffer` without
rescanning or copying. No proportional collection is created by pool progress.
String stops and cache authority remain in the inference-owned semantic lease.
The full view exists only through output setup; the scheduler receives the
narrow O(1) Luna stop-token projection plus an independently opaque prefix
projection exposing only validated cache permission/scope bytes. That cache
projection is bound to the exact tokenizer digest when the scheduler request
is assembled; it exposes no strings, limits, storage, epoch, or release power.

Preparation publishes one opaque `LunaPreparedRequest` shell, deliberately
without `Debug`, lane, generation, receipt, or deadline accessors. `take_claim`
revokes every retained shell alias and transfers scheduler plus optional pooled
token/output lease authority. `LunaPreparedRequestClaim::release` is mandatory,
generation-authenticated, and non-idempotent. Semantic release occurs first,
so any live scheduler or online-worker retention rejects early release
transactionally and leaves the claim and lane intact. Successful release after
lower retirement returns the lane to the free ring. Output mutation exists only on the
claim, never on the shell. Its `scheduler_request` result is a trusted borrowed
view for the designated online admission bridge: it must not outlive the claim,
and no other production package may retain or consume it.

String stops remain in private semantic/output owners while only opaque
stop-token membership and cache-identity projections enter the scheduler
request. Generated pieces can then be copied into
caller-owned fixed storage with split UTF-8 and cross-token stop matching.
Known stop-token IDs are exposed only as a bounded membership query and are
rejected by `push_token_into_status`, preventing their pieces from leaking as
ordinary decoded output.

The legacy `prepare_luna_request` facade remains synchronous for compatibility,
but drives the same semantic validation, output setup, and claim engine. Its
exact-shape Luna output factory preserves small-request memory behavior. It is
not the reactor-safe production path. The package does not
own a scheduler, request handle, socket, or async task. The legacy object-form
frame materialization path remains only for compatibility; the direct Luna
framed path does not construct `GenerateRequest`, `Input`, `TextInput`,
`StopConditions`, `CachePolicy`, or a standalone token buffer. This package
still owns no socket or listener; `service/online_session` and
`service/online_tcp` now compose its trusted receipt into one-shot, reusable,
pipeline, and serialized OpenAI ingress. Concurrent-client arbitration remains
outside that composition. `service/online_session` claims the Luna
owner, authenticates its scheduler handle and publication sequence, and must
release the claim on every rejected-admission or terminal path.

For a natural terminal immediately following a generated token, that session
owner holds the token publication until it observes the terminal. It then
places `finish_into_status` bytes in the typed Luna `Completed` tail,
preserving unmatched stop prefixes without inventing a text-only token event.
Outer protocol adapters may then encode that semantic view as event-v2 or
another transport representation without receiving ACK authority.
