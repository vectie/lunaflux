# Online request admission

This package is the synchronous, transport-neutral bridge from one canonical
framed `GenerateRequest` to the scheduler's token-only contract. The opaque
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
and reference-cell accounting. Every lane permanently owns one Luna tokenizer
worker, token buffer storage, and incremental-output workspace. The byte budget
includes both the worker's exact fixed input backing and the output workspace's
matcher/decode backing for every lane. `try_submit`
returns Saturated or Draining without consuming its `ReceivedRequest`; an
admitted request is pinned to its lane and exact generation until explicit
discard or claim release. Once a free lane is selected, `try_submit` consumes
the received authority before fallible lane start; a rejected start resets and
releases that lane but intentionally does not restore the invalid submission's
receipt capability.

Only the pool advances work. Its fixed active ring grants one FIFO lane a
configured quantum per `progress` call. One charged preparation unit is one
input-byte copy into the fixed tokenizer backing, one tokenizer state-machine
unit, one token scalar read/write, one incremental-output setup unit, or one
constant-size assembly transition. Work is checked against both a per-call
quantum and an exact total-work ceiling. The monotonic deadline is checked
before each lane quantum, immediately before Ready publication, and again
before Prepared and Claimed transfer. Cancellation is observed by the central
owner; Ready and Failed results remain pinned rather than being silently
evicted.

Ready assembly may allocate only constant-size scheduler, prepared, and claim
records. Text bytes remain canonical and immutable while BPE and token copying
are cooperative. TokenIds reuse their already-validated `TokenBuffer` without
rescanning or copying. No proportional collection is created by pool progress.
String stops remain in the pooled incremental-output lease while the scheduler
receives the O(1) token-only stop view.

Preparation publishes one opaque `LunaPreparedRequest` shell, deliberately
without `Debug`, lane, generation, receipt, or deadline accessors. `take_claim`
revokes every retained shell alias and transfers scheduler plus optional pooled
token/output lease authority. `LunaPreparedRequestClaim::release` is mandatory,
generation-authenticated, and non-idempotent; only successful subordinate lease
release returns the lane to the free ring. Output mutation exists only on the
claim, never on the shell. Its `scheduler_request` result is a trusted borrowed
view for the designated online admission bridge: it must not outlive the claim,
and no other production package may retain or consume it.

String stops remain in the private incremental-output owner while only stop
token IDs enter the scheduler request. Generated pieces can then be copied into
caller-owned fixed storage with split UTF-8 and cross-token stop matching.
Known stop-token IDs are exposed only as a bounded membership query and are
rejected by `push_token_into_status`, preventing their pieces from leaking as
ordinary decoded output.

The legacy `prepare_luna_request` facade remains synchronous and detached for
compatibility. It is not the reactor-safe production path. The package does not
own a scheduler, request handle, socket, or async task, and framed parsing plus
request materialization are still synchronous. Therefore end-to-end
reactor-safe ingress is not yet claimed even though Luna preparation itself is
fixed-lane and cooperatively bounded. `service/online_session` claims the Luna
owner, authenticates its scheduler handle and publication sequence, and must
release the claim on every rejected-admission or terminal path.

For a natural terminal immediately following a generated token, that session
owner holds the token publication until it observes the terminal. It then
places `finish_into_status` bytes in the typed Luna `Completed` tail,
preserving unmatched stop prefixes without inventing a text-only token event.
Outer protocol adapters may then encode that semantic view as event-v2 or
another transport representation without receiving ACK authority.
