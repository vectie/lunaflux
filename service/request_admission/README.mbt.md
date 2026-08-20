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

Preparation publishes one opaque `LunaPreparedRequest` shell, deliberately
without `Debug` or receipt/deadline accessors. Its complete
`LunaPreparedRequestClaim` is allocated during off-reactor preparation, not
during ownership transfer. Read-only Accepted and binding metadata may be
inspected while the shell is live. `take_claim` clears the shell's sole claim
slot before returning scheduler and incremental-output authority; every
retained shell alias then rejects. Busy or draining instance disposition can
therefore leave the shell intact for a bounded mailbox retry, while a
successful take cannot be replayed into a second scheduler admission. Output
mutation exists only on the claim, never on the shell.

String stops remain in the private incremental-output owner while only stop
token IDs enter the scheduler request. Generated pieces can then be copied into
caller-owned fixed storage with split UTF-8 and cross-token stop matching.
Known stop-token IDs are exposed only as a bounded membership query and are
rejected by `push_token_into_status`, preventing their pieces from leaking as
ordinary decoded output.

The package deliberately does not own a scheduler, request handle, socket, or
async task. The incremental receiver is a trusted synchronous composition
primitive, not framed ingress or a server. The synchronous preparation boundary
is designed to run on a future bounded tokenizer worker before
`service/online_session` claims the Luna owner and binds it to the
production-owned worker. That session authenticates the exact request handle
and publication sequence, owns Accepted/Token/Usage/Completed/Failed
one-credit progression, applies stop/cancel/deadline cuts, and performs
recovery/cleanup off-reactor. Async tokenizer-pool and network-ingress
orchestration remain open. Because tokenizer encoding is synchronous CPU work,
`prepare_luna_request` must run off an async network reactor.

For a natural terminal immediately following a generated token, that session
owner holds the token publication until it observes the terminal. It then
places `finish_into_status` bytes in the typed Luna `Completed` tail,
preserving unmatched stop prefixes without inventing a text-only token event.
Outer protocol adapters may then encode that semantic view as event-v2 or
another transport representation without receiving ACK authority.
