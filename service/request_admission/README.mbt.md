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
time, deadline scalar, nor a detachable receipt capability is public. Admission
binds the exact selected model and an independently expected tokenizer digest,
tokenizes text with special-token and overflow rejection, samples the clock
again after tokenization, and preserves that retained deadline without
rebasing it.

String stops remain in the private incremental-output owner while only stop
token IDs enter the scheduler request. Generated pieces can then be copied into
caller-owned fixed storage with split UTF-8 and cross-token stop matching.
Known stop-token IDs are exposed only as a bounded membership query and are
rejected by `push_token_into`, preventing their pieces from leaking as ordinary
decoded output.

The package deliberately does not own a scheduler, request handle, socket, or
async task. The incremental receiver is a trusted synchronous composition
primitive, not framed ingress or a server. `service/online_session` consumes
this bridge internally and
binds it to the production-owned worker without exposing the intermediate
admitted owner. That session authenticates the exact request handle and
publication sequence, owns Accepted/Token/Usage/Completed/Failed one-credit
progression, applies stop/cancel/deadline cuts, and performs recovery/cleanup
off-reactor. Async tokenizer-pool and network-ingress orchestration remain
open. Because tokenizer encoding is synchronous CPU work, `admit` and the
aggregate preparation must run off an async network reactor.

For a natural terminal immediately following a generated token, that session
owner holds the token publication until it observes the terminal. It then
places `finish_into` bytes in the canonical event-v2 `Completed` tail,
preserving unmatched stop prefixes without inventing a text-only token event.
