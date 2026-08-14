# Online request admission

This package is the synchronous, transport-neutral bridge from one canonical
framed `GenerateRequest` to the scheduler's token-only contract. `receive`
captures monotonic time before frame parsing and returns one opaque, single-use
owner that keeps both facts inseparable. Admission binds the exact selected
model and an independently expected tokenizer digest, tokenizes text with
special-token rejection and overflow
rejection, samples the clock again after tokenization, and preserves the
original absolute deadline rather than rebasing it.

String stops remain in the private incremental-output owner while only stop
token IDs enter the scheduler request. Generated pieces can then be copied into
caller-owned fixed storage with split UTF-8 and cross-token stop matching.
Known stop-token IDs are exposed only as a bounded membership query and are
rejected by `push_token_into`, preventing their pieces from leaking as ordinary
decoded output.

The package deliberately does not own a scheduler, request handle, socket, or
async task. `service/online_session` now consumes this bridge internally and
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
