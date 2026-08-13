# Online request admission

This package is the synchronous, transport-neutral bridge from one canonical
`GenerateRequest` to the scheduler's token-only contract. A `RequestReceipt`
captures monotonic time before frame parsing. Admission binds the exact selected
model and tokenizer, tokenizes text with special-token rejection and overflow
rejection, samples the clock again after tokenization, and preserves the
original absolute deadline rather than rebasing it.

String stops remain in the private incremental-output owner while only stop
token IDs enter the scheduler request. Generated pieces can then be copied into
caller-owned fixed storage with split UTF-8 and cross-token stop matching.
Known stop-token IDs are exposed only as a bounded membership query and are
rejected by `push_token_into`, preventing their pieces from leaking as ordinary
decoded output.

The package deliberately does not own a scheduler, request handle, socket, or
async task. A future session owner must bind the returned request to the exact
scheduler `RequestHandle`, consume globally ordered publications, reserve
outbound capacity before stepping, and cancel on disconnect. Because tokenizer
encoding is synchronous CPU work, `admit` must run on a bounded tokenizer
worker rather than an async network reactor.

For a natural terminal immediately following a generated token, that session
owner must hold the token publication until it observes the terminal. It can
then append `finish_into` bytes to the same token delta before publication,
preserving unmatched stop prefixes without inventing a text-only event.
