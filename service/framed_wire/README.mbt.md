# Canonical framed service wire

`service/framed_wire` owns the bounded binary request-v1 and event-v2
representations of the native service contracts. It is a codec package, not a
listener, transport, scheduler, tokenizer, or execution service.

The request frame carries one complete `GenerateRequest`: protocol and request
identity, model content and plan digests, text or token input, generation and
context ceilings, sampling and deterministic seed, stop tables, streaming
preference, relative deadline budget, cache scope/permission, and optional
trace correlation. The event frame carries exactly one of `Accepted`, `Token`,
`Usage`, `Completed`, or `Failed`. Event v2 adds an optional bounded UTF-8 tail
to `Completed`, allowing unmatched incremental stop-prefix bytes to be flushed
in the terminal frame without inventing a text-only token. Event v1 is
rejected.

Both formats use fixed little-endian headers, a versioned magic/kind tuple,
an exact total length, an FNV-1a checksum that excludes its own field, zeroed
reserved bytes, and one canonical payload order. Optional values have explicit
presence flags; unused scalar fields must be zero. Digests remain lowercase
SHA-256 identities and all strings are reconstructed through their owning
bounded contract constructors.

`FramedWireLimits` combines caller-supplied inference limits with a frame-byte
ceiling capped at 16 MiB. Decoders check collection counts and individual
payload lengths against those limits before multiplication, allocation, UTF-8
decoding, or contract construction. Errors expose only frame kind and rule;
they never contain request text, stop strings, cache scopes, traces, decoded
token text, public failure codes, paths, or raw bytes.

`RequestFrameBuffer` and `EventFrameBuffer` own fixed-capacity transport and
scratch storage. Encoding accepts only already-admitted contract values and
writes the single canonical representation. Loading validates completely
before it replaces the prior frame, so rejection is transactional. Validated
views are epoch-bound and become stale after the owner publishes a replacement.

`LunaFramedRequestWorkspace` is the transport-neutral cooperative request-v1
scanner. `begin` issues an epoch-authenticated `LunaFramedRequestWork`; `offer`
copies no more than the configured step budget and returns the exact accepted
byte count, while `progress` performs no more than that many bounded validation
steps. One validation step is a fixed header group, one payload byte or token,
one duplicate comparison, or one phase transition; it is not a claim that a
step contains only one scalar read. A ready work item transfers a single
`LunaFramedRequestView`. Its scalar and indexed-byte accessors read directly
from fixed-capacity workspace storage and become stale after `release` and
workspace reuse. No accessor constructs `GenerateRequest`, `Input`, stop
arrays, strings, digests, or optional scalar wrappers.

`required_byte_cells` reports the exact canonical frame backing and
`required_int_cells` reports the combined stop-string offset and length tables.
Both use the same checked capacity calculation as workspace construction, so a
later fixed-lane aggregate can authenticate total startup storage before any
lane allocation.

`RequestFrameBuffer::load` remains the allocating object-form compatibility
surface. It synchronously drives that same scanner, proportionally
materializes `GenerateRequest`, and publishes only after both operations
succeed; there is no second request validator. The buffer binds the scanner to
its existing scratch array, so it retains two frame-sized arrays in total plus
bounded stop-string offset and length tables. Future ingress can drive a
standalone workspace over multiple reactor turns without using the
compatibility materializer.

`IncrementalRequestReader` is a one-frame, fixed-capacity compatibility owner
above `RequestFrameBuffer::load`. It accepts at most the exact 16-byte
magic/version/kind/declared-length prefix first, authenticates the unsigned
declared total before exposing payload capacity, snapshots each accepted byte
range, and rejects undersize, oversize, trailing, or pipelined input. Prefix or
canonical-frame failure permanently poisons the reader; an incomplete frame can
continue, while a completed validated frame can be taken exactly once.

`LunaFramedEventAdapter` is the canonical boundary from an epoch-bound
`LunaEventView` to event-v2 bytes. It writes directly into unpublished,
preallocated frame storage and exposes the frame only after all view evidence
has been authenticated. Construction requires capacity for the larger of the
configured decoded-delta envelope and the fixed 64-byte public failure code,
so every valid semantic event is frameable before request work begins. Busy,
stale-view, and defensive capacity rejection leave its single transport credit
unchanged. `release` returns only that transport credit; semantic event
retirement remains exclusively with `LunaEventOwner`, so framed, SSE, and
OpenAI-compatible adapters can share the same semantic source without parsing
another adapter's bytes or acquiring ACK authority.

This package deliberately has no async, filesystem, socket, native-FFI, clock,
or engine dependency. `service/request_admission` owns the trusted receipt
clock ordering above the incremental reader. `service/online_session`
publishes semantic Luna event credits; outer adapters compose transport
representations from those credits. Listener and network-protocol adapters
remain future outer work.
