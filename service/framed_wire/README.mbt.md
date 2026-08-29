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

`LunaFramedEventWorkspace` is the authoritative cooperative boundary from an
epoch-bound `LunaEventView` to event-v2 bytes. Construction preallocates the
exact 208-byte header plus the larger of the configured decoded-delta envelope
and fixed 64-byte public failure code, together with one semantic-view reference
slot. The checked `required_byte_cells` and `required_reference_cells` report
those same startup requirements. `begin` issues one epoch-authenticated Work.
Each `progress` call performs at most its configured budget of bounded steps:
one fixed scalar/header group, one header-clear byte, one payload/digest byte,
one checksum byte, or one phase transition. `last_work_units` and
`total_work_units` report exact charged steps. A semantic failure pins an
authenticated Failed Work until `abort`.

The final checksum transition reauthenticates the semantic event before
publishing Ready, then detaches it. `take_view` transfers the immutable frame to
one opaque `LunaFramedEventView`. `copy_chunk_to` copies exactly one positive
caller-selected range no larger than the configured step budget; it has no
transport cursor and cannot claim that bytes were written. The TCP or other
outer transport must retain its own confirmed-write cursor across partial
writes. `release` returns only framed-byte authority and never acknowledges the
semantic event.

`LunaFramedEventAdapter` is the synchronous compatibility facade over that same
Workspace/Work/View engine. It retains no second frame buffer, drives Work to
Ready proportionally, and chunks a full compatibility copy through View. Busy
and stale rejection leave its sole transport credit unchanged. The cooperative
Workspace APIs, not `stage` or full `copy_to`, are the bounded reactor-facing
surface. Semantic retirement remains exclusively with `LunaEventOwner`, so
framed, SSE, and OpenAI-compatible adapters can share the same semantic source
without parsing another adapter's bytes or acquiring ACK authority.

This package deliberately has no async, filesystem, socket, native-FFI, clock,
or engine dependency. `service/request_admission` owns the trusted receipt
clock ordering above the incremental reader. `service/online_session`
publishes semantic Luna event credits; outer adapters compose transport
representations from those credits. The online TCP package now supplies the
one-shot, reusable native, pipelined native, and serialized OpenAI listener
owners; TLS and concurrent-client arbitration remain future outer work.

`LunaFramedTextRequestWorkspace` is the canonical object-free v1 text-request
encoder. Startup copies one exact `ModelIdentity` and one explicit nonempty
`CacheScope`; cache permission is always Disabled and trace is absent.
`begin_text` preserves the compatibility object surface with empty stops.
`begin_text_with_sampling_scalars` accepts already-validated scalar sampling,
seed, and declared stop counts without constructing a request object. Its
generation-bound Write receives the declared UTF-8 input, stop token IDs, and
length-prefixed stop-string bytes through scalar methods. Incomplete or
out-of-envelope stop payload cannot become Work.

Work emits one header byte, cache-scope byte, checksum input byte, checksum
byte, or phase transition per charged unit. Its View permits only nonzero
copies bounded by the configured step budget. The result is byte-identical to
`RequestFrameBuffer::encode` for the same semantic request and carries no
absolute receipt timestamp. Original receipt ownership therefore remains with
request admission rather than being rebased here.
