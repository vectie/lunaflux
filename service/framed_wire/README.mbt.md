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

`IncrementalRequestReader` is a one-frame, fixed-capacity owner above the same
authoritative `RequestFrameBuffer::load`. It accepts at most the exact 16-byte
magic/version/kind/declared-length prefix first, authenticates the unsigned
declared total before exposing payload capacity, snapshots each accepted byte
range, and rejects undersize, oversize, trailing, or pipelined input. Prefix or
canonical-frame failure permanently poisons the reader; an incomplete frame can
continue, while a completed validated frame can be taken exactly once.

`CanonicalEventWriter` owns one explicit outbound credit. Its direct Token and
Completed paths validate caller-owned UTF-8 byte ranges and write the same
canonical event-v2 frame without constructing a payload `String`, `Bytes`, or
`StreamEvent`. Direct Accepted, Usage, and Failed writers consume bounded
scalar/identity evidence. The pinned frame must be copied and retired before
another event can replace it.

This package deliberately has no async, filesystem, socket, native-FFI, clock,
or engine dependency. `service/request_admission` owns the trusted receipt
clock ordering above the incremental reader, and `service/online_session` is
the current in-process composer of frames with admission and execution.
Listener and network-protocol adapters remain future outer work.
