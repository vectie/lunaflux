# Streaming safetensors startup boundary

This family-neutral native package parses bounded headers of sharded
safetensors files beneath a caller-owned `approved_fs` root without ever
forming a whole-file `Bytes`. Header allocation is independently bounded, file
and payload positions remain 64-bit. There is no full-file SHA-256 scan during
inspection, copy, or upload. Every opened file is deterministically closed on success and
failure.

Callers register exact BF16, F32, or opaque fixed-width dtype spellings. The
package validates unique names across and within shards, positive bounded
shapes (including rank-zero scalars), exact dtype-derived byte counts, payload-relative ranges, and
non-overlap. It never interprets tensor values.

Header range validation sorts an ordinal view and checks adjacent intervals in
O(n log n), rather than comparing every tensor pair. Published tensor order and
error coordinates remain in original header order. This supports auxiliary
shards containing tens of thousands of scalar scales without quadratic startup
work; caller-owned count, byte and total-population limits still apply.

`copy_ranges` reopens only shards referenced by copy requests,
then reads validated source ranges directly into caller-supplied final
`FixedArray` spans. Its visitor receives only tensor metadata and copied-range
coordinates; neither an approved file nor filesystem authority appears in the
callback type. Destination bytes must be discarded if the operation raises.

`transfer_ranges` streams tensor slices through a reusable caller-owned scratch
span into opaque destinations such as device allocations. Slice lengths and
destination offsets stay 64-bit; only the current chunk is managed-array sized.
The synchronous consumer must finish reading each chunk before returning. Packed
quantized payloads are copied unchanged. A referenced shard is reopened once
per transfer call; scoped sessions reuse opened handles. No complete tensor
or host weight arena is created by this operation.

Inventory digests are declared labels only, not proof of payload integrity.
Transfer-integrity checks, when needed, belong to acquisition tooling. Header
geometry, destination bounds, same-file metadata checks, and explicit close
remain execution-correctness checks; no filesystem or crypto work enters generation.

The package imports no model family, device, kernel, scheduler, KV, or API
package. Family adapters should translate their already-closed dtype and
semantic vocabularies into dtype specifications and copy requests without
moving family branching into this package.
