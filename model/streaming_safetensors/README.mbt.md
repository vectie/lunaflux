# Streaming safetensors startup boundary

This family-neutral native package parses and authenticates sharded
safetensors files beneath a caller-owned `approved_fs` root without ever
forming a whole-file `Bytes`. Header allocation is independently bounded, file
and payload positions remain 64-bit, and complete-file SHA-256 authentication
is incremental. Every opened file is deterministically closed on success and
failure.

Callers register exact BF16, F32, or opaque fixed-width dtype spellings. The
package validates unique names across and within shards, positive bounded
shapes, exact dtype-derived byte counts, payload-relative ranges, and
non-overlap. It never interprets tensor values.

`copy_ranges` reopens and completely reauthenticates all inspected shards,
then reads validated source ranges directly into caller-supplied final
`FixedArray` spans. Its visitor receives only tensor metadata and copied-range
coordinates; neither an approved file nor filesystem authority appears in the
callback type. Destination bytes must be discarded if the operation raises.

The package imports no model family, device, kernel, scheduler, KV, or API
package. Family adapters should translate their already-closed dtype and
semantic vocabularies into dtype specifications and copy requests without
moving family branching into this package.
