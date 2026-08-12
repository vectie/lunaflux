# Phase 1 reference artifact admission

`model/artifact` admits the exact bytes used by LunaFlux's offline Phase 1
correctness path. The caller supplies three distinct paths and independently
approved SHA-256 identities for `config.json`, `tokenizer.json`, and one
`safetensors` file. The package never scans a directory, executes metadata, or
infers additional files.

`load_reference_bundle` is intentionally a whole-file host snapshot. It checks
each non-symlink regular file's size against per-file and aggregate limits
before allocating, reads from one open file handle, verifies the canonical
SHA-256 of the returned bytes, and parses those same immutable bytes. A same-size
replacement or mutation therefore fails the expected digest instead of
reaching a parser. Paths are locators, not trust identities; the approved
digest is authoritative.

MoonBit's current async filesystem API does not expose an atomic
`openat`/`O_NOFOLLOW` operation. Admission therefore performs a non-following
kind check before open and a regular-file check on the opened handle. This is
appropriate for the product contract's approved read-only model mount. If a
writable or adversarial directory is ever admitted, production must first add
a narrow native `openat` plus `fstat` wrapper rather than treating these two
checks as atomic.

The validated weights digest becomes `ModelIdentity.content`, while the model
plan digest is derived from validated configuration semantics. The tokenizer
digest and configuration-file digest remain separately available through
`ArtifactIdentity`. Public errors contain only bounded categories and never
retain paths or file contents.

This package is not the production weight loader. Production loading must use
streaming or mapping and place tensor bytes directly into their final device
allocations through the materialization boundary. It must not retain a complete
model-sized host copy.
