# LunaFlux tokenizer

This package owns two closed, validated BPE profiles, deterministic
encode/decode behavior, and one reusable bounded tokenization workspace. The
original profile is raw ByteLevel BPE. The selected Llama compatibility profile
is the exact SentencePiece-derived `tokenizer.json` pipeline used by the pinned
upstream model: prepend/space-to-U+2581 normalization, no pre-tokenizer, BOS
template processing, fused unknown policy with complete byte fallback, and the
ordered replace/byte-fallback/fuse/leading-strip decoder. Arbitrary normalizer,
pre-tokenizer, template, decoder, and SentencePiece variants still fail during
artifact admission.

`TokenizerSpec` constructs immutable dense token tables and a CSR merge index
sorted by left token and then right token. The ranked arrays remain the
authority for group-merge order. Request work therefore performs resumable
binary lookup without a hash-table probe or request-time index construction.

`LunaTokenizerWorker` preallocates its input-byte, token, and link storage at
startup. SentencePiece workers reserve a checked profile-specific symbol bound
covering normalization expansion, template output, and special-token segment
boundaries; the raw profile retains its smaller one-symbol-per-byte bound.
`begin_luna_input` validates and claims one declared byte count in
constant work, returning an opaque epoch-authenticated
`LunaTokenizerInputWrite`. Each `push_byte` performs one constant,
allocation-free scalar copy. `finish` transfers authority to
`LunaTokenizerWork` only after the exact declared count was written; incomplete
or over-declared writes remain abortable. Retained writer or work aliases become
stale after transfer, abort, or worker reuse. No request may begin while an
earlier writer or work credit is live.

Each `progress` call performs at most its `LunaTokenizerStepBudget`. One charged
unit is one input-symbol append, special-piece byte/candidate transition,
adjacency setup, CSR binary-search comparison/advance, merge candidate,
compaction/finalization transition, or copied token. Progress performs no
managed allocation or proportional hidden helper scan. Result copying is
ranged, capped by the same budget, and reported through `last_work_units`.
Callers transferring into an opaque owner can instead use `token_status`,
which reads exactly one ready token without exposing the worker's mutable fixed
storage. `TokenizerSpec::luna_worker_required_int_cells` reports the exact three
profile-specific preallocated integer arrays and `required_byte_cells` reports
the exact input backing before a pool performs aggregate allocation.

The merge state machine preserves canonical BPE behavior: it first selects the
best-ranked pair across the current symbol list, then merges every current
non-overlapping occurrence from left to right. Pairs created by that pass are
considered only by the next selection pass. Longest special-token matching,
lowest-ID ties, rejection, ordinary-byte treatment, overflow rejection, and
truncation metadata use the same owner.

The SentencePiece profile validates UTF-8 incrementally, applies normalization
without constructing request strings, looks up one-scalar model pieces through
a startup-sorted table, and otherwise emits the authenticated 256-entry byte
fallback alphabet. Template BOS is part of the same output bound. Decoder-side
model pieces are precomputed at startup, while byte-fallback pieces remain raw
bytes so the declared decoder order is preserved across token boundaries.

`LunaTokenizerWorker::begin_bytes` is a proportional synchronous compatibility
copy: it drives the same authoritative writer one scalar byte at a time and
retains no caller `Bytes`. It is not a reactor quantum. `TokenizerSpec::encode_bytes`
is the synchronous compatibility facade over that same worker rather than a
second encoder. `encode_text` still performs whole-string UTF-16 validation and
UTF-8 materialization for standalone callers; cooperative request preparation
uses the scalar writer to copy canonical input under its own work budget. This
package does not claim an OS-thread pool, listener, or async server.
