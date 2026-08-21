# LunaFlux tokenizer

This package owns the validated byte-level BPE vocabulary, ranked merge
semantics, deterministic encode/decode behavior, and reusable bounded
tokenization workspace. It deliberately supports only the selected
`tokenizer.json` subset; unsupported normalization, pre-tokenization, template,
unknown-token, and byte-fallback behavior fails during artifact admission.

`TokenizerSpec` constructs immutable dense token tables and a CSR merge index
sorted by left token and then right token. The ranked arrays remain the
authority for group-merge order. Request work therefore performs resumable
binary lookup without a hash-table probe or request-time index construction.

`LunaTokenizerWorker` preallocates its input-byte, token, and link storage at
startup. `begin_luna_input` validates and claims one declared byte count in
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
storage. `required_int_cells` reports the exact three preallocated integer
arrays and `required_byte_cells` reports the exact input backing before a pool
performs aggregate allocation.

The merge state machine preserves canonical BPE behavior: it first selects the
best-ranked pair across the current symbol list, then merges every current
non-overlapping occurrence from left to right. Pairs created by that pass are
considered only by the next selection pass. Longest special-token matching,
lowest-ID ties, rejection, ordinary-byte treatment, overflow rejection, and
truncation metadata use the same owner.

`LunaTokenizerWorker::begin_bytes` is a proportional synchronous compatibility
copy: it drives the same authoritative writer one scalar byte at a time and
retains no caller `Bytes`. It is not a reactor quantum. `TokenizerSpec::encode_bytes`
is the synchronous compatibility facade over that same worker rather than a
second encoder. `encode_text` still performs whole-string UTF-16 validation and
UTF-8 materialization for standalone callers; cooperative request preparation
uses the scalar writer to copy canonical input under its own work budget. This
package does not claim an OS-thread pool, listener, or async server.
