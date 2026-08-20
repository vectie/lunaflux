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

`LunaTokenizerWorker` preallocates its token and link storage at startup.
`begin_bytes` validates and claims one input in constant work, returning an
opaque epoch-authenticated `LunaTokenizerWork`. Retained work aliases become
stale after `abort` or worker reuse. No request may begin while an earlier work
credit is live.

Each `progress` call performs at most its `LunaTokenizerStepBudget`. One charged
unit is one input-symbol append, special-piece byte/candidate transition,
adjacency setup, CSR binary-search comparison/advance, merge candidate,
compaction/finalization transition, or copied token. Progress performs no
managed allocation or proportional hidden helper scan. Result copying is
ranged, capped by the same budget, and reported through `last_work_units`.
Callers transferring into an opaque owner can instead use `token_status`,
which reads exactly one ready token without exposing the worker's mutable fixed
storage. `required_int_cells` reports the exact three preallocated integer
arrays before a pool performs aggregate allocation.

The merge state machine preserves canonical BPE behavior: it first selects the
best-ranked pair across the current symbol list, then merges every current
non-overlapping occurrence from left to right. Pairs created by that pass are
considered only by the next selection pass. Longest special-token matching,
lowest-ID ties, rejection, ordinary-byte treatment, overflow rejection, and
truncation metadata use the same owner.

`TokenizerSpec::encode_bytes` is a synchronous compatibility facade over this
authoritative worker rather than a second encoder. `encode_text` still performs
whole-string UTF-16 validation and UTF-8 materialization for standalone callers;
request admission can use the canonical immutable bytes retained by
`TextInput`. A future request-admission slice must compose those bytes with a
fixed-capacity cooperative tokenizer pool before network ingress; this package
does not claim an OS-thread pool, listener, or async server.
