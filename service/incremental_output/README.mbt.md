# Incremental decoded output

`service/incremental_output` owns bounded generated-token decode state. A
`LunaIncrementalOutputWorkspace` allocates its exact maximum pattern, failure,
matcher, carry, and decode storage once. Each request begins one opaque
epoch-authenticated `LunaIncrementalOutputWork`; setup copies admitted stop text
into that fixed storage and builds its KMP failure cells cooperatively. One
charged setup unit is one pattern transition, UTF-16 scalar transition, emitted
UTF-8 byte, failure comparison/advance, or failure fallback. No setup call
performs a hidden proportional scan or grows a collection.

`begin_luna_request_semantics` is the direct authority-safe path from an
authenticated inference `LunaRequestSemanticView`. It first binds the exact
`InferenceLimits` without mutating the workspace, then copies one already
validated UTF-8 stop byte per charged text unit. The view remains retained and
is reauthenticated immediately before Ready publication and lease transfer;
successful transfer detaches it before token decoding. A stale view or another
setup error enters a payload-safe authenticated Failed state. The same Work
replays that error with zero work, cannot publish a lease, and remains the sole
authority that may abort the failure back to Idle.

Workspace construction also reserves one private semantic-view capability
slot. It is filled and cleared without growing a collection, so semantic setup
does not allocate a per-request option wrapper. `required_reference_cells`
reports that single reusable slot alongside the existing exact integer and byte
cell queries.

Ready work transfers exactly one `LunaIncrementalOutputLease`. Retained work
aliases reject after transfer, while retained lease aliases reject after
release or workspace reuse. The lease alone may decode, query lifecycle, flush
the final carry, and release the workspace; it never exposes the raw mutable
output engine. The legacy synchronous `IncrementalOutput::new` allocates one
workspace and drives this same authoritative setup and lease engine rather than
maintaining parallel matching semantics.

The active lease copies exact tokenizer pieces into fixed workspace scratch,
validates UTF-8 across token boundaries, and matches admitted stop strings
across both token and code-point boundaries. Per-token work allocates no
collection or text value and writes only to caller-owned fixed storage.

Stop bytes are withheld. When a match completes, bytes preceding the match are
published, the matching bytes and the remainder of that token piece are
discarded, and the owner becomes terminal. Without a match, only the longest
suffix that could still become a stop string—or an incomplete UTF-8 scalar—is
retained. `finish_into` copies an unmatched stop prefix for the semantic Luna
`Completed` tail but rejects a
truncated UTF-8 scalar.

`push_token_status` and `finish_into_status` are the narrow
allocation-free owner-resident variants used after an online session consumes
authenticated scheduler evidence. They return scalar status and mutate only on
success, including for the preallocated string-stop matcher shape.

Workspace construction preflights the complete logical cell count before its
first allocation. `LunaIncrementalOutputWorkspace::required_int_cells` and
`LunaIncrementalOutputWorkspace::required_byte_cells` expose the exact reusable
backing-array requirements without allocating; the workspace has no
request-varying reference-cell allocation. Request setup rejects an envelope whose
decoded-delta capacity cannot hold one maximum tokenizer piece, the longest
admitted stop string, and the UTF-8 carry. Stream errors are payload-safe and
never contain token IDs, stop strings, decoded bytes, or tokenizer diagnostics.

This package is not a scheduler, socket writer, event codec, or cancellation
owner. The `service/online_session` foundation retains it together with the
exact production worker authority and semantic Luna event owner. Token/stop
composition preserves one-credit acknowledgement; an outer adapter may stage,
copy, and release a transport frame before acknowledging the semantic credit.
The aggregate uses the scalar matcher for exact string-stop cuts.
