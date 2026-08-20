# Incremental decoded output

`service/incremental_output` owns one request's bounded generated-token decode
state. It copies exact tokenizer pieces into fixed startup-owned scratch,
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

Construction rejects an envelope whose decoded-delta capacity cannot hold one
maximum tokenizer piece, the longest admitted stop string, and the UTF-8 carry.
All pattern and failure-table allocation occurs at construction. Stream errors
are payload-safe and never contain token IDs, stop strings, decoded bytes, or
tokenizer diagnostics.

This package is not a scheduler, socket writer, event codec, or cancellation
owner. The `service/online_session` foundation retains it together with the
exact production worker authority and semantic Luna event owner. Token/stop
composition preserves one-credit acknowledgement; an outer adapter may stage,
copy, and release a transport frame before acknowledging the semantic credit.
The aggregate uses the scalar matcher for exact string-stop cuts.
