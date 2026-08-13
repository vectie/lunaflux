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
retained. `finish_into` publishes an unmatched stop prefix but rejects a
truncated UTF-8 scalar.

Construction rejects an envelope whose decoded-delta capacity cannot hold one
maximum tokenizer piece, the longest admitted stop string, and the UTF-8 carry.
All pattern and failure-table allocation occurs at construction. Stream errors
are payload-safe and never contain token IDs, stop strings, decoded bytes, or
tokenizer diagnostics.

This package is not a scheduler, socket writer, event codec, or cancellation
owner. A later online session must retain the exact scheduler handle, follow
the scheduler publication sequence, and acknowledge output only after a full
frame write.
