# Output-row projection

The executor compiles a terminal-output view of the step descriptor. Attention,
KV writes and all upstream operations retain the original descriptor. Only the
terminal BF16 head and sampler consume a stable, compact row-end view. Original
request rows map to compact result slots in startup-allocated host storage.
The descriptor writer builds both views in its existing single pass.

Storage is appended to the existing count allocation: the original five I32
counts remain at offset zero, five output counts begin at byte 20, and selected
row ends begin at byte 40. The count upload publishes both views in one call;
there is no extra allocation or transfer call in a token step. This adds
`20 + 4 * (maximum_rows + 1)` bytes to that startup allocation. Existing public
descriptor arguments still expose exactly their original 20-byte count region.

Captures keep the same graph topology and pointers. With zero producing rows,
the head and sampler return before arithmetic. Mixed frames use compact
terminal rows but retain all KV effects. A one-output frame that originally
had multiple rows retains the matrix arithmetic branch using one zero-padded
inactive row; compaction must not silently replace its matrix dot with the
single-row GEMV reduction tree. Padding has no request, RNG or publication slot.

This eliminates head/sampling arithmetic, not every upstream pure operation in
a captured graph. Eager effect-only prefix execution remains the stronger
all-nonproducing optimization. FP8/I8 executors and diagnostic observations do
not acquire this BF16 terminal-row transformation implicitly.

Required validation: mixed/empty/alternating descriptor equivalence, fixed
request-to-result mapping, seeded sampling preservation, capture replay,
correctness on ragged token rows, sanitizers, and end-to-end Qwen measurement.
