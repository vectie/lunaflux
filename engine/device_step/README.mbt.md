# Reusable device-step staging

This package translates one authenticated `SubmittedSchedulePlan` into seven
fixed-capacity little-endian Int32 device operands:

1. counts: prefill rows, decode rows, total rows, query tokens, page entries;
2. token IDs;
3. absolute token positions;
4. CSR query-row offsets, including the terminal token count;
5. full sequence lengths after each row;
6. CSR page offsets, including the terminal page count;
7. physical page indices.

Rows retain protocol order: prefill first, then decode. Prefill positions are
`sequence_token_start + local_query_index`; decode has one query at
`sequence_token_position`. Every row carries exactly
`ceil(sequence_length / tokens_per_page)` pages. A positive `PageId`
generation is only a structural check here; generation and active-residency
authorization occurred in `scheduler/core` before this exact-epoch submitted
plan was committed, and the scheduler retains active references through
completion. Generations are deliberately stripped before upload, so the GPU
receives only range-checked physical indices. A future wire transport must
carry equivalent generation and page-lease authority.

The owner is confined to the worker that owns its borrowed device context.
Host buffers and seven matching device allocations are created once at
startup. A stage performs a complete read-only preflight before writing host
storage, including plan-sequence and stage-epoch authorization. The startup
limits bind the loaded model's exact maximum sequence length and token ID;
protocol admission is never trusted as the only semantic envelope. Staging
then uploads an invalid zero counts header, all payloads, and the real counts
header last. Any upload failure poisons the owner, preventing kernel binding,
finish, or another stage until deterministic close.

This staging slice exposes neither launch arguments nor any allocation alias.
A future executor in this owner package must mediate launch under lifecycle
authorization. `finish` retires the exact stage and advances the accepted
sequence predecessor; replacement owners may start from an explicit committed
predecessor seed. No close authority escapes.

This is descriptor staging, not paged attention or model execution. The
release reuse test and generated-code inspection show stable fixed storage and
allocation-free packing helpers, but are not a release allocation gate. The
existing repository hot-path probe covers scheduler and worker-protocol paths,
not this public stage/transfer path. Runtime allocation instrumentation for
that path, plus physical-CUDA transfer, sanitizer, and leak evidence, remains
open before Phase 3 promotion.
