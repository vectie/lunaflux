# Batched decoder request frames

`BatchedDecoderFrame` translates the existing scheduler wire plan into
preallocated request ports for a slot-addressed hybrid decoder. It is a runtime
effect owner: model-independent and backend-independent, with no kernel
selection, numerical comparison, hashing, device submission or HTTP handling.

- Slot identity is `(request_id, request_generation)`, not batch ordinal.
- Mixed chunked-prefill/decode rows share one flattened token buffer.
- New owners reset both sparse history and recurrent state; continuation uses
  the exact committed token position. An unseen prefix cannot be assumed live.
- Selected output rows exclude non-final chunks. Sampling remains in the
  original immutable wire plan, with an explicit output-to-plan-row mapping.
- GPU completion commits all positions; failed execution poisons the owner.
- Release is explicit, generation-aware and prohibited during in-flight work.
- Construction allocates storage; staging/copy/completion use fixed arrays and
  existing by-value wire views, not growing maps or newly allocated token arrays.

The batch ABI has counts, IDs, sequence offsets/slots, recurrent and sparse reset
vectors, selected output rows, absolute positions, per-token slots and per-sequence
base positions. It is **not compatible** with single-history sparse kernels.
With an immutable `PagedHistoryDomain`, staging also copies the scheduler's
page IDs/generations and per-sequence page offsets, plus per-token sequence
ordinals. The domain bounds shared physical rows separately from request
length and cache-slot count; only page-table metadata scales with concurrency.
Short page coverage or an index outside the physical pool rejects the entire
stage without advancing retained state. DMA copies only the live page prefix.
Device owners must implement this ABI before serving can advertise concurrency.
This host owner does not establish GLM HTTP, multi-request GPU execution,
prefix reuse, stochastic TP sampling or full-service deployment by itself.
