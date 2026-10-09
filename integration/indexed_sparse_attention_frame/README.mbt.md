# Projected indexed sparse attention

`IndexedSparseAttentionPrecision` describes the model-independent geometry and
storage. Query rows and retained history are independent: a one-row decode frame
may read a long history without allocating a selected-index row for every key.
CUDA lowering preserves stable selection, ordered F32 score/softmax arithmetic,
BF16 probability and BF16 output rounding. It currently uses correctness-grade
serial schedules, not a throughput-tuned sparse serving implementation.

The producer frame composes pooling -> index selection -> sparse attention.
History counts, keys, validity and query counts/absolute positions are borrowed
separately. Intermediate allocations/functions are prepared once and borrowed
into the caller's ordered queue. No extra completion boundary or host readback
is introduced. Projected histories must contain one request and the caller must
retain and reset them correctly when using the borrowed-history API.

`IndexedSparseHistoryPrecision` / `IndexedSparseHistory` add a request-owned
alternative. The cache owns projected index keys, pool gates, attention K/V,
validity, history length, append descriptor and query positions. At startup,
`prepare` binds one fixed query-count buffer, reset flag and projected frame.
`prepare_retained` prepends reserve/reset -> parallel BF16 copy -> publication
before pooling/index/attention: six launches in one queue, no host handoff.
The same frame handles prefill chunks and single-row decode via its live count.
An idle step preserves history; a reset flag clears validity before a new
request. Capacity exhaustion is reported through the device append descriptor,
not silent truncation. A scheduler must route that error before output delivery.
Distinct request owners do not share storage. Queue/frame closure precedes
history closure; caller updates borrowed metadata only when that queue is idle.

`for_shared_indices` prepares one attention launch, with zero owned workspace,
using another layer's device-resident selection. It does not repeat pooling,
score computation or top-k. The model execution plan owns producer/consumer
layer ordering. Close all borrowing queues and consumers before the producer;
closing a consumer never releases the borrowed selected indices or histories.

The small GPU fixture has a 32-key capacity, 15 valid history keys, three prefill
queries and one decode query. It verifies stable selected pools, visible tail,
GQA, zero inactive output, exact prefill/decode equality and clean memory/race/
sync checks. This is not whole-model GLM/DeepSeek execution or a long-context
throughput benchmark. The retained-history fixture additionally checks chunked
prefill versus decode bitwise output, independent requests, reset/reuse, idle
preservation, exact-full capacity and overflow with clean memory/race/sync
results. Model projections, complete DSA block, batched slot scheduling and
worker integration remain necessary features.
