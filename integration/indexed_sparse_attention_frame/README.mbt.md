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
retain and reset them correctly; this frame does not yet own KV cache append.

`for_shared_indices` prepares one attention launch, with zero owned workspace,
using another layer's device-resident selection. It does not repeat pooling,
score computation or top-k. The model execution plan owns producer/consumer
layer ordering. Close all borrowing queues and consumers before the producer;
closing a consumer never releases the borrowed selected indices or histories.

The small GPU fixture has a 32-key capacity, 15 valid history keys, three prefill
queries and one decode query. It verifies stable selected pools, visible tail,
GQA, zero inactive output, exact prefill/decode equality and clean memory/race/
sync checks. This is not whole-model GLM/DeepSeek execution or a long-context
throughput benchmark. Model projections, cache append, complete DSA block and
worker integration remain necessary features.
