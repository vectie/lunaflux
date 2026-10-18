# Batched decoder rank effects

This model-neutral owner binds `BatchedDecoderFrame` to stable device ports and
preallocated asynchronous transfer lanes. It preserves sequence slots, logical
positions, page IDs/generations, output-row mapping and the original sampling
plan. The GLM adapter borrows these addresses through `GlmDecoderInputs`.

The lifecycle is explicit:

`begin → inputs_ready → submit → progress → commit`

`progress` retires local execution, append errors and live output transfers. It
does not commit model state. The coordinator calls `commit` only after every
rank retires; a local or peer error poisons the request histories. Cancellation
must first retire device work, then release the matching request generation.
Close drains DMA leases before closing the descriptor-owning queue and ports.

Capacity, arrays, scalar metadata and transfer lanes are prepared once. Empty
optional ports never enqueue a zero-length transfer; inactive output tails are
not downloaded. Native transfer-double tests require zero warmed allocations
and no blocking synchronization. These tests do not execute CUDA arithmetic.

This is not a service or a stochastic sampler. A greedy suffix cannot satisfy
non-greedy sampling parameters merely because this owner preserves them. The
TP coordinator, global-vocabulary sampler, HTTP adapter and full-model admission
must be connected and independently validated before serving is enabled.
