# GLM Flash sparse projections

This adapter maps Flash DSA layers to the general rope-free projection precision
plan and binds twelve exact checkpoint operands. Use the existing streamed
`CheckpointDeviceWeights` uploader, then pass allocations in the returned order
to `IndexedSparseProjectionFrame::prepare`. No dense bank is duplicated and no
Python/ModelOpt runtime is imported. Normalization vectors remain rank one,
including the index LayerNorm bias.

One query requires 181,056 bytes of projection intermediates; the twelve BF16
weights total 115,610,112 bytes. These totals exclude retained expanded KV,
pool/index scratch, pool APE, output projection, mHC and FFN. Compute the history
budget independently before loading any banks; one million expanded KV rows do
not fit a Spark simply because the query frame is small.

The index LayerNorm epsilon is a required caller argument. The installed config
has low-rank RMSNorm epsilon 1e-5 but does not declare a separate index LayerNorm
epsilon; this adapter does not silently borrow DeepSeek-v3.2's epsilon.
The current generic implementation uses BF16 index projection intermediates.
It does not claim equivalence to a quantized FP8 index cache or complete GLM
checkpoint numerics. Whole DSA block, checkpoint GPU execution and model worker
composition remain separate feature work.
