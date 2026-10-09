# Channel-major projection CUDA source

This package renders family-neutral correctness-grade pointwise Conv1d source
for contiguous `[batch,channel,time]` tensors. The ABI uses separate F32 input,
`[output,input,1]` weight, bias, and output buffers. Accumulation starts from
bias and proceeds in ascending input-channel order with explicit F32
round-to-nearest multiplication and addition. It provides source only, not
compilation, loading, launch, or qualification authority.
