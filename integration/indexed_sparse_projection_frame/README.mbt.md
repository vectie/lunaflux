# Prepared sparse projection frame

`IndexedSparseProjectionFrame` owns thirteen BF16 intermediate allocations and
twelve AOT functions, borrowing hidden/counts and checkpoint weights. The
projection precision plan computes workspace and weight bytes before allocation.
All argument binding happens at startup. `launches()` and named borrowed
`SparseProjectionOutputs` connect to caller-owned execution; no request-path
source rendering, allocation or host copy is added.

`prepare_retained_attention` prepares history and attention with these outputs,
then returns eighteen launches: reserve, twelve projections, append/publish,
pool/index/attention. Plan geometry/capacity must agree. The caller owns the
single execution queue and its completion. Close it first, followed by attention,
history, projection and finally borrowed checkpoint/hidden allocations. Partial
prepare failures remain explicitly closeable.

This rope-free numerical chain does not implicitly implement other models'
rotary, FP8 index caches or index transforms. The GLM Flash adapter lives in
`integration/glm53_sparse_projection`; model names do not enter the precision
plan or CUDA renderer.

`IndexedSparseDecoderFrame` composes the residual-stream envelope, the eighteen
sparse stages and BF16 output projection. Its twenty-three launches are borrowed
into the same caller queue as a following FFN. The immutable decoder precision
plan accounts for the expanded attention output and envelope workspace as well
as projection/attention scratch; persistent history is budgeted separately.
Projection weights, output weight, pool APE and four envelope operands remain
caller-owned. Close the caller queue before the frame. The device append-error
descriptor is exposed for worker output-delivery error propagation.

The GLM adapter adds its second mHC envelope and routed/shared FFN to make a
thirty-seven-launch block. Whole-model worker dispatch and real-checkpoint GPU
numerical execution are not established by this component composition.
