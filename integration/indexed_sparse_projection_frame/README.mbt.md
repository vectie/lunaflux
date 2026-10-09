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
plan or CUDA renderer. A full decoder must still add output projection, mHC/FFN
and model worker semantics.
