# Rope-free sparse projection program

Pure `IndexedSparseProjectionPrecision` owns dimensions, distinct RMSNorm and
affine LayerNorm epsilon, BF16 intermediate boundaries and startup memory sizes.
CUDA lowering composes eight dense projections, two low-rank RMSNorm operations,
KV splitting and index-key affine LayerNorm. Index head/dimension score scaling
remains in index scoring, not duplicated in the head-weight projection.

The twelve launch stages produce Q, K, V, index queries/keys/head weights and
pool gates from hidden rows. They are numerical reference schedules, not tuned
Tensor Core serving kernels. This explicit rope-free contract fits GLM Flash's
zero rotary dimension; it does not replace other models' rotary, activation
quantization or Hadamard/index-cache transforms.

`integration/indexed_sparse_projection_frame` allocates thirteen intermediates
and binds all operands once. Its retained-attention composition executes reserve,
the twelve projection stages, append/publication and pool/index/attention as
one eighteen-launch caller queue. It introduces no new completion boundary.
The queue closes before attention/history and then projection owners. Hidden,
counts and checkpoint allocations remain caller-owned.

`IndexedSparseDecoderCudaSource` lowers the composed pure decoder plan by
reusing these projections, retained-history and attention sources, mHC source
and the existing BF16 output projection. It adds no new numerical arithmetic
or device strategy to the model adapter. The complete model block may append
an FFN envelope/program without adding a second queue. This is executable
component composition, not a tuned whole-model worker.
