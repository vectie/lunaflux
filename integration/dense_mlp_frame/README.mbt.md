# Prepared dense SwiGLU branch

`DenseSwiGluPrecision` defines geometry, exact BF16 storage and ordered numerical
rounding independently of the model or device. The CUDA lowering consumes that
plan and separate row-major gate, up and down weights. Gate/up and SiLU produce
one shared BF16 product frame; down projection consumes it without recomputing
gate/up for each output column.

`DenseMlpFrame` owns the product and two AOT functions. Preparation binds the
counts-dependent launches once. A caller borrows their view into its larger
ordered queue, with no intermediate completion event or host handoff. Close
that queue before closing the frame. Partial preparation retains deterministic
cleanup authority. Counts, weights and input/output allocations are borrowed.

There is no model-family branching, request JIT or token-step allocation here.
The current per-output ordered-F32 projection schedule establishes executable
numerical behavior; it is not a tensor-core GEMM or a throughput claim. A faster
schedule must preserve the declared BF16 rounding points or explicitly choose
another numerical contract.
