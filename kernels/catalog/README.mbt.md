# Exact kernel catalog

`kernels/catalog` resolves each semantic model operation to one exact startup
implementation for one device target. A content-addressed implementation names
a CUDA module digest plus a stable kernel-family identity; profile-specific
entry points are selected later by `kernels/launch_contract`.

The implicit cuBLASLt BF16 implementation is legal only for
`OutputProjection` and `LanguageModelHead`, whose current semantic layouts are
representable by the narrow single-GEMM device ABI. `QkvProjection` has three
weights and a packed Q/K/V output contract, so it requires an AOT family until
an explicit multi-descriptor vendor ABI exists. Silent decomposition or layout
fallback is forbidden. A nonempty vendor workspace requires the ABI's exact
256-byte alignment; incompatible catalog entries fail during admission.
