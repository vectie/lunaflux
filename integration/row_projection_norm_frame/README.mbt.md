# Prepared row projection and normalization

This model-neutral owner lowers `RowProjectionNormPrecision` using the existing
packed E4M3/UE8M0 projection and weighted BF16 RMSNorm kernels. The intermediate
BF16 round is an explicit numerical boundary, not silently fused away.

Preparation checks context ownership, counts, input and compact payload/scale/norm
spans before allocating two output frames. Functions and argument arrays are
prepared once. The parent ordered queue borrows both launches and owns their
submission/completion; the frame adds no token-step allocation or synchronization.

Close the borrowing queue first, then this frame, then borrowed weights/module.
Output storage is borrowed and ready only after the parent completion retires.
The native device-double tests cover exact budgets, short operands, partial abort,
32 warmed submissions without allocation, and balanced deterministic release.
The independent GB10 fixture covers packed projection, BF16 boundaries, weighted
normalization, zero/extreme scales, invalid/inactive rows and deterministic replay.

This is a reference executable implementation, not a tuned GEMM performance claim.
