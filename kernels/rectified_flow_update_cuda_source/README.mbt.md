# Rectified-flow F32 update lowering

Family-neutral CUDA source for an ordered eta-zero update: compute clean state
from velocity, then blend current and clean state using precomputed sigma
ratios. All arrays are F32. One thread owns one element; intermediates stay in
registers, with no full-tensor scratch, barriers or per-element division.

The four operands are state, velocity, three F32 coefficients, and output.
Exact state/output alias is legal after the joint denoiser has finished reading
both modality states. Regions must otherwise be disjoint; no partial alias.
Launch block size must match the rendered block size and grid must cover the
element count. Coefficients are prepared once for each modality and step.

This is source lowering, not a compiled artifact or runtime binding. GPU
numerics, sanitizer and timing remain required. It must not be substituted for
the older BF16 delta-update ABI under the same symbol or numerical identity.
