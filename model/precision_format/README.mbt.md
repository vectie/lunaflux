# Backend-neutral precision encodings

Immutable `Representation` separates payload elements, row-aligned bit packing,
scale grouping, scale dtype and global scaling. `PackedTensor` implements the
scalar oracle and deterministic offline quantization. All input codes/scales
are validated at construction, never in a serving token loop.

BF16, FP16, F32, FP8 E4M3FN/E5M2, symmetric INT8/INT4, E2M1, NVFP4, MXFP4
and MXFP8 are represented explicitly. Model-specific checkpoint packing is a
separate adapter; device-specific scale swizzles are a terminal lowering choice.
UE8M0 FP8 scale grids are explicit block geometry (default 1x32), including
1x128 activations and 128x128 matrix weights. A scale encoding alone does not
force all operands to share one block shape.

See [precision architecture and actual completion scope](../../docs/PRECISION_IR_2026-10-04.md).
