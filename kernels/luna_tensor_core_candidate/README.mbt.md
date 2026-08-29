# LunaTile tensor-core candidate

This package emits one deterministic, qualification-only CUDA WMMA candidate:
exact `sm120`, BF16 row-major `m16n16k16`, F32 accumulation/output, one warp per
tile, 32-byte global operand alignment, no dynamic shared memory, a
128-register compilation ceiling, and an identity epilogue. An opaque
`luna_tile_ir` proof binds the exact seven-instruction operand graph. Lowering
also requires exact program, serial-oracle, parallel-plan, compute-capability,
tile-shape, and block-width identities.

The generated CUDA uses the typed WMMA API; it does not fabricate an inline
PTX or SASS claim. Source emission alone does not prove that either disassembler
observes an allowed tensor-core instruction, that numerics pass, or that the
resource bounds hold. The candidate is always `manifest_bindable=false` and
promotion-ineligible. No NVIDIA execution has occurred for this candidate.
