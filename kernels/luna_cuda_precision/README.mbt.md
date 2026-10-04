# CUDA precision conversion lowering

Lowers precision IR into deterministic AOT source. CUDA headers and conversion
instructions stay in this terminal package. NVFP4 global amax, block scaling and
packing are ordered kernels, not workgroups pretending to synchronize globally.
One thread owns each packed byte, including its two INT4/FP4 nibbles.

`tile_load_source` is consumed by the existing projection compiler's producer.
It reconstructs and rounds BF16 operands; it is not native low-bit MMA. Formats
are specialized at AOT generation, with no runtime format switch or JIT.

See [GPU validation and limitations](../../docs/PRECISION_IR_2026-10-04.md).
