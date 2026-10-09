# Decoder scaled RoPE CUDA source

This package renders model-neutral, offline-only CUDA source for scaled YaRN
rotation of adjacent BF16 pairs. The six-pointer ABI is step counts, absolute
I32 positions, query RoPE input/output, and key/value RoPE input/output. Query
and key/value buffers contain only `[row, head, rope_dimension]` suffix slices;
unrelated head components are deliberately outside the contract.

The separate inverse renderer accepts one in-place complete BF16
attention-output head. It leaves the non-RoPE prefix untouched and applies the conjugated YaRN
frequency only to the adjacent-pair RoPE suffix, matching the official
post-attention, pre-output-projection boundary.

The source fixes ordered F32 arithmetic, CUDA `powf`/`logf`/`floorf`/`ceilf`
frequency construction, `cosf`/`sinf`, and BF16 round-to-nearest-even output.
Its transcendental behavior remains compiler/toolchain-bound and requires
independent numerical qualification. It conveys no compiled-artifact, loader,
launch, or runtime authority.
