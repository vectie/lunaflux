# Family-neutral gated RMSNorm CUDA source

This package renders a deliberately serial correctness CUDA kernel for BF16
`[row, head, dimension]` inputs. Each head vector is RMS-normalized with an
ordered F32 sum, multiplied by a BF16 per-dimension weight in F32, multiplied
by the F32 sigmoid of a separate BF16 gate, and rounded once to BF16 output.

The package owns no model identity, projection, residual, cache, compiler,
artifact, launch, qualification, or execution authority.
