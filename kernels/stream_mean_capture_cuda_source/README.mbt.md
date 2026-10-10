# Segmented residual-stream mean lowering

`StreamMeanCapturePrecision` declares BF16 `[row, stream, hidden]` input and
BF16 `[row, segment, hidden]` output. Each stream fold accumulates in F32,
divides in F32, then rounds once to BF16. There is no RMS normalization,
learned control, vocabulary projection or model-family decision in this package.

Segment ownership is fixed at AOT generation. The lowering emits one symbol per
segment, so the existing pointer-only native launch ABI needs no scalar extension
and no per-step segment descriptor upload. CUDA grid/thread choices stay here.
Only live rows are written; inactive rows retain their existing storage.

`scripts/run-stream-mean-capture-probe.mbtx` exports the actual renderer output
and compares it with an independent CPU mean oracle on GB10. The initial run
passes 192 cases and all three CUDA sanitizers, with zero memcheck leaks. This
is a small component correctness test, not DSpark model accuracy or throughput.
