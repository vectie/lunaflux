# Luna projection strategy

This package owns backend-neutral projection shape classification, decode and
prefill dispatch plans, serving row buckets, and deterministic selection of
offline autotune measurements. The explicit workload classes are
`DecodeGemv`, `SmallRowTiledGemm`, and `LargeRowTiledGemm`. Row buckets are
`1`, `2-8`, `9-16`, `17-32`, then power-of-two prefill extensions through the
bounded runtime maximum. It deliberately contains no CUDA, warp, WMMA, SM,
HIP, or device-instruction vocabulary.

Backends translate `SubgroupGemv`, `MatrixTile`, and `ScalarTile` into their
private lowering. Each offline measurement key binds backend, target, numeric
type, projection shape, and full strategy identity. Stable strategy identifiers
must be unique inside one bound shape and bucket. Startup rejects duplicate or
conflicting keys and materializes a bounded dispatch table. Missing records
use the documented static fallback. Token-step dispatch performs only fixed
comparisons plus one table lookup; it does not allocate, scan, or benchmark.
