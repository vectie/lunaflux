# Luna projection strategy

This package owns backend-neutral projection shape classification, decode and
prefill dispatch plans, power-of-two row buckets, and deterministic selection
of offline autotune measurements. It deliberately contains no CUDA, warp,
WMMA, SM, HIP, or device-instruction vocabulary.

Backends translate `SubgroupGemv`, `MatrixTile`, and `ScalarTile` into their
private lowering. Autotuning is an offline release activity; token-step runtime
dispatch reads an already-selected strategy and never benchmarks.
