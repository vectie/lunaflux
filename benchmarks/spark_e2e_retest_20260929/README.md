# Spark Qwen3-0.6B serving retest

Measured source: `53deab59`, with the explicit sm121/CUDA 13.0.88 benchmark port.
`luna-full` denotes explicit full-ingress evaluation, not the wrapper's unfused
default. See the [report](../../docs/BENCHMARK_SPARK_ARCHITECTURE_RETEST_2026-09-29.md)
for the route regression, setup, limitations, output agreement and raw archive.

- `summary.json`: 60 engine/workload cells, five measured batches per cell.
  Throughput is aggregate output tokens/sec including prefill. Latencies are
  client-observed milliseconds, not GPU instruction timing.
- `memory-summary.json`: cell-boundary system-memory observations in KiB,
  not continuous process/GPU peaks. Int64 values are serialized as strings.

All engines completed 620 measured requests. Raw benchmark automation is MoonBit
`.mbtx`; no production runtime change or deployment was performed in this retest.
