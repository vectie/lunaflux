# Spark counter diagnosis, 2026-09-29

See [the report](../../docs/BENCHMARK_SPARK_COUNTER_DIAGNOSIS_2026-09-29.md)
for workload, timings, source attribution, memory limits and capture caveats.

- `counters.json`: 37 records extracted from three Nsight Compute CSV exports.
  Values retain the original strings and units. Missing metrics are omitted,
  not inferred to be zero. Warp instructions must not be called scalar thread
  instructions. Replay durations must not be used as serving performance.
- `memory-summary.json`: minimum sampled host available memory by service
  campaign, including the initial incomplete vLLM trace. The standalone decode
  probe has an 8 GiB container cap, not a separate continuous-memory series.

`luna-counters` is a bounded three-launch service capture;
`decode-counters` is an exact-CUBIN synthetic-input CPU-oracle-checked probe;
`vllm-counters-final` is the final serving capture, not the earlier partial
exports. The original reports and reproducibility scripts are retained in the
downloaded archive named in the report. SGLang was not counter-profiled here.

The subsequent [three-agent source review](../../docs/BENCHMARK_SPARK_THREE_KERNEL_SOURCE_REVIEW_2026-09-29.md)
adds CPU-only SASS analysis. `prefill-opcodes.mbtx` sums the `Instructions
Executed` column by complete opcode in the first matching kernel section of
the archived three-column Nsight SASS CSV (predicate prefixes are removed).
For the extracted archive directory, run:

```sh
moon run benchmarks/spark_counter_diagnosis_20260929/prefill-opcodes.mbtx /tmp/lunaflux-counter-diagnosis.rUW2WBeZ/luna-counters-source.txt lunaflux_attention_prefill_tile_compiler_v1
moon run benchmarks/spark_counter_diagnosis_20260929/prefill-opcodes.mbtx /tmp/lunaflux-counter-diagnosis.rUW2WBeZ/vllm-counters-final-source.txt flash_fwd_splitkv_kernel
```

This reproduces MOV counts4,187,136/380,672 and BF16 HMMA counts
4,325,376/4,456,448. It neither launches CUDA nor profiles new work.
