# Offline projection observations

This package parses immutable tuning inputs; it performs no device queries,
filesystem operations or runtime profiling. Selection happens during AOT export.

## Fold pipeline records

`parse_projection_fold_records` accepts the following tab-separated format:

```text
luna-projection-fold-v1
scope<TAB>DEVICE<TAB>TOOLCHAIN_SHA256
record<TAB>BASE_SOURCE_SHA256<TAB>ARTIFACT_SOURCE_SHA256<TAB>MEDIAN_NS<TAB>SAMPLES<TAB>CHOICES
```

The caller provides the exact scope. Each source family needs a baseline record
whose two hashes match and whose choices are `-`. An alternative's choices are
comma-separated `ROLE:STAGES:TRANSFER_TILES:FRAGMENT_STAGES`; roles are `matrix`,
`sibling`, or `intermediate`, with no repeated role. At least three samples and
positive median time are required. Compare matching workload vectors; a full
MLP-chain measurement must not be mixed with gate-only or down-only timing.

The CUDA AOT consumer regenerates a candidate before selecting it and compares
its source identity. Unsupported shared-memory combinations are excluded.
Unmeasured source families retain the baseline. Equal-cost baseline wins;
equal-cost alternatives have deterministic source-identity ordering.
The consumer defaults to a 1% minimum improvement over the baseline to avoid
switching schedules for small timing fluctuations. This explicit selection
parameter is independent of model family and does not alter raw observations.

The record does not imply that deeper pipelines are faster. Hardware timings
remain authoritative, and this format does not replace query/history/batch
attention routing or a complete serving benchmark.
