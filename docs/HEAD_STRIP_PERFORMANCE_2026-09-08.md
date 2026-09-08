# Selected-row head reduction strip: physical result

Compiler implementation: `8d2d6c1`. This is an isolated projection-kernel
comparison, not end-to-end serving or a new vLLM/SGLang benchmark.

## Change

The pure compiler plans a 128-element operand-storage strip for selected-row
matrix projection with one output tile per workgroup. Its eight ordered
16-element matrix folds are unchanged. CUDA lowers the plan to gathered input
and contiguous weight copies with one warp per 16 vocabulary columns.

For the measured BF16 `K=1024, N=151936` head, shared storage falls from
40,960 bytes (current eight-warp full-input residency) to 9,216 bytes. The
one-warp variant launches 9,496 CTAs instead of 1,187. This is functional
strip mining and lifetime reduction, not reassociation or a Qwen-name branch.
The ordinary one-row strategy remains unchanged; no bucket-1 record is added.

## Measurements

RTX 5060 Ti, CUDA 13.1.115, exclusive GPU. Three independent runs, each with
three warmup graph replays and ten timed launches. The 311,164,928-byte weight
matrix exceeds the 33,554,432-byte L2 cache. Medians below are kernel latency
in microseconds, not total request latency. Kernel order within a trial was
fixed; this is not an order-balanced statistical campaign.

| Compiled variant bound | Actual tokens / selected rows | Current resident group8 | New strip group1 | Speedup |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 8 / 8 | 1,523.005 | 767.949 | 1.983× |
| 256 | 8 / 8 | 1,526.733 | 767.405 | 1.989× |
| 256 | 32 / 32 | 1,649.949 | 1,407.277 | 1.172× |
| 256 | 256 / 32 | 1,667.203 | 1,399.270 | 1.191× |

Both compiled bounds were separately exported and tested. **The bound-256
tuning latency below measures eight actual tokens and eight selected rows
inside the 256-token-capable variant**, targeting the short-C8 workload. It is
not a claim of 767 µs for 256 actual tokens or 256 selected rows. The latter
selected-row count exceeds this fixture's 32-row maximum. The 256-token /
32-row case was independently measured and is shown separately above.

Simply choosing group1 while retaining full-row residency regressed badly
(approximately 6.4 ms at eight rows). The strip-storage change, not distribution
alone, is essential to this result.

All six full-output runs passed bitwise comparisons against the existing
kernel, including empty, single-row, ragged, projected-padding and row-tail
cases. Bounded full-stride memcheck reported zero errors; racecheck reported
zero hazards, errors or warnings. Resources were explicitly released and the
final GPU process list was empty. Compiler tests: 26/26; AOT tests: 22/22.

## Tuning handoff

Fields follow `luna-projection-tuning-v1`: family4 = language-model head,
numeric1 = BF16, shape `(1024,151936,0)`, stable strategy1001, matrix class3,
tile `(16,16,16)`, output groups1, residency1 = reduction-tile streaming.
The K128 lifetime is separately bound into the compiler schedule identity.
Latencies convert measured median microseconds to integer nanoseconds by
truncation; sample count3. These are workload-scoped measurements, not universal
performance guarantees for every occupancy accepted by a compiled bound.

```text
record	4	1	1024	151936	0	1001	256	3	16	16	16	1	1	767404	3
record	4	1	1024	151936	0	1001	8	3	16	16	16	1	1	767948	3
```

Scope: `cuda-aot-v1`, `sm_120`, device
`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, toolchain digest
`1ebf9cc2850735f80ce09a6d7bfbff2371916a3e0bbab81c182f09c4eb8ccbb7`.
The bound256 record replaces the previous stable3008 resident-head selection;
do not retain stale competing measurements as if measured in this campaign.

Remote results: `/dev/shm/lunaflux-head-strip-20260908-r2-results`.
Downloaded results: `/private/tmp/lunaflux-head-strip-20260908-r2-results-downloaded`.
Generated handoff snapshot: `/private/tmp/lunaflux-head-strip-tuning-20260908.v1`.
