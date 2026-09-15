# Current compiler closure validation

This is an in-progress validation record, not completion of all five closure
items or a production deployment. Sources are commit-pinned; unrelated local
worktree changes are excluded.

## Clean Linux boundary

The clean `86f48eb7` source passed warning-denied native check and 3079/3079
tests. The release allocation probe also passed on Linux and macOS after its
missing mimalloc interception was repaired. The earlier 3800-test local result
includes unrelated uncommitted changes and is not this clean-source result.

Seven actual release entry packages built successfully: runtime, device child,
token bridge, serving supervisor, candidate exporter, fused bundle exporter and
release binder. An unqualified all-package release build instead tried to link
the `internal/process` library as an executable and failed with missing `main`.
Adding an explicit library declaration did not solve it and was reverted. This
toolchain/build-invocation issue is not represented as a successful root build.

`d50f3b90` additionally fixes offline exporter error reporting: an awaited
stderr write survives captured pipes before abort. The real invalid-argument
subprocess regression passes on macOS and Linux; exporter tests pass 11/11 on
both. This changes only failure reporting, not inference execution or kernels.

## Fresh attention measurements

The prior tuning table binds frontier `0744f940…4385`; current export produces
`dac7cee6…7e00`. Reusing that table is rejected. Removing it permits static
export, compilation and binding, but the static cost estimate chooses c318.
The c322 variant source digest differs only because its entry symbol now has a
candidate suffix. After measured selection, the ordinary selected c322 source
is byte-identical to the historical selected source (`d2d6598a…bbed`). The full
frontier identity still differs; the prior timings were not relabelled as new
measurements. A frontier change is not proof of a changed selected kernel body.

The current twelve exported variants were compiled with their own register
limits and launch geometry. Eight synthetic workloads cover query totals
129/1528, rows 1/8 and history bases 0/4096. Each workload performs numerical
reference and read-only KV checks followed by five alternating baseline/variant
timing trials. These are isolated GPU event timings, not real-model throughput,
and this workload set is not a complete mixed-arrival distribution.

| Candidate | Median summed GPU time across eight workloads, ms |
| --- | ---: |
| 318 | 5.932336 |
| 319 | 10.742767 |
| 320 | 7.128292 |
| 321 | 9.390782 |
| 322 | 4.323038 |
| 323 | 5.917112 |
| 324 | 4.481788 |
| 1003 | 15.169631 |
| 1004 | 7.404372 |
| 1005 | 16.947890 |
| 1006 | 6.525925 |
| 1008 | 4.655865 |

Measured selection retains c322. This demonstrates why static estimates alone
must not replace measurements; it does not establish a speedup over the prior
c322 runtime. New records bind the current frontier/device/toolchain and are
accepted by a fresh export.

Remote work directory: `/tmp/lunaflux-current-eb927891.zY0UED/physical`.
`attention-tuning-current-r2` contains the successful measurements. The first
tuner attempt assumed every variant had 128 threads and stopped on a 64-thread
variant before timing; the correction reads each variant's launch recipe.
The initial tuned-driver retry accidentally paired the new table path with the
old file's digest and was rejected; its logs are retained separately. Neither
failed preparation is counted as a kernel failure or a successful campaign.

The selected c322 passed memcheck and bounded racecheck/synccheck at query
total 1528, eight rows and history base 4096: zero errors/hazards. The first
racecheck ran the timing loop unnecessarily; it was explicitly interrupted,
preserved as a non-passing run, and replaced by four selected-kernel launches
without timing loops. This is bounded selected-kernel coverage, not sanitizer
coverage of every serving path.

## Uninstrumented current-source serving

The `d50f3b90` source archive has SHA-256
`2b6dcbb944da0e31420f9e675d06fb97ebde966d37951655e93744737a2a0862`.
All standard/fused artifacts were generated and compiled for this campaign;
the faster partial ingress chain remains selected. No activation readback,
fixed-graph replay or timing instrumentation was installed in the worker.

Four input/output vectors at C1/C8/C16 completed one warmup and five measured
trials each. Every request returned its required output count, and the owned
server was stopped after the campaign.

| Input/output | C1 output tok/s | C8 output tok/s | C16 output tok/s |
| --- | ---: | ---: | ---: |
| 512/64 | 209.84 | 1034.34 | 1383.78 |
| 1528/32 | 149.53 | 371.55 | 409.93 |
| 3072/32 | 107.74 | 186.18 | 196.55 |
| 4096/64 | 116.36 | 216.49 | 230.06 |

These are medians of batch completion throughput. At 4096/64, pooled request
TTFT p50/p95 is 178/179 ms (C1), 992/1463 ms (C8), and 1704/2952 ms (C16).
The historical C16 value was 230.63 tok/s, so this is essentially unchanged,
not a demonstrated speedup. vLLM/SGLang were not rerun in this campaign.

The numerical issue persists in the current source: at 3072/32, 30 of 40 C8
responses and 15 of 80 C16 responses differ from the first measured response
in their respective cell, with the first changed token at index 31. The other
ten cells have no within-cell differences in these measured trials. These
counts use a cell-local reference, not an independent correctness oracle.
Completion of requests therefore does not close cross-batch numerical
acceptance. Final-token causality and remaining common strategy/effect
integration are still open.

## Mixed workload coverage

Seven cases at C8/C16 completed one warmup and three measured repetitions,
rotating case order between repetitions. All responses returned the requested
count. Arrival lag p95 was at most 1 ms, and the owned server was stopped.

| Case | C8 median wall ms | C16 median wall ms | C16 TTFT p50/p95 ms |
| --- | ---: | ---: | ---: |
| Uniform, shared input | 1220 | 2185 | 693/1280 |
| Uniform, distinct inputs | 1222 | 2190 | 693/1281 |
| Unequal input lengths | 1562 | 2432 | 710/1421 |
| Unequal input/output lengths | 1765 | 2706 | 599/1422 |
| Uniform, staggered arrivals | 1222 | 2191 | 524/912 |
| Unequal lengths, staggered | 1794 | 2769 | 574/1123 |
| Unequal lengths, reversed/staggered | 1770 | 2740 | 640/1144 |

Each C16 case contains 32768 input and 1024 output tokens, but the unequal-input
cases have 58753032 causal prefill pairs versus 33570816 for uniform input.
Hence equal token totals do not isolate scheduling overhead. Staggering also
changes per-request waiting and overlap; these runs are not a kernel-only
comparison or a natural-language accuracy evaluation.

Downloaded source/measurement logs, generated artifacts and serving responses:
`/tmp/lunaflux-current-results.7SihfN/results.tar.gz`, SHA-256
`5297ebaf0584f2020e6e6a8a614014133c55ff46a5cd66be5fb539134640006d`.
Launch argument files, process command lines and temporary build directories
are excluded. The remote archive is
`/tmp/lunaflux-current-compiler-20260915-results.tar.gz`.
