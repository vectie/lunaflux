# Current compiler closure validation

This is an in-progress validation record, not completion of all five closure
items or a production deployment. Sources are commit-pinned; unrelated local
worktree changes are excluded.

## Latest integrated refactor: 3836a256

The clean source archive SHA-256 is
`1f0638ec159a867a48847047d2ab230ff82cce79df20322ae46548dc7f4db4f1`.
Info, format, native warning-denied check and all 3,098 native tests passed.
All seven release entry packages built, followed by fresh candidate export,
standard/fused CUDA compilation, release binding, materialization and capacity
validation. Previous measured tuning records were explicitly reused as selection
inputs and accepted; they are not new tuning measurements.

The rebuilt uninstrumented runtime completed one warmup and five measured trials
for each cell below. Every response returned the required token count. These
are median batch-completion output throughput, not isolated GPU kernel timing.

| Input/output | C1 output tok/s | C8 output tok/s | C16 output tok/s |
| --- | ---: | ---: | ---: |
| 512/64 | 209.84 | 1030.18 | 1380.05 |
| 1528/32 | 151.66 | 372.09 | 409.27 |
| 3072/32 | 107.74 | 186.32 | 197.15 |
| 4096/64 | 117.22 | 216.49 | 230.32 |

4096/64 request TTFT p50/p95 is 174/179 ms at C1, 992/1460 ms at C8,
and 1701/2947 ms at C16. Compared with the earlier d50f3b90 C16 value of
230.06 tok/s, performance is essentially unchanged. The behavior-preserving
effect-plan refactor is not a demonstrated speedup. No competitor was rerun.

The 3072/32 numerical issue persists: 30/40 C8 and 15/80 C16 responses differ
from their cell's first measured response, first at token index 31. The other
ten cells have no within-cell token changes. This comparison is not an
independent numerical oracle and does not close numerical acceptance.
Current-source mixed workloads, fixed-graph diagnosis and relevant physical
sanitizer coverage are not established by this uninstrumented matrix.

Remote source/results: `/tmp/lunaflux-integrated-3836a256.9vOS4i`.
The runner stopped its owned service; no live owned-group processes remained
and the target GPU was idle after the run. The reused offline runner emitted
unused-helper and future cancellation-cleanup warnings; these are separate
from the warning-denied production build. Downloaded test/build/measurement
logs and responses: `/tmp/lunaflux-integrated-results.AHwvNG/results.tar.gz`,
SHA-256 `18b669dd36f2be14d7e63351b27c5abd1814bb8c02061103b798d97b44a95f78`
(verified against the remote archive). Launch argument and process-argument
files are excluded.

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

## Current-source identical-prefix diagnostic (3836a256)

The isolated eager diagnostic `/tmp/lunaflux-shared-head-final.ZCZe13`
reproduces the final-token difference with identical 3072-token inputs and
32 generated tokens, C1 then C8, one trial each. This is instrumented numerical
diagnosis, **not a throughput measurement**. All nine responses share their
first 31 generated tokens. C1 ends with token 16; two C8 responses end with
22 and six end with 16.

All 288 GPU greedy selections agree with a CPU argmax over the actual BF16
logits. At position 3102, two owners prefer 22 over 16 by 0.125; six owners
(including C1) prefer 16 over 22 by 0.125; one owner has an exact 25.625 tie
and selects 16. The tie therefore does not explain the two flips.

There are 894 all-owner head-input captures. Duplicate captures at the final
position agree exactly. Relative to C1's 1024-component head input, C8 owners
already differ in 869–911 BF16 components before the vocabulary projection.
This localizes at least part of the numerical difference upstream of the head;
it does not establish which earlier layer first diverged, whether the head
adds error, or which final token an independent reference requires. Request
identities come from the execution trace; identical responses are not used to
guess request-to-owner mappings. Numerical closure remains open.

The initial diagnostic worker was accidentally paired with an uninstrumented
parent that closes stderr. A subsequent launcher retained stale executable
digests. Both attempts failed and were stopped, not counted as numerical
results. The successful run uses rebuilt diagnostic parent/bridge/supervisor
and refreshed startup identities. No diagnostic edits entered production.

## Identical-prefix first-QKV follow-up

The same `3836a256` kernels were observed at unfused operation 2 in
`/tmp/lunaflux-shared-qkv.KkEoid`, again with identical 3072/32 requests at C1
then C8. The trace contains 602 input/output captures. Pairing must use
epoch/operation/row, because all-row observation emits all inputs before all
outputs; the former adjacent-pair parser incorrectly rejected this order.
The parser now accepts grouped and reversed output ordering while rejecting
missing, duplicate, orphan and owner-substituted pairs (seven regressions).

Of 1702 exactly equal-input pairs, 350 have different QKV outputs, with at
most six of 4096 components changed. Every changed pair crosses a single-token
and multi-token execution shape; none compares two multi-token shapes. This
is direct first-layer projection evidence, not proof that these few changes
alone cause the final-token flip. Existing high-precision dot-product work
must be extended to these actual observations before choosing a numerical
policy; forcing scalar and matrix paths to agree is not itself correctness.

Separately, the preceding head capture shows that seven C8 owners match C1
exactly at position 3071, then differ in 920 components at position 3072.
The eighth differs already at 3071. Owner identity and the observed execution
shape matter; a response's client row must not be guessed from identical text.

## Earlier mixed workload measurements

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
