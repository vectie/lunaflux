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

## Current selected-graph repeat closure

`/tmp/lunaflux-fixed-current.WQZvj4` rebuilds the current production sources
from `3836a256` with only execution tracing and fixed-graph repetition added.
The production directories have no changes between that revision and
`c96d4b13`; subsequent commits changed diagnostic tools and documentation.
The diagnostic parent, bridge, supervisor and worker were rebuilt together.

The four input/output vectors (512/64, 1528/32, 3072/32, 4096/64), C1/C8/C16,
and six repetitions per cell complete. Every emitted token's actual output
row is read before and after repeating the same selected execution graph,
without advancing the request. All 28,800 comparisons, totaling 8,751,513,600
logit bytes, are identical. This count equals the complete requested output
count, not merely a nonempty sample. The owned service was stopped afterward.
All 4,116 observed execution markers select captured graphs; none selects
eager execution.

This closes current selected-partial-runtime repeatability for this matrix.
It does not close cross-batch numerical acceptance, full-fusion qualification,
or all compiler integration. The repeated work and readbacks make its printed
throughput diagnostic-only, not a replacement for uninstrumented E2E timing.

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

FP64 sequential dot references over the actual BF16 inputs and numeric tensors
`lunaflux.tensor.v1.2`, `.3`, `.4` now cover all 880 changed-component
observations in the 350 pairs. Single-token output is closer in 856
observations, matrix output in 24, with no equal-error observations. These are
pair observations, not 880 independent samples. Maximum absolute error across
both paths is 0.00195430067833513; maximum error divided by the sum of absolute
products is 0.0008174184027177878. The latter is not a declared acceptance
threshold. Neither path is uniformly more accurate, and making them bit-equal
is not justified by these results. The next causal experiment must isolate
propagation through later operations, rather than classify the final token
from this first projection alone. The reference tool now preserves explicit
row identities; ambiguous legacy multi-row inputs fail instead of guessing.

The follow-up diagnostic plumbing now also preserves operation identity in
each changed-output pair. The FP64 tool matches epoch/row/operation, so captures
from another operation at the same row cannot be mistaken for this projection.
A single supplied tensor set is restricted to one operation; mixed-operation
differences are rejected instead of applying the wrong weights. Regressions
cover explicit multi-operation captures, missing operation identity and mixed
tensor attribution. Existing single-operation reports remain valid. This fixes
multi-layer observation plumbing, not the unresolved propagation question.

### Same-run propagation boundary capture

`/tmp/lunaflux-propagation.LBpwrF` observes inputs of operations 2, 6, 9, 12
and 282 in one C1/C8 identical-input 3072/32 run, retaining the same `3836a256`
kernel artifacts for diagnosis. All nine responses contain 32 tokens; their
first 31 tokens match. C1 ends with 16; two C8 responses end with 22 and six
with 16. The server stopped and released the GPU. This eager/readback run is
not a throughput measurement or final-current-source qualification.

At position 3072, compared with C1:

| Observed input | Changed BF16 components | Maximum absolute delta |
| --- | ---: | ---: |
| First QKV (2) | 0 | 0 |
| First output projection, after attention (6) | 5 | 0.000030517578125 |
| First MLP (9) | 3 | 0.0009765625 |
| Second-layer QKV (12) | 127 | 0.0009765625 |
| Final head (282) | 920–928 | 1 |

There are 2399 captures and 2053 baseline-owner comparisons. Repeated launches
at the same baseline operation/position have identical captured words; they
are observations, not independent samples. At position 3071, operations
2/6/9/12 still match while some head inputs already differ. Thus the numerical
divergence expands through the network and is not confined to the head. This
does not isolate accumulated KV differences from attention arithmetic or prove
which operation violates its numerical contract. Further causal isolation is
still required; no blanket bit-equality policy is inferred.

Downloaded logs and summaries:
`/tmp/lunaflux-propagation-download.LOU99r/results.tar.gz`, matching remote SHA-256
`52b06a22e75456b1d5e27b4e8a0408f2f9935b7e18fda50eeeee3e0c0734e434`.

### Latest whole-worktree regression

After `b9440271`, `moon test --target native --deny-warn` passes all 3825
tests in the local working tree, which includes unrelated uncommitted changes.
This is not the clean-commit test count. A separate clean Linux extraction of
`b9440271` is under `/tmp/lunaflux-integrated-b9440271.RgOXzJ`, source archive
SHA-256 `eb456db2a76bcd76118c4aa1da4ba151b72ce44c1dcd0e0f18849bcf2a0aea3b`.
Its interface generation, format check and native warning-denied check pass;
all 3104 clean-commit tests and all seven release-entry builds pass.

### Current-source uninstrumented matrix: b9440271

Fresh candidate export, base/fused kernel compilation, release binding and
runtime materialization passed in the same clean tree. The driver preparation
initially replaced only the first old path occurrence; the corrected `driver-r2`
uses complete replacement. The failed driver did not run GPU kernels.

One warmup and five measured trials completed for every cell below. Every
response has the requested output length and the runner stopped its processes
and released the GPU. Median output tokens/s:

| Input/output | C1 | C8 | C16 |
| --- | ---: | ---: | ---: |
| 512/64 | 208.469 | 1028.112 | 1385.656 |
| 1528/32 | 150.235 | 371.014 | 409.600 |
| 3072/32 | 107.383 | 186.453 | 197.303 |
| 4096/64 | 117.002 | 216.399 | 230.527 |

The previous 3836a256 4096/64/C16 result was 230.319 tokens/s: this difference
does not establish a speedup. No competitor was rerun. Within-cell comparisons
still find final-token-index-31 differences for 3072/32 C8 and C16; the other ten
cells have none. Counts depend on which first response becomes the cell-local
reference, so they must not be interpreted as an error-rate change. This closes
the fresh uniform end-to-end measurement for the refactor, not numerical
acceptance, mixed-workload validation or all remaining compiler integration.

Results and clean validation logs were downloaded to
`/tmp/lunaflux-b9440271-download.1hvzzG/results.tar.gz`; local and remote SHA-256
agree: `74f0259bf2ad1a955be302dead35306f84474c04cbe98585dbd3c223b9374b74`.

### Current-source mixed workload: b9440271

The same uninstrumented runtime completed all seven mixed workloads at C8/C16,
one warmup and three measured repetitions, rotating case order. Arrival lag
p95 was at most 1 ms. All requested output lengths and token timestamp counts
match; the owned server stopped and released the GPU.

| Case | C8 median wall ms | C16 median wall ms | C16 TTFT p50/p95 ms |
| --- | ---: | ---: | ---: |
| Uniform shared | 1221 | 2184 | 692/1279 |
| Uniform distinct | 1219 | 2192 | 696/1279 |
| Unequal input | 1570 | 2429 | 900/1421 |
| Unequal input/output | 1768 | 2734 | 729/1414 |
| Uniform staggered | 1226 | 2193 | 524/912 |
| Unequal staggered | 1791 | 2767 | 573/1123 |
| Unequal reversed/staggered | 1767 | 2742 | 643/1141 |

Comparing identical request bodies with the earlier mixed run in
`/tmp/lunaflux-current-eb927891.zY0UED/physical/diversity/lunaflux` covers 504
requests / 32256 output tokens. Eight requests differ across runs; repeat
comparisons within the old run find five differing requests, and within this
run six. First differing indices include 22, 32 and 102 in unequal-output
workloads, so numerical variability is not restricted to the uniform matrix's
last token. These are observed differences, not independent-oracle error rates
or proof of a regression. Current mixed measurement is complete; numerical
acceptance remains open.

Archived results: `/tmp/lunaflux-b9440271-diversity-results.tar.gz`, SHA-256
`492d234e33e2bb29d1e39650021454ec3cfdc8c4d981fcf02a6fb0e533c88c78`.
The downloaded copy at
`/tmp/lunaflux-b9440271-download.1hvzzG/diversity-results.tar.gz`
has the same verified digest.

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
