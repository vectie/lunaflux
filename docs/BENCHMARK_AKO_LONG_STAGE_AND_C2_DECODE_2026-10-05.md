# Dual-Spark long-stage attribution and measured C2 decode

## Outcome and correction

Both Sparks worked concurrently: .179 collected serving timelines and a
route-only serving ABBA; .178 collected detailed selected-prefill counters
and independently repeated the decode-chain experiment. Each GPU's work was
serialized. No production instance, runtime default, model semantics or
numerical tolerance changed.

**Correction:** pure C1 uses partitioned decode, but the preceding measured
bundle's pure C2 uses ordinary decode. The previous mixed-companion report
misidentified numeric owner 77. A numeric owner alone is not kernel identity.
The new Nsight symbols establish the distinction, and source explains it:
`paged_decode_split_prepare.mbt` defaults the blockwise partition route to
one row; an explicit measured candidate 4 can override this fallback through
`measured_attention_routes.mbt` and the existing graph owners.

The representative eight-row C2 envelope improves by 21.0% on .179 and 19.7%
on .178. A benchmark-only .179 table now measures and selects that decode
bucket. Uninstrumented C2 completion improves **5.47%**, with actual selected
kernel propagation confirmed separately. **It remains experimental:** C2
token vectors differ across routes and some candidate repeats differ too.
This is fixed-token-count timing, not strict numerical equivalence or a
production/model-quality pass. No vLLM/SGLang rerun occurred this round.

## Frozen workload and budget

Qwen3-0.6B BF16; each request has 32,512 input tokens and exactly 64 greedy
output tokens. C1/C2 use the saved varied token-ID prompts, a 2048-token
scheduler/prefill chunk, unchanged worker and AOT modules. The model's admitted
context remains 32768; these runs do not establish 64K/1M serving.

Initial budget: two profiled serving cells and two selected c322 captures.
The discovered pure-C2 selection gap justified one five-pair exact-envelope
trial per GPU, one changed-boundary sanitizer run, one four-start ABBA, and a
second two-cell timeline to verify propagation. No further kernel rewrite was
performed. GPU probes/counters are limited to 8 GiB, zero swap and finite
runtime. Serving uses the existing 64-GiB engine/2-GiB bridge launcher plus a
48-GiB no-swap outer controller and a 32-GiB available-memory reserve. Completed
serving reports check every memory sample, empty runtime stderr and terminal
drain/child closure. Both GPUs return idle.

GPU UUIDs: .179 `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, .178
`GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`, both GB10/sm121.
nvcc 13.0.88 SHA-256:
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
Selected c322 cubin:
`9aedf7709b833ad4ba05bf017d54199b33908e89d63c7fae51e73f94cf0a655d`.
Ordinary and partial/merge decode cubin:
`b4e24a7bd1014106c55f0eb215a4dff2cfa7c29e4c33ee2edb60b97fc4a5dd7a`.
Serving worker:
`dd4e8e2b5ddfb66cc6c901cf96b5ca14f12f9138841644c4fc2c4b4480928622`.

## Baseline serving timeline: prioritize by actual executed work

These are summed GPU kernel durations across **two profiled request waves**,
not uninstrumented throughput, latency percentages or time saved by removing
a stage. Full invocation start/end, graph-node/correlation IDs, requests and
token timestamps are retained alongside the aggregates.

| Stage | C1 sum, ms | C2 sum, ms |
| --- | ---: | ---: |
| c322 prefill attention | 4007.745 | 8088.557 |
| pure decode attention | 2031.841 | 5254.736 |
| full QKV ingress, prefill-sized calls | 1009.094 | 2027.321 |
| gate/up, prefill-sized calls | 501.959 | 1006.596 |
| down, prefill-sized calls | 283.206 | 572.919 |
| output projection, prefill-sized calls | 193.881 | 392.986 |
| all recorded kernels | 8883.174 | 18491.445 |

C1 has 32 prefill steps and 126 pure-decode steps across two waves.
C2 has 64 prefill steps, 124 ordinary paired-decode steps, two mixed steps
with ordinary companions and two solo-tail partitioned steps. For 28 layers,
that is 3472 paired ordinary-decode calls. The ordinary paired grid is
`8×8×1`, not a compact two-row synthetic grid. Consequently the prior compact
probe was inadequate for registering this serving bucket.

## Exact-envelope paired decode experiment

`selected_policy_probe.cu` now accepts a diagnostic-only explicit captured
query bound. `--query-bucket-bound 8` retains two active rows plus inactive
capture slots on **both** sides; it cannot be combined with a mixed, parity or
full-capacity replay mode. The pure geometry helper rejects undersized,
non-power-of-two and over-capacity bounds, with a C++ regression test.

Five alternating pairs time 30 warmed complete chains each. The alternative
includes partial **and merge**, not partial-only timing. Allocation, transfer
and correctness checks are untimed. Both use the same immutable KV and module.

| GPU / active rows / capture bound / history | Ordinary median, µs | Partitioned chain median, µs | Median paired reduction | Worst pair |
| --- | ---: | ---: | ---: | ---: |
| .179 / 2 / 8 / 32513 | 1481.316 | 1177.542 | 21.01% | 19.34% |
| .178 / 2 / 8 / 32513 | 1449.645 | 1183.265 | 19.69% | 17.01% |

Both clear the declared 3% threshold in every pair. .179 full differential
output is bitwise equal for these synthetic operands; sampled independent
FP64 oracle maximum error is 0.000237117. This does not prove bitwise equality
for real model activations. The .179 exact eight-row contract passes memcheck,
racecheck and synccheck with zero errors/hazards; raw commands and outputs are
saved. Its early generic sanitizer result caption still says mixed-chain;
the saved contract/command proves this invocation was pure decode. The adapter
caption is corrected in source.

Only .179 observations enter .179's table. The additive binder retains all
prior prefill/mixed records and adds decode candidate 1 (ordinary baseline)
versus candidate 4 (partitioned) at rows=2, query=2, context=32514. The required
table baseline is candidate 1, not candidate 3: the first bind attempt using
3 failed before serving and is preserved. No fallback cap is globally lifted.

## Uninstrumented serving and propagation

Fresh-start order is baseline, candidate, candidate, baseline. Each start
runs C1/C2 with one excluded warmup and three measured waves per cell. The
report validates identical saved request bodies and all 64-token outputs;
full individual timing/vector arrays are retained.

| Cell | Baseline completion, ms | Candidate completion, ms | Reduction | Baseline / candidate TTFT, ms | Baseline / candidate output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 4573.5 | 4591.5 | -0.39% | 3112 / 3125 | 13.994 / 13.939 |
| C2 | 9467.5 | 8949.5 | 5.47% | 4718.5 / 4747.5 | 13.520 / 14.302 |

C1 is unchanged within noise. C2 median per-request mean TPOT improves from
74.508 to 65.960 ms, while TTFT does not improve—consistent with a decode-only
change. Throughput counts output tokens, not input-plus-output tokens.

The independent candidate timeline shows 3528 partitioned partial and 3528
merge calls across two C2 waves, replacing the 3472 ordinary paired calls plus
56 pre-existing solo partials. Mixed companions remain ordinary (56 calls).
Pure-decode sum goes from **5254.736 to 4164.915 ms**, about 545 ms per wave;
uninstrumented completion improves 518 ms. This agreement supports the
decode-route attribution, rather than extrapolating a microbenchmark alone.
Candidate prefill attention is 8186.485 ms across those two traced waves, not
an optimized prefill path. Profiled timing is not substituted for the ABBA.

All baseline repeats are stable in this capture; C1 candidate repeats also
match. Candidate C2 has 41 changed positions in some same-row repeats and
41–51 positions different from baseline. Earlier margin investigation found
pre-token BF16 logit differences and ties; it is not established that this
new variation is a race or that partitioning uniquely causes it. The numerical
gate remains open, so the table is benchmark-only, not a promoted default.

## Selected prefill counters: update the hypothesis

.178 captures the exact c322 symbol, grid `63×16×1`, 128 threads, 235 registers,
at query=2048, history=28672 and one/two active rows. These synthetic probes
mirror geometry but are not the exact layer values or serving phase mix.
Detailed NCU replay durations are diagnostics, not paired timing.

| Counter | One active row | Two active rows |
| --- | ---: | ---: |
| Tensor active cycles / elapsed peak | 54.17% | 54.33% |
| Issue-active fraction | 33.49% | 33.65% |
| Eligible warps / scheduler | 0.418 | 0.412 |
| Fixed-latency wait / average warp latency | 35.26% | 35.90% |
| Long-scoreboard / average warp latency | 18.65% | 13.90% |
| Barrier / average warp latency | 3.03% | 2.37% |
| Local spilling requests | 0 | 0 |
| Source-derived excessive shared wavefronts | 0 | 0 |

These percentages use explicit warp-latency denominators, not fractions of
wall time. Fixed dependency wait is now larger than global-load dependency or
barriers. R2 executes 1.125 billion warp instructions. Source-correlated totals
include 119.669M HMMA, 123.408M FMUL, 99.101M FADD, 102.764M MOV and 64.229M
FFMA. HMMA locations receive 110,088 fixed-latency samples; this is a sampled
dependency attribution, not proof that every HMMA is redundant. Async
`LDGSTS.E.BYPASS.128` executes 14.959M times: this is not the old synchronous
c318 route. Zero source-derived excessive wavefronts does not prove every
hardware bank metric is zero.

The next falsifiable compiler experiment should expose more independent MMA
accumulation chains / fragment-consumption scheduling, while reducing scalar
softmax and ownership moves under the existing numeric contract. Do not
blindly add more copy stages, pursue only bank counters, or silently replace
strict arithmetic with fast math. Any alternative must retain the pure
schedule/ownership/effects boundary, be generated AOT and win the complete
representative attention chain before another serving trial.

## Validation and saved results

Warning-denied checks and tests pass for the changed `.mbtx` adapters/reporters
(1 chain, 2 route, 1 timeline and 1 counter-parser tests), plus the C++ geometry
regression. CUDA probe recompilation succeeds on both sm121 GPUs. This is a
scoped diagnostic phase, not a full-suite/release-validation claim.

Remote roots:

- .179 baseline timeline: `lunaflux-ako-long-stage-20261005.79EtgTQH`.
- .179 exact chain and preserved failed bind: `lunaflux-ako-pure-decode-20261005.ZxvWJtdR`.
- .179 candidate serving: `lunaflux-ako-c2-serving-20261005.t6OtchBi`.
- .179 candidate timeline: `lunaflux-ako-c2-selected-trace-20261005.RbvfPoiL`.
- .178 selected counters: `lunaflux-ako-prefill-detail-20261005.Ogk0ltFH`.
- .178 independent chain: `lunaflux-ako-c2-envelope-20261005.VB6JdI57`.

All are under the corresponding user's home. Archives are downloaded without
overwrite under `/tmp/lunaflux-ako-c2-envelope-20261005.w0kcOTGe/`, with archive
hashes and extracted per-member manifests verified:

| Archive | SHA-256 |
| --- | --- |
| baseline-trace179.tar.gz | `5d1d29e3a59bc6eb299369612e6a6e78a1a5b04e1d39c21f77d81b915ef68685` |
| chain179.tar.gz | `b310fdc4cee324735c4ee8c00b71774696e8267dc50d4240b1d7b0ad9b5f4d58` |
| serving179.tar.gz | `8c45b29a6c644ad09c8ca97fc8922093da4bcbc00460561cdaf8ff0be45c3924` |
| selected-trace179.tar.gz | `33445cbbf3e68d6629822136766c0ab2e8792ea92c7c95c63bd64de8997793d2` |
| counters178.tar.gz | `39310b2bcfdf4730fb3092132469136f7876e2116db06c77bf9df7589345cfb4` |
| chain178.tar.gz | `5847981f5c1dac70a105626c51952816ed93b16ef870886b046c722592fabefd` |

The .178 installed MoonBit runtime lacks its core library. Two interpreter
attempts and an incorrectly chosen helper executable failed before GPU work;
the exact current ARM trial binary built on .179 completed the replication.
No toolchain, driver permissions or user workloads were globally modified.

The AKO loop influenced this change by requiring runtime capture geometry,
whole-chain timings, actual kernel propagation and explicit losing/numerically
unstable results. The measured records remain offline startup inputs; no
profiler, hashing or diagnostic scan enters the token hot path.
