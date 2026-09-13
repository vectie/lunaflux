# Prefill composition integration

The five composition seams are implemented and the full Qwen serving matrix
has been retested. The end-to-end improvement is approximately 0–1%, not the
larger isolated attention improvement and not the half-baseline-time target.

- Query-owned single-storage async K/V scheduling separates K-score readiness
  from V readiness. The previous two-storage candidate remains measurable, not
  the default merely because it uses async instructions.
- Warp-uniform interior/boundary specialization shares the maximum fold.
- Projection folds share immutable fragment lifetime lowering, including the
  sibling-reuse path. Wide row products retain a compact register lifetime.
- Offline attention observations are parsed in a backend-neutral package and
  bound to the exact frontier, device, toolchain, and workload vector. They
  choose one AOT variant; this is not per-request runtime autotuning.
- Optional version-1 attention metadata is constructed once per step in
  preallocated descriptor storage and shared across heads/layers. Bundle v5
  explicitly distinguishes metadata from legacy row offsets. Use the `.mbtx`
  fused builder for metadata-enabled recipes; do not use the legacy shell
  builder's intermediate bundle as a metadata runtime.

## Measurements so far

Nsight Compute confirmed the old async regression's occupancy cost: c320's
81,936-byte shared allocation permitted one CTA/SM, versus two for c318's
49,168 bytes. Achieved occupancy was 8.32% versus 15.91%, with no eligible warp
on 86.77% versus 79.87% of scheduler cycles. These are profiling runs, not the
ordinary timing samples below. Single-storage async restores the smaller
footprint but still must compete with synchronous schedules on measured workloads.

On the RTX 5060 Ti, `metadata-r1` compared with `activation-r3` using identical
inputs and strict arithmetic. With 1,528 **total** query tokens, eight rows and
no history, paired median kernel times were approximately 134 → 112 microseconds
for synchronous query-owned attention and 134 → 123 for single-storage async.
Single-row async remained slower (approximately 469 → 500 microseconds).
All 16 comparisons were bitwise equal, with an independent sampled scalar
referee, unchanged KV, and memory/race/synchronization sanitizer passes.

These timings exclude host metadata construction and upload. End-to-end testing
must account for those costs before enabling the metadata ABI by default.

The fragment campaigns covered 45 paired shape/family cases each, all bitwise
equal, plus three sanitizer modes for each family. The sibling-reuse lookahead
variant regressed about 4% in the bare static configuration; the subsequent
compact lifetime policy removed that regression (`fragment-r3`, about
509 versus 507 microseconds at 1,528 tokens/eight rows). That small difference
is not treated as a substantial speedup.
Bare static projection recipes are not the production tuning configuration.
Paired five-trial medians at 1,528 total query tokens/eight rows:

| Kernel | Previous microseconds | New microseconds |
| --- | ---: | ---: |
| QKV | 342.97 | 329.77 |
| Output projection | 176.04 | 172.53 |
| Gate/up | 509.16 | 507.33 |
| Down | 256.91 | 250.44 |
| Head (unchanged direct-register path) | 740.88 | 741.22 |
| Query-owned synchronous attention, no history | 133.69 | 112.37 |

These are different kernels and workload domains, not additive serving savings.

The real measured mixed-history table was consumed successfully by the exporter
and selected c322. Its eight-shape aggregate was approximately 6.86 ms versus
7.13 ms for c318. This profile includes 4,096-token history; its winner must not
be generalized to no-history prefill. The fresh-serving comparison uses c318.

## Full-runtime integration fixes

- Removed unused CUDA metadata bindings that failed warning-denied AOT builds.
- Composed the metadata suffix with the existing ingress-evaluation arguments.
- Added MoonBit `.mbtx` v5 builder/materializer/assembler paths. The legacy
  shell builder's intermediate bundle is not used as a metadata runtime.
- Corrected the worker bootstrap dispatch: v5 was erroneously sent to the
  legacy parser despite the reusable parser already supporting it. A regression
  now exports metadata and exercises the bootstrap parser selection.

The failed startup was diagnosed as
`FusedArtifact(InvalidFusedArtifact(Manifest))` in an isolated instrumented
worker. The diagnostic worker was not used for timing or deployed. The fixed
release runtime successfully served every matrix request.

The exact clean committed tree (`5f5d75de`) passed 3,018/3,018 native tests.
The main working tree, including unrelated uncommitted model work, passed
3,739/3,739; that larger count is not attributed to this commit. Native check
and `moon info` also passed. Kernel differential and sanitizer results are
described above. No production deployment was performed.

## Full-serving results

RTX 5060 Ti; Qwen3-0.6B BF16; identical repeated/truncated token-ID input,
greedy output with EOS ignored, prefix reuse disabled, 2,048-token prefill
chunks. Each cell has one warmup and five measured trials. Engines ran
sequentially on the same GPU. Numbers are mean output tokens/s, including
prefill, not isolated decode throughput.

| Input/output tokens | Concurrency | Previous | New | Change |
| --- | ---: | ---: | ---: | ---: |
| 512/64 | 1 | 210.69 | 211.23 | +0.26% |
| 512/64 | 8 | 1021.56 | 1026.88 | +0.52% |
| 512/64 | 16 | 1353.79 | 1366.44 | +0.93% |
| 1528/32 | 1 | 148.44 | 148.16 | −0.19% |
| 1528/32 | 8 | 348.49 | 351.65 | +0.91% |
| 1528/32 | 16 | 381.46 | 385.37 | +1.02% |
| 3072/32 | 1 | 101.59 | 101.66 | +0.07% |
| 3072/32 | 8 | 166.32 | 167.91 | +0.96% |
| 3072/32 | 16 | 174.84 | 176.28 | +0.83% |
| 4096/64 | 1 | 110.69 | 110.73 | +0.04% |
| 4096/64 | 8 | 193.27 | 194.37 | +0.57% |
| 4096/64 | 16 | 155.15 | 156.17 | +0.66% |

For 4096/64 C16, mean TTFT changed 1930.14 → 1914.69 ms. Single-request
TTFT was essentially unchanged (207.8 → 207.6 ms). The remaining long-input
and concurrency gap is therefore not resolved by this composition work.
Sub-percent differences need interleaved repetitions before being treated as
robust gains. No new vLLM/SGLang run was made in this campaign.

The first control run exhibited an approximately one-second fixed TTFT delay.
It is preserved but excluded from the comparison. HTTP tracing showed no
`Expect: 100-continue` and about 22 ms to the first token on a subsequent old
binary run. Fresh untraced control/current runs then returned normal timings.
The transient delay's root cause was not established; it is not credited as a
compiler fix or removed arithmetically from measurements.

All 500 measured requests per version completed with the expected token counts.
There were no novel output sequences relative to the repeated control. At
3072 tokens/C8 and C16 both versions produced two existing sequence variants;
ordinal request matching was 462/500, so this is not a claim of bitwise
end-to-end equivalence or resolution of the earlier last-token discrepancy.
Streaming inter-token intervals in the JSON include scheduling and concurrent
prefill delays; they are not isolated decode GPU timings.

Timing runtime source: `e5f5211a`; kernel artifacts: `3ef92622` (kernel generators
unchanged between these commits). Subsequent `5f5d75de` contains fixture/format
changes only. The original deployment and failed-run outputs remain preserved.

Kernel results and original Nsight reports were downloaded to
`/tmp/lunaflux-composition-results.rhGfam/kernel-results.tar.gz`; local and remote
SHA-256 both equal
`98d87d2165ab418b1adf464706cfff1513de190f03a7896c642d6dc789dbe353`.

Full-serving results, source archives, driver scripts and startup diagnosis:
`/tmp/lunaflux-composition-final.Tg8IeW/composition-final-results-r1.tar.gz`.
Downloaded SHA-256 matches the remote archive:
`782d60abe17564faa2872bada4849914ea6dc1281ae187b90a12247bb0d4c6d1`.
The remote campaign is
`/run/lunaflux-toolchain-4896771-20260913/composition-3ef92622-r2`;
the comparable runs are `current-fixed` and `control-repeat`.
`deployment.complete.sha256` supplies the complete inventory after the recovered
release-bind copy; the earlier partial inventory was preserved, not overwritten.
