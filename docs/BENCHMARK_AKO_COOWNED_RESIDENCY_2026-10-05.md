# AKO coowned siblings: residency cliff and resource-policy follow-up

## First experiment: rejected

Explicit coownership at 64 token rows / 16 consumers preserves the selected
primary grid (1536 CTAs at 2048 tokens) and 512 threads per CTA. Each consumer
owns one row tile of both products instead of two row tiles of one product.
The ordered reductions, strict numerical map and down kernel are unchanged.
The shared epilogue handoff disappears; dynamic shared storage falls from
16384 bytes to zero. Static shared storage remains 24576 bytes.

This is an offline alternative, **not selected serving code**. No new serving
or reference-framework throughput measurement is claimed.

The existing generic `CoownedSiblingValues` physical plan generates the source;
there is no model-name branch or source-text substitution. The AKO exporter and
runner now support the two-record experiment and reuse the frozen probe/control
instead of rebuilding either. Other experiment modes retain their original
four-record exports. See the preceding
[ownership experiment](BENCHMARK_AKO_SIBLING_OWNERSHIP_2026-10-05.md) for the
probe, numerical contract, fixed GPU/tool identities and timing methodology.

Source hash: `6b033e353b3871c8d5037130872710c0c41197130ef2a02c47cc9cfbe41b45fd`.
Baseline source remains `c6b7f77a4f6c5e6bcd47d0f2c7a24ffcfa947c8d40d36a5217dc1971d2b96397`.

### Complete-chain paired time reductions

Positive is faster; these are medians of paired reductions, not ratios of
independent medians. Every workload regresses. Token vector is
`[32,129,512,2048]`, row count 32, cached or 28 distinct layer-weight addresses.

| Tokens | Cached | Distinct layers |
| --- | ---: | ---: |
| 32 | −2.85% | −0.83% |
| 129 | −10.92% | −13.19% |
| 512 | −23.83% | −26.11% |
| 2048 | −42.81% | −33.73% |

Five additional processes at distinct-layer/2048 (25 alternating pairs) give
median paired reductions of −25.43%, −41.42%, −41.94%, −34.98%, −41.00%.
Thus the initial regression is confirmed, not promoted as a noisy win.

### Fresh selected-primary hardware counters

Separate Nsight replay, 2048 tokens. Replay time is not unprofiled benchmark
time; normalized stalls cannot be added as end-to-end milliseconds.

| Metric | Baseline | Coowned C16 |
| --- | ---: | ---: |
| Executed warp instructions | 93020160 | 71811072 |
| Registers/thread | 58 | 72 |
| Waves/SM | 16 | 32 |
| Active warps/scheduler active cycle | 7.89 | 3.98 |
| Eligible warps/scheduler active cycle | 0.80 | 0.28 |
| Issue-active | 32.58% | 18.16% |
| Barrier ratio per issue-active | 7.85 | 7.02 |
| Long-scoreboard ratio per issue-active | 2.62 | 4.37 |
| Tensor activity, elapsed-normalized | 40.79% | 28.00% |
| Local spilling requests | 0 | 0 |

Instructions fall 22.8%, but the launched geometry does not preserve residency.
Registers cross the 64-register budget for two 512-thread blocks on this GPU;
Nsight's waves double and active warps halve. Long-load waits increase.
This is not spilling and not evidence that the removed epilogue was free.

### Concrete source-policy gap and next bounded experiment

`source_sibling_resources.mbt` currently emits `__launch_bounds__(512,2)` only
when `physical.sibling_reuse` is true. Explicit coownership is excluded even
when every launched consumer is active. Consequently, the baseline has a
two-block compiler target while this candidate does not.

After preserving this rejected experiment, test exactly one follow-up: extend
that existing CUDA resource policy to the fully occupied coowned topology,
while retaining the caller-budget check, small-workgroup exclusions and
unmodified down entry point. Verify actual register use, spilling, timing and
sanitizers. A qualifier is a compiler target, not a residency guarantee; do not
assume it improves performance. Keep production selection unchanged pending a
robust complete-chain result and serving propagation test.

### Validation and retained results

All 8 initial workloads and 5 confirmation processes pass full bitwise
output/workspace comparison and the sampled scalar oracle (reported error 0).
Tail 129 passes memcheck, racecheck and synccheck; zero leaks. Projection tests
106/106, warning-denied affected-package check and AKO trial tests 3/3 pass
using the preexisting migration warning exclusions. This does not resolve the
earlier serving-worker sanitizer startup blocker.

Remote: `/home/wlc004s/lunaflux-ako-coowned-20261005.q88Z7MyE`.
Local: `benchmarks/qwen3_comparison/results/ako-coowned-20261005.HTjzGFAd/measurement.tar.gz`.
Archive SHA-256: `76cdf9f9784fd46005fbf86464c92649fd396f719c135f60e74ce202dc5356ce`.
All **164 manifest files** verified after download. GPU work serialized;
systemd units configured MemoryMax=8G, MemorySwapMax=0 and finite runtime limits;
timing probes preserve the 32 GiB MemAvailable reserve.
