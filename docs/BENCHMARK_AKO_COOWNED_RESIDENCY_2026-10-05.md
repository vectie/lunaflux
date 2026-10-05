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

## Follow-up: resource target restored, no robust selection win

The one additional experiment extends the existing CUDA launch-bound policy to
explicit coownership only when all 16 launched consumer groups are active and
each consumer owns one row tile of each product. It retains the caller's
register-budget guard, excludes smaller workgroups and larger live row products,
and leaves the down entry point unqualified. This is a physical-topology policy,
not a model-name or token-count special case. No new IR layer, runtime tuning,
allocation or request-path validation was introduced.

Changed implementation: `kernels/luna_cuda_projection_aot/source_sibling_resources.mbt`.
Regression: `sibling_resources_wbtest.mbt` covers budgets 32/63/64/128/255,
fully occupied C16, larger row products, smaller consumer maps and unchanged
down qualification. Actual source hash:
`6c6f89472bf902923f8c883f8bf49d7b00731e9cda8cb1f2b8d6af5c6e6810cf`.
The selected baseline's source hash remains unchanged.

The emitted target propagates to the compiled artifact: registers drop from 72
to 60 with zero stack/spill loads/stores, and static shared storage stays 24576
bytes. Fresh paired counters confirm restored residency; this is not inferred
solely from `__launch_bounds__`.

### All follow-up whole-chain paired reductions

| Tokens | Cached | Distinct layers |
| --- | ---: | ---: |
| 32 | −1.09% | −0.20% |
| 129 | −1.56% | −1.39% |
| 512 | +1.31% | +0.49% |
| 2048 | +0.15% | +0.39% |

None clears the all-pairs 3% rule. Five more distinct-layer/2048 processes:

| Repeat | Median paired reduction | Worst pair | Decision |
| --- | ---: | ---: | --- |
| 0 | +2.46% | −0.88% | inconclusive |
| 1 | +0.30% | −3.23% | inconclusive |
| 2 | +0.74% | −3.82% | inconclusive |
| 3 | +0.11% | −4.54% | inconclusive |
| 4 | +2.17% | −6.51% | inconclusive |

The severe unbounded-coownership regression is largely removed, but that is
not improvement over the selected baseline. The candidate remains unselected.
No new end-to-end or vLLM/SGLang claim follows from these measurements.

### Fresh counters after fixing the resource target

| Metric | Paired baseline | Resource-bounded coowned C16 |
| --- | ---: | ---: |
| Executed warp instructions | 93020160 | 69623808 |
| Registers/thread | 58 | 60 |
| Waves/SM | 16 | 16 |
| Active warps/scheduler active cycle | 7.88 | 7.85 |
| Eligible warps/scheduler active cycle | 0.77 | 0.53 |
| Issue-active | 31.41% | 23.89% |
| Average warp latency/instruction issued | 25.07 cycles | 32.84 cycles |
| Barrier ratio per issue-active | 7.73 | 10.96 |
| Long-scoreboard ratio per issue-active | 2.88 | 7.44 |
| Tensor activity, elapsed-normalized | 34.50% | 34.19% |
| Local spilling requests | 0 | 0 |

Instructions fall **25.15%**, but issue-active falls **23.94%** and eligible
warps fall **31.17%** despite matching residency. Tensor activity is effectively
unchanged. These counters explain why instruction removal does not translate
into a robust latency win. The increased normalized stall ratios are not proof
that absolute barrier time increased by the same percentage: their issue-active
denominator changed substantially.

The split renderer consumes one weight fragment across two row folds; the
coowned renderer consumes two weight products in one row fold, through the same
packed-fragment lowering. Both retain the ordered reductions and shared operand
ring. After correcting residency, remaining differences concern ready-work and
operand/fragment dependency scheduling, not the removed epilogue alone. The
counter capture does **not** pinpoint an individual load instruction; a next
experiment needs source-correlated dependency/SASS isolation before changing
that pipeline. Neither occupancy nor instruction count alone should select it.

### Follow-up validation and archive

All 8 workloads and 5 confirmation processes pass full output/workspace bitwise
comparison and the sampled scalar oracle (reported error 0). Tail 129 again
passes memcheck/racecheck/synccheck with zero leaks. Projection tests **107/107**
and warning-denied affected-package check pass with the existing migration
warning exclusions. `moon info --target native` completes with zero errors and
4250 preexisting whole-tree migration warnings; this is not a warning-clean
whole-tree release claim. No unrelated dirty files were staged.

Remote: `/home/wlc004s/lunaflux-ako-coowned-bound-20261005.3aI9tQz8`.
Local: `benchmarks/qwen3_comparison/results/ako-coowned-bound-20261005.5dMLy2dr/measurement.tar.gz`.
Archive SHA-256: `06b39ad8c19a00f0a9094e4ca3976708248b3dd7fc52a329f819302ecfd788a3`.
All **164 manifest files** verified locally. Same serialized GPU, memory reserve
and configured finite systemd bounds as the first experiment. The finite budget
of one ownership candidate plus one counter-driven resource follow-up is
complete; no production selector change or additional blind sweep is made.
