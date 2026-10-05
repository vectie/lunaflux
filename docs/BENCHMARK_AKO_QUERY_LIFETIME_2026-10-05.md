# Dual-Spark query-lifetime/consumer ablation, 2026-10-05

## Outcome

Keep the frozen approximate-exp2 c30322 schedule. Partial retention and joint
consumption reach the executable, but none clears the paired 3% improvement
threshold on either Spark. These are attention-kernel measurements, not new
end-to-end token throughput or a vLLM/SGLang comparison.

Both `.178` and `.179` ran timing concurrently with identical compiled inputs.
After timing, `.178` ran sanitizers while `.179` captured counters. Compilation
was performed once on `.179`, not during GPU timing. This is the follow-up to
[the larger-tile/full-retention experiment](BENCHMARK_AKO_DUAL_SPARK_SCHEDULE_2026-10-05.md),
not a new implementation of retention or joint consumers.

## Experiment and source correction

The immutable frontier already contains these alternatives:

| Label | Candidate | Query dimensions retained | Consumer |
| --- | ---: | ---: | --- |
| baseline | 30322 | 0 | ordinary |
| retain64 | 70644 | 64 | ordinary |
| joint64 | 80644 | 64 | joint right columns |
| joint128 | 80645 | 128 | joint right columns |

All use Q64/K64, the split single-slot async pipeline and explicit
`approx-base2-f32-v1`. The numerical fold/order is unchanged. Workloads use
2,048 active queries distributed across one or two requests and histories of
28,672 or 8,192 tokens per request. Runtime bucket geometry is retained:
2,048 bucket tokens, 32 bucket rows, grid 63×16×1, block 128.

Budget: three alternatives × three workloads × two machines × five
alternating timing pairs, plus three matched counter captures and nine
irregular-tail sanitizer checks. All 18 timing cells completed.

The first export failed **before GPU work**: the complete joint consumer
emitted an unused `lf_matrix_mma_key` helper, rejected by CUDA's warning-denied
AOT build. Commit `40db806c` makes helper emission follow reachability in the
existing immutable lifetime plan. Partial/ordinary consumers retain the helper;
complete joint consumption does not. Value-consumer emission is unchanged.
This is a general lowering fix, not a model branch or a runtime check.

Only the patched exporter was rebuilt in a new source copy; the runtime,
original source and baseline cubin remained frozen. The failed attempt was
preserved separately; the corrected trial used a new non-overwriting directory.
Each corrected candidate compiled twice to identical cubins with CUDA 13.0.88,
sm121, the unchanged precise arithmetic flags and `--Werror all-warnings`.

## Unprofiled paired times

Times are median microseconds. The final column is the median paired
`candidate/baseline - 1`; it is not computed from the two aggregate medians.
Negative values indicate less time, but are not necessarily a robust win.

| Host | Alternative | Requests / history | Baseline µs | Candidate µs | Paired time change |
| --- | --- | --- | ---: | ---: | ---: |
| .178 | retain64 | 1 / 28672 | 7189.96 | 7343.90 | +2.27% |
| .178 | retain64 | 2 / 28672 | 7142.73 | 7342.32 | +2.86% |
| .178 | retain64 | 2 / 8192 | 2113.49 | 2140.90 | +1.44% |
| .179 | retain64 | 1 / 28672 | 7405.10 | 7629.56 | +2.70% |
| .179 | retain64 | 2 / 28672 | 7380.69 | 7619.71 | +3.10% |
| .179 | retain64 | 2 / 8192 | 2169.14 | 2209.08 | +1.87% |
| .178 | joint64 | 1 / 28672 | 7235.19 | 7454.33 | +1.99% |
| .178 | joint64 | 2 / 28672 | 7162.58 | 7265.42 | +1.43% |
| .178 | joint64 | 2 / 8192 | 2119.26 | 2114.27 | −0.30% |
| .179 | joint64 | 1 / 28672 | 7399.78 | 7504.30 | +1.55% |
| .179 | joint64 | 2 / 28672 | 7429.68 | 7430.62 | +1.04% |
| .179 | joint64 | 2 / 8192 | 2185.02 | 2180.73 | −0.17% |
| .178 | joint128 | 1 / 28672 | 7327.39 | 7435.47 | +2.98% |
| .178 | joint128 | 2 / 28672 | 7173.53 | 7414.84 | +2.79% |
| .178 | joint128 | 2 / 8192 | 2122.75 | 2148.17 | +1.50% |
| .179 | joint128 | 1 / 28672 | 7498.44 | 7608.04 | +1.66% |
| .179 | joint128 | 2 / 28672 | 7440.99 | 7578.32 | +1.77% |
| .179 | joint128 | 2 / 8192 | 2193.27 | 2209.80 | +0.78% |

The two slightly faster joint64 cells are inconclusive: their five pairs do
not consistently improve. The other 16 cells are regressions. No alternative
was propagated into serving.

## What changed in the executable

Matched `.179` Q2048/R2/H28672 captures show:

| Metric | Baseline | retain64 | joint64 | joint128 |
| --- | ---: | ---: | ---: | ---: |
| Registers/thread | 234 | 247 | 242 | 254 |
| Dynamic shared bytes/CTA | 49,168 | 49,168 | 49,168 | 49,168 |
| Resident CTAs/SM | 2 | 2 | 2 | 2 |
| Active warp percent | ~15.89% | 15.87% | 15.88% | 15.87% |
| Executed warp instructions | 936,583,040 | 961,878,912 | 940,394,368 | 945,537,728 |
| Non-transposed LDSM | 37,396,480 | 33,665,024 | 33,665,024 | 29,933,568 |
| HMMA | 119,668,736 | 119,668,736 | 119,668,736 | 119,668,736 |
| Async LDGSTS | 14,958,592 | 14,958,592 | 14,958,592 | 14,958,592 |
| Workgroup barriers | 2,808,832 | 2,808,832 | 2,808,832 | 2,808,832 |
| IADD3 | 10,828,544 | 18,293,504 | 10,828,544 | 8,685,440 |
| LOP3 | 23,710,720 | 26,511,360 | 23,706,624 | 24,457,216 |
| NOP | 25,242,624 | 39,266,304 | 30,852,096 | 40,201,216 |

Partial ordinary retention saves 3.73M shared matrix loads but adds 25.30M
total instructions (+2.70%). A smaller retained prefix alone does not avoid
the backend cost seen with complete ordinary retention.

Joint64 removes that additional IADD3/LOP3 work and limits the total increase
to 3.81M instructions (+0.41%). Its reuse is real, but it still adds 5.61M
executed NOPs and about 0.94M MOVs. Joint128 removes more shared loads and
address additions, yet adds 8.95M total instructions (+0.96%), including
14.96M additional NOPs. All keep two resident CTAs and report zero local bytes;
an occupancy collapse or spilling cannot explain these regressions.

NOP is an executed opcode, **not** a direct stall-cycle measurement. This
supports investigating instruction scheduling/live-range interactions, not
assigning the whole timing gap to NOPs or deleting correctness synchronization.
The captures also retain significant wait/scoreboard contributions; their
percentages have changing denominators and cannot be added into a causal time
decomposition. One instrumented joint64 launch was slightly faster; the
unprofiled five-pair results on both devices remain the selection criterion.

Next bounded experiment should change the epoch-local consumer scheduling
without retaining more query state across epochs, then compare emitted SASS
and paired time. Another retention-size sweep is not justified by these results.
The semantic and lifetime plans remain pure; hardware instruction realization
belongs in CUDA lowering. No extra IR layer, model-specific rule or hot-path
validation was added.

## Correctness, limits and retained results

All 90 timing pairs were bitwise equal to the frozen approximate-law baseline;
sampled FP64 oracle checks stayed below the fixed 0.003 ceiling. This does not
establish whole-model quality or equivalence to the strict numerical law.
All three candidates passed memcheck, racecheck and synccheck on Q129/R2/H128
with zero errors/hazards. Both devices were idle after completion.

Timing jobs were bounded to 16 GiB, zero swap, 64 tasks and 600 seconds, with
a 32 GiB MemAvailable reserve. The minimum recorded availability was
121,941,580 KiB on `.178` and 121,503,904 KiB on `.179`.

Local root: `/tmp/lunaflux-ako-query-lifetime-20261005.V0Rr3gH7`.
Remote roots:

- `.178`: `/home/wlc003s/lunaflux-ako-query-lifetime-20261005.oSWSiJcq/experiment-v2`.
- `.179`: `/home/wlc004s/lunaflux-ako-query-lifetime-20261005.Cn0gTInk/experiment-v2`.

Downloaded archive hashes matched; all 105 `.178` and 3,023 `.179` measurement
manifest entries verified locally:

- `measurement178.tar.gz`: `ce3962a6fa1e9978ad6f4f7e9e197d166c5c9e2b9f5187746183ce08277e5af7`.
- `measurement179.tar.gz`: `0b633d9c24a6f50f6ea7f27cc44f154f7b52f84166725d59179fd28066855ff2`.
- Preserved pre-GPU failure archive: `5228ffa88abb039a6b128b118f15022a057b6f23b3e3887135147c6703440064`.

Cubins: baseline `66881bc0055ce4c4340e6001e7746d0abd64a8b7f9a02b741e6b87463f0e2922`,
retain64 `722fe97b6adb085a8750da7f1b8d01cae7be9979d788f833618d4a8e3751cbb9`,
joint64 `d7c15abe38cc85740e0ffb45dae2bd5e5dd297dd70f7b1d45c2fe3dd28364ef1`,
joint128 `09a1d7dfa24bf184b099654edb0e2b163f08bd3fe34dd5b4b68461b62ec7004f`.

The emitter package passed 84 native tests and scoped check with existing
dependency warnings 20/79/29 excluded; the full repository is not claimed
warning-clean. Package info produced no public API change. Both `.mbtx`
adapters passed native warning-denied check and their regression test without
exclusions. The generalized adapter reuses the two experiment kinds instead
of copying benchmark orchestration. Production selection remains unchanged.
