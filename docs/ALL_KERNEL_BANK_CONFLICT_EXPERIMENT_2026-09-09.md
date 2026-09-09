# All-kernel bank-conflict follow-up

The all-kernel task is **not complete**. This experiment does not supersede the
accepted gate/up change in `33458b1`. No deployment was performed.

## Projection experiment: rejected for latency regression

On the RTX 5060 Ti (`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`), an experimental
compact shared-memory operand layout and explicit matrix fragments reduced
conflicts substantially but made the selected 1024-token kernels slower.
The final r5 experiment used scalar packed operand loads, after an earlier
`ldmatrix` variant also regressed. Both kept the ordered K reduction.

Paired CUDA-event measurements below are medians of three trials, 30 repeats
per trial. They are isolated kernel times, not end-to-end serving measurements.
Counter runs were separate from timing runs, with unmodified GPU clocks.

| Kernel | Reference µs | Experiment r5 µs | r5 total shared bank conflicts |
| --- | ---: | ---: | ---: |
| QKV | 415.98 | 634.61 | 37,307 |
| Output projection | 202.88 | 329.74 | 8,109 |
| MLP down | 186.38 | 367.61 | 6,680 |

All three passed bitwise comparison at token counts
`1,7,17,63,64,65,255,256,257,504,1024`. Local-memory load/store sector counts were
zero in the selected counter runs. Neither correctness nor the lower conflict
count justified accepting the latency regressions.

Instruction profiling found additional address/rearrangement and synchronization
work in the explicit-fragment implementations. Replacing `ldmatrix` with packed
scalar loads reduced QKV's executed instruction count from 96,485,376 (r3) to
80,691,200 (r5), but did not recover latency. Consequently those observations
are not a complete causal explanation of the slowdown; further per-instruction
and memory-pipeline attribution is needed before choosing another lowering.

## Attention experiment: not accepted

The experimental matrix-fragment/layout rewrite passed the independent 312 and
316 numerical oracle cases, covering context lengths through 4096. Maximum
absolute error was 0.0102608 within that existing oracle's tolerance.

However, strict old/new comparison stopped at the 33-token case: four BF16
outputs differed by one encoding unit. Earlier short cases were approximately
5–10% slower. This is not bitwise equivalence, and no successful long-context
timing or all-conflicts-zero claim is made for that rewrite.

## Disposition and remaining coverage

The experimental production-source changes were restored to their pre-experiment
state. The previously committed gate/up fix remains intact. No unrelated
working-tree changes were restored or staged.

Still needed: a non-regressing fix for QKV/output/down and attention, plus a
complete selected-path sweep of vocabulary head, normalization, sampling,
decode and short-shape fallbacks. Zero conflicts on one selected gate/up case
does not establish zero conflicts for every shape or kernel.

Reproducibility artifacts retained without overwriting:

- Remote projection experiments:
  `/run/user/1000/lunaflux-all-bank-20260909-r2` through `-r5`.
  r5 contains paired logs, generated CUDA, cubins, probes and full Nsight reports.
- Remote attention experiments:
  `/run/user/1000/lunaflux-all-attention-20260909-r1` and `-r2`.
- Local source patch: `/private/tmp/lunaflux-all-bank-experiment-20260909.patch`.
- Local attention helper snapshot:
  `/private/tmp/lunaflux-attention-matrix-fragments-20260909.mbt`.

These temporary paths are diagnostic artifacts, not a durable release archive.
