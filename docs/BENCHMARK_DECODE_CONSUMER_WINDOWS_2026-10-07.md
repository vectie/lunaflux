# Decode consumer windows and the remaining concurrency gap

The current C16 workload spends most of its time executing GPU kernels. A
bounded attempt to shorten decode value-operand live ranges reduced registers
but made every tested control slower. It is rejected, and its production IR
and lowering prototypes have been removed. The production compiler and serving
selection are unchanged by this experiment.

The durable changes are diagnostic: replay the actual launch envelope, require
a byte-identical rebuild of the executed baseline, and optionally require exact
output bits in semantics-preserving AKO trials. A resource-policy regression
test ensures that fewer registers at unchanged residency cannot outrank faster
measured completion. This does not close the remaining framework gap.

## Exact C16 serving attribution

Qwen3-0.6B BF16, 4096 input tokens, 64 output tokens, concurrency 16; Spark .179,
GB10 sm121 with 48 SMs. UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
The source trace is the measured wave in the previously verified common-route
campaign, excluding startup and warmup. This is not a new unprofiled benchmark.

| Measured wave component | Time ms |
| --- | ---: |
| Client completion | 4532 |
| GPU activity union | 4401.70 |
| No GPU activity within the client window | 130.77 |
| Decode attention and merge, summed kernels | 2119.39 |
| Prefill attention, summed kernels | 500.70 |
| Other kernels, summed | 1783.04 |

Kernel sums can overlap and must not be added to host API durations. Decode
alone accounts for about 47% of client time; the no-activity interval is about
2.9%. Prefill-only improvements therefore cannot remove most of this C16 cost.

Ordinary decode actually uses several captured envelopes:

| Grid and block | Calls | Summed time ms |
| --- | ---: | ---: |
| 16 by 8, block 64 | 1316 | 1429.21 |
| 32 by 8, block 64 | 868 | 539.22 |
| 8 by 8, block 64 | 224 | 107.84 |
| Split partial 8 by 8 by 8, block 64 | 168 | 38.11 |

The grid-32 invocations are real mixed-step executions, not automatically
wasted standalone decode launches. Replay scope must retain both useful rows
and the captured envelope. The corrected standalone C16 counter replay uses
16 rows, envelope 16, history 4095 and grid 16 by 8.

## Executed source and counter scope

The serving module is
`318d74ff5b8ebfb5e0a97ea6daea16501a98a5ef1b67116350cf8b737266c185`.
Its composed source is
`dc9f5a1a30e0d2f969f7a9e82cdcc7b8b9a8c80f2058c698b19ea9416e9d43b6`.
The earlier intermediate compiled module has a different digest and is not
the serving baseline. Recompiling the composed source with pinned nvcc
13.0.88 reproduced the serving bytes before any transformation.

The corrected C16 replay executes approximately 41.50 million warp instructions,
including 9.21 million FADD and 8.95 million FMUL instructions. Tensor activity
is zero. It uses 148 registers, no recorded spills and at most two resident
blocks because of shared memory; the register limit alone would permit six.
Average active occupancy is 8.14%. Warp-latency contributions are 3.85 cycles
for short-scoreboard waits, 3.04 for long-scoreboard waits, 2.23 for MIO throttle
and 0.40 for barriers. These are replay counters, not end-to-end latency shares.

The first capture mistakenly retained envelope 32. Its raw grid is 32 by 8;
its original caption is preserved with a scope-correction annotation. It is
not used as the exact C16 counter authority.

## Value window experiment

The selected owned-eight blockwise fold fully unrolls its complete 32-key PV
tile. Experimental windows of four and eight keys retained ascending updates,
the same accumulator, arithmetic expressions and publication barriers. Both
ordinary and partitioned entry source blocks were transformed, but the tests
below execute the ordinary entry only. No partition-chain or serving win is
claimed.

Five alternating pairs were measured for each control. All 30 rows were
bitwise equal to baseline; maximum scalar-oracle error was below 0.000244.
The table reports independent medians, not a universal speed claim.

| Rows and envelope and history | Baseline us | Window 4 us | Baseline us | Window 8 us |
| --- | ---: | ---: | ---: | ---: |
| 16 and 16 and 4095 | 1148.43 | 1174.88 | 1151.83 | 1173.22 |
| 2 and 8 and 32767 | 1450.11 | 2058.75 | 1450.72 | 2025.76 |
| 1 and 1 and 127 | 8.23 | 11.15 | 8.23 | 10.39 |

The median paired slowdown is approximately 2.0% and 0.9% at C16/4K, 43.8%
and 39.4% at C2/32K, and 35.6% and 26.3% for the short control. Neither window
clears the required positive-effect gate.

## Why fewer registers lost

A matched baseline/window-eight capture uses two useful rows, envelope eight,
history 32767 and grid 8 by 8 for both invocations.

| Replay counter | Baseline | Window 8 |
| --- | ---: | ---: |
| Registers per thread | 148 | 141 |
| Warp instructions | 41.30 million | 46.18 million |
| Active occupancy | 5.67% | 5.59% |
| Eligible warps per scheduler cycle | 0.34 | 0.28 |
| Short-scoreboard contribution, cycles | 0.68 | 1.19 |
| Fixed-latency wait contribution, cycles | 0.65 | 0.98 |
| Long-scoreboard contribution, cycles | 0.19 | 0.12 |
| Barrier contribution, cycles | 0.09 | 0.05 |
| Source-correlated excessive shared wavefronts | 0 | 0 |

The scalar arithmetic counts are unchanged. Looping introduces additional
branches, index updates, predicates and scalar probability loads. The hottest
candidate waiting instructions are BF16-unpacking IMAD operations following
shared loads. Instruction count rises about 11.8%, and shared-load/fixed-latency
waits increase while eligible warps fall. Lower registers do not increase
residency because shared memory already sets the limit. This falsifies the
specific proposal that a shorter sequential PV window improves these controls;
it does not establish that all bounded read-ahead implementations are slow.

Further work must preserve useful operand independence while reducing scalar
supporting work. Removing barriers or globally enabling FMA without the
corresponding ownership and numerical contract is not a valid substitute.
The exact mixed-step decode and split/merge chains remain separate admission
and whole-serving measurements, not extrapolations from this ordinary replay.

## Reproduction and retained evidence

Automation is MoonBit-only in
`benchmarks/gpu_pipeline/decode_consumer_window_20261007.mbtx`,
`decode_value_windows_trial_20261007.mbtx` and
`finish_decode_consumer_20261007.mbtx`. Rejected rendering stays in offline
benchmark automation; it is not a public IR option or request-path transform.
The generic selector already ranks eligible observations by measured latency;
the new regression test locks that behavior for unchanged residency.

Remote evidence roots are:

- `/home/wlc004s/lunaflux-decode-window-trial-20261007.V3vXcVWY`
- `/home/wlc004s/lunaflux-decode-consumer-traced-20261007.fXBvspgF`
- `/home/wlc004s/lunaflux-decode-window-counter-20261007.kjlBjIPR`
- `/home/wlc004s/lunaflux-decode-consumer-20261007.SVotHjpY` for the mismatched capture.

GPU work was serialized, diagnostic processes were capped at 8 GiB with no
process swap, and counter capture monitored the 32 GiB host reserve every
500 ms. Paired trials checked the reserve before and after each cell. No
reference server was rerun in this experiment, and no production deployment,
default selection or numeric law changed.

All four archives were downloaded without overwrite and verified locally,
including every `FILES.sha256` entry. The local copy is
`/tmp/lunaflux-decode-consumer-evidence-20261007.X3ylztev`.

| Archive | SHA256 | Verified members |
| --- | --- | ---: |
| Trial | `fb598bf2223ea01cf46509e587d3a7080abe85bbb20f0974c687ef6b0e42e17e` | 98 |
| Corrected C16 | `a4e09500fd8741bef7e3e0259bccdd4719219fc1fbe8087a4a0d42962568020f` | 29 |
| Matched C2 pair | `4601b61a976d9af10266c0ebca0a0ca52593bfa50ac8ba46809d5c6763a03630` | 33 |
| Mismatched initial capture | `49fedb3cc126da9d59f241729963ad729560a3bc497deb7605d039d4d467ce73` | 29 |

Validation passed: 90 attention-source tests, 17 physical-tile IR tests,
10 resource-policy tests and five benchmark-automation tests. Native checks
retain the existing migration warning exclusions `-79-20-29-25`; standalone
automation checks deny all warnings. No production kernel change survived this
rejected experiment, so these results are not a new sanitizer or serving pass.
