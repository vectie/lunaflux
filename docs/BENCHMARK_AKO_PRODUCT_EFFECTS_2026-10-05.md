# Shared-read / pure-product split — dual Spark, 2026-10-05

## Decision

Do not enable this candidate. Both machines ran the three timing cells
concurrently; paired reductions range from −0.02% to +1.02%. None passes the
unchanged five-pair, 3%-per-pair threshold. Production sources and selection
remain unchanged. This is a measured rejection, not a new serving speedup.

The AKO loop requires executable propagation before interpreting timing, and
keeps the non-win outside production. No new framework/E2E comparison is claimed.

## One hypothesis, six cells

The forwarded key/value product helper previously puts a shared LDSM and two
MMA instructions in one volatile memory-clobbered assembly region. The offline
candidate separates its convergent volatile shared read from pure dependent
MMA arithmetic. Accumulator reduction order, RHS word permutations, BF16
probability rounding, exp2 numeric law, page-batch addressing, Q64/K64/D128,
one stage and all barriers remain unchanged. Unlike the earlier rejected
product-pair experiment, it does not prescribe a half-major pairing schedule.

This tests compiler scheduling freedom, not removal of necessary read effects
or synchronization. The helper is a pure source transformation with regression
tests, in an offline MoonBit driver. It adds no request JIT, model-specific
branch, runtime check or new production IR. Its recipe explicitly says
production_admission=false; retained baseline lowering identities are not
presented as an authentic production receipt for the transformed artifact.

Both hosts use CUDA13.0.88, sm121, fixed compiler flags and the same frozen
accepted page-batch baseline. Each cell has five alternating pairs with30
CUDA-event repetitions/member. There is one GPU job per host. After timings,
.178 runs short sanitizers and long memcheck while .179 runs Nsight counters.

## Actual executable propagation

The selected symbol is lunaflux_attention_prefill_tile_compiler_exp2_v1.
Executable text changes78,208→78,592 bytes:

- Baseline SHA: 62e5de8dd753c52009f7b9385a6b14a2350024fbd422cadc684edbc75719f65c.
- Candidate SHA: a1be8ae5ca5b433d351f3938baa39bf9fc2e2da30eb137b9d3efe34667761e8e.

Candidate executable text matches across hosts; whole cubin hashes differ
because independently compiled debug/source paths differ. Each host's two
candidate compiles match exactly. Registers remain234 (240 allocated), zero
spills/stack, two resident register/shared-limited CTAs. Launch block128,
49,168 dynamic shared bytes and profile grid63×16×1 are unchanged.

## Paired results

The reduction column is median paired ratios, not ratio of separate medians.

| Host | Active rows / history | Baseline µs | Candidate µs | Paired reduction | Worst pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| .178 | 1 / 28,672 | 6383.513 | 6364.018 | +0.096% | −1.297% |
| .178 | 2 / 28,672 | 6382.146 | 6352.335 | +0.521% | −0.605% |
| .178 | 2 / 8192 | 1916.403 | 1904.852 | +0.364% | −0.883% |
| .179 | 1 / 28,672 | 6611.469 | 6565.488 | +0.342% | −0.238% |
| .179 | 2 / 28,672 | 6605.463 | 6609.389 | −0.020% | −2.510% |
| .179 | 2 / 8192 | 1991.739 | 1971.445 | +1.019% | +0.265% |

## Why this freedom does not give a useful speedup

Matched .179 Q2048/R2/H28672 counters:

| Metric | Baseline | Candidate |
| --- | ---: | ---: |
| Executed warp instructions | 891,530,112 | 904,586,112 |
| Tensor HMMA | 119,668,736 | 119,668,736 |
| Non-transposed LDSM | 37,396,480 | 37,396,480 |
| Transposed LDSM | 29,917,184 | 29,917,184 |
| MOV | 97,258,496 | 98,199,552 |
| NOP | 25,242,624 | 26,177,536 |
| CTA barriers | 2,808,832 | 2,808,832 |
| Active warps, percent of peak | 16.024% | 16.032% |
| Wait / average warp latency | 35.58% | 31.69% |
| Long scoreboard / average warp latency | 11.31% | 10.89% |
| Math throttle / average warp latency | 16.09% | 14.60% |
| Barrier / average warp latency | 3.08% | 5.57% |

Arithmetic and matrix-read counts do not decrease. Address/control support
increases: LOP3 +3,739,648; IMAD.SHL +2,785,280; IADD3 +936,960; S2UR/UMOV/ULEA
each +934,912; SHF.R +915,456, along with MOV/NOP increases. Total instructions
grow1.46%. Lower wait/throttle fractions are not acceptance evidence when
unprofiled time stays flat and other costs increase.

HMMA not-issued wait samples fall106,474→102,476 and math-throttle samples
84,165→79,097. The hot page-bound comparison's long-scoreboard samples increase
46,276→51,260. These are sampled observations, not cycles or additive framework
gap contributions. Instrumented durations7.198/7.192ms are not used for timing
acceptance.

The result rejects simply loosening this assembly boundary as the main remedy.
It does not show that all MMA scheduling is optimal. The next intervention
must reduce the real operand/address/publication critical path without adding
support work; another maximum/guard cleanup or blanket assembly qualifier
change is not justified by this data. Any explicit product pipeline must
demonstrate changed issue/load dependencies and whole-chain time, rather than
assuming freedom alone produces better code.

## Correctness, bounded memory and reproduction

All timing outputs are bitwise equal/maxabs0 and meet the sampled BF16 oracle
bound0.003. .178 Q129/R2/H128 memcheck/racecheck/synccheck and
Q2048/R2/H28672 memcheck pass with zero errors/hazards. This is kernel diagnostic
coverage, not whole-model quality or untested shape admission.

Minimum observed MemAvailable122,200,532KiB exceeds the33,554,432KiB reserve.
CPU units8GiB/no swap/TasksMax128/300s; GPU units16GiB/no swap/TasksMax64/600s.
Both GPUs are idle at terminal checks. No million-token/model allocation occurs.

Helpers: ako_product_effects.mbtx prepares the frozen baseline transformation;
ako_long_schedule.mbtx handles contract/validation/profile/finish with
product-effects-v1; ako_trial.mbtx performs fixed paired trials;
ako_code_identity.mbtx compares symbol text; ako_long_schedule_report.mbtx
reduces the raw timing/SASS counters. Standalone warning-denied checks and
tests pass: product1, schedule1, report2. No unrelated dirty changes are staged.

- Local root: /tmp/lunaflux-ako-product-effects-20261005.rODeJ2eJ.
- .178: /home/wlc003s/lunaflux-ako-product-effects-20261005.u6Kd9wGH.
- .179: /home/wlc004s/lunaflux-ako-product-effects-20261005.JZTuPsN8.
- Baseline source: 55ceeda9b04e30239ef289004bd64bdafc4b3436dc0d0557d2bde7b1310b9f9e.
- Baseline cubin: 7b42b3531466f5fe22c59e5aac2883db68984907c49a5637ac5f1358ad2eab57.
- .178 candidate cubin: 64d8e92ed680b4ad0e9b137d4b5919350641bd6ce89b0cd1d5bc41153b2bd59e.
- .179 candidate cubin: 942dcc8db2d4682ab79c6acce0014bc78c3794b2fb2975b2ce232dabb23dd408.
- .178 archive: 97fdcd7fc4d962239db30fd979aebd5364e82e5a435fc2db66bcc00abbe90bf5.
- .179 archive: b382c3f52d8811aec5173dd2e179e791b956ffaa08bad2be9d6063c4ea974c7a.

Downloaded archives use unique paths; archive hashes and each measurement
manifest entry verify locally. Raw generated source, recipe, exact probe/spec,
timing pairs, profiler report/SASS and sanitizer logs are preserved.
