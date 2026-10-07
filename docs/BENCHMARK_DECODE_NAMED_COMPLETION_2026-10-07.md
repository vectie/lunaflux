# Decode named barrier resource limits

The named-barrier diagnostic removes explicit phase polling and passes short
history race checks, but does not improve paired timings. Dynamic barrier
identifiers reserve all 16 hardware barriers and limit residency to one block
per SM. Copy-completion waits and consumer rendezvous remain costly. The
long-history sanitizer gate exceeds its bounded memory allowance, so this
candidate is not qualified or selected for production.

## Explicit completion and reader retirement

This follows the [slot-completion experiment](BENCHMARK_DECODE_SLOT_COMPLETION_2026-10-07.md).
The bounded follow-up changes only synchronization realization, with one
candidate, 22 numerical boundaries, six paired timing cells and one matched
C16 counter pair. Numerical law, arithmetic, KV32/D128/GQA2 geometry and
frozen control remain unchanged.

The immutable `NamedSlots` allocator assigns separate K/V readiness and
reader-release resources to each slot, excluding the initialization barrier.
Tests cover allocation bounds and disjoint resource identifiers. The CUDA
renderer publishes readiness only after a producer `cp.async.wait_group 0`;
consumers wait on the matching named barrier. Consumers arrive at retirement
after their reads; producers wait for retirement before reusing a slot.
An explicit warp synchronization also retires probability reads before the
next epoch can reuse score storage.

This replaces the previous asynchronous mbarrier arrivals and retry loop,
not the arithmetic. It is an exact-source offline experiment, not a public
IR extension or a production pass. CUDA barrier details remain in terminal
lowering. NVIDIA documents the paired producer-arrive/consumer-wait and
consumer-arrive/producer-wait protocol, including restrictions on reuse before
the preceding rendezvous resets. [NVIDIA PTX barriers](https://docs.nvidia.com/cuda/parallel-thread-execution/)

## Alternating paired results

Five alternating pairs per cell preserve bitwise differential equality, the
scalar oracle and KV integrity. Acceptance requires at least 1% improvement
in every pair. All 30 pairs and all 22 boundary cases pass numerical checks;
no timing cell passes acceptance.

| Useful rows / envelope / history | Chain | Control median µs | Named median µs | Median paired gain | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| 16 / 16 / 4095 | Ordinary | 1146.420 | 1181.799 | -3.134% | Regression |
| 2 / 8 / 32767 | Ordinary | 1450.967 | 1694.569 | -18.303% | Regression |
| 1 / 1 / 127 | Ordinary | 8.227 | 8.205 | +0.155% | Inconclusive |
| 16 / 16 / 4095 | Partial and merge | 1168.954 | 1174.609 | -0.530% | Regression |
| 2 / 8 / 32767 | Partial and merge | 1194.653 | 1188.430 | +0.121% | Inconclusive |
| 1 / 1 / 127 | Partial and merge | 10.161 | 10.262 | -0.765% | Regression |

## Hardware resource and wait diagnosis

The counter pair contains exactly two ordinary launches, grid 16 × 8,
control block 64 and candidate block 128. Replay timing is diagnostic, not
the unprofiled acceptance result.

| Metric | Control | Named completion |
| --- | ---: | ---: |
| Replay duration µs | 1193.472 | 1249.312 |
| Registers per thread | 148 | 109 |
| Warp instructions | 41,500,160 | 44,429,312 |
| Occupancy limit from barriers in blocks per SM | 24 | 1 |
| Active warps percent of peak | 8.11 | 7.66 |
| Eligible warps per scheduler cycle | 0.09 | 0.08 |
| L2 read sectors | 8,398,378 | 8,401,748 |
| Long scoreboard per active issue | 2.97 | 5.30 |
| Short scoreboard per active issue | 3.82 | 0.62 |
| Barrier per active issue | 0.40 | 4.55 |
| MIO throttle per active issue | 2.26 | 0.02 |
| Branch resolving per active issue | 0.02 | 0.03 |
| Source excessive shared wavefronts | 0 | 0 |

NVCC reports 16 barriers for both compute entries because barrier identifiers
reach PTX as register operands. The occupancy counter confirms a one-block
barrier limit. Lower register demand therefore does not yield higher residency.

The explicit retry loop is gone: `BRA` executes 399,616 times, versus
13,910,761 in the preceding polling candidate. Hardware and source execution
counts agree in this capture. However, instructions still exceed control by
about 7.1%. `BAR.SYNC.DEFER_BLOCKING` executes 132,096 times and `BAR.ARV`
131,072 times; readiness and retirement rendezvous are not free.

Mathematical work remains identical: `FADD` 9,207,296 and `FMUL` 8,945,408.
128-bit async copies remain 524,288. Shared excessive wavefronts and spilling
remain zero. The candidate's hottest sampled uniform move has 27,738 barrier
samples out of 27,778 samples. Two warp-sync sites have over 26,000
load-dependency samples each. These are waiting locations, not proof that the
named instruction independently caused those cycles.

This rules out “remove phase polling and the speed gap disappears.” The next
lowering experiment must specialize the finite barrier-resource domain to
immediate identifiers and verify actual compiled residency. That would address
the demonstrated resource limit, but it would not by itself prove that
serialized producer copy completion is efficient. Those two dependencies need
separate measurements. Stall ratios cannot be summed into wall-time shares;
L2 sector counts are not DRAM byte counts.

## Bounded sanitizer results

At ordinary histories 0 and 32, memcheck with leak checking, racecheck and
synccheck pass. History 4096 memcheck passes; racecheck then exceeds the
8 GiB user-unit limit. The systemd result is `oom-kill`, not a correctness
pass or a GPU memory exhaustion claim. No GPU process remains afterwards.

A fresh qualification directory retains byte-identical CUDA source, recipe
and cubin. A single-worker racecheck rerun changes worker concurrency only;
async-copy checking, hazard reporting and the error exit code remain enabled.
It reaches the same history-4096 racecheck memory limit. Both failed units and
their preceding completed-case logs are preserved. The remaining split-chain
sanitizer cases were not run. The candidate remains blocked.

A diagnostic runner initially attempted equality on asynchronous file `Data`,
which does not implement `Eq`/`Debug`. It was corrected to a byte comparison;
the failed compile snapshot and unit remain preserved. This does not affect
the executed candidate or timing inputs.

GPU workloads were serialized, capped at 8 GiB with no swap and a 32 GiB
available-memory reserve. Instrumentation limits were not raised to force a
pass. No production artifact, route, compiler default or numerical tolerance
changed. The broader serving gap and full-model parity gate remain open;
there is no new end-to-end token/s result from this experiment.

## Evidence and identities

Remote root:
`/home/wlc004s/lunaflux-decode-named-completion-20261007.tgLnrBJR`.
Local downloaded root:
`/tmp/lunaflux-decode-slot-sealed-20261007.4Up9g4/named-evidence`.
Archive SHA-256:
`ff459a971963fd5cc9876a04c085e07654fd05ac6d61a4d2c48c6ef8a6a1523c`.
All 326 manifest members verify locally against the sealed checksums.

| Identity | SHA-256 |
| --- | --- |
| Candidate CUDA source | `f05d703d1f6981785f82e69db09ca68fad7f5f1dc7caae8b8e8a696e2eb80feb` |
| Candidate cubin | `d2455442f5bb7ad1efd064326c8471b48783610adf0c1b5dd3c3bfc91b3bd0c7` |
| Candidate recipe | `bdc1f18948ad586af66a044a4afe3e9b44497529e44aa4ef9b96d21a3ef9cdf6` |
| Repaired probe | `49560803fad1705204bea72b1c71546cd69fcac1d931f92323ad6de74be382cd` |

GPU UUID is `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, Spark GB10 sm121.
Compiler remains the pinned NVCC 13.0.88; the frozen control identities are
unchanged from the preceding report. The sealed result explicitly records
`qualification=blocked` and `production_changed=false`.
