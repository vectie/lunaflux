# Selected prefill dependency samples and operand lookahead

## Result

Rolling two-slot QK operand lookahead reduces paired kernel time by a median
2.34% on the long tail and 1.56% on the two-row history workload. It does not
pass the declared 3% improvement gate in every pair. The short control also
contains regressions. Production selection remains unchanged.

Fresh instruction samples locate distributed fixed-dependency waiting at tensor
instructions and NOPs, rather than primarily at exponential instructions. A
single hottest PC is insufficient to describe this distribution. These samples
identify waiting consumers; they are not additive elapsed-time contributions
or proof of a particular producer dependency.

This is one bounded offline lowering experiment, not a serving optimization or
a new vLLM/SGLang comparison. The [October 6 end-to-end report](BENCHMARK_DECODE_ROUTE_AND_FAIRNESS_FIX_2026-10-06.md)
remains authoritative: 4096/64/C16 is 225.13 tok/s versus 243.03/241.85, and
32512/64/C2 is 16.95 versus 17.58/18.85. Concurrent numerical parity remains
unresolved; isolated bitwise kernel agreement does not settle that issue.

## Selected workload and capture

Spark .179, GB10 sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, driver 580.178.04. The selected
symbol is `lunaflux_attention_prefill_tile_compiler_exp2_v1`, candidate 30322,
Q64/K64, single-stage split-copy lifetime, numerical law
`approx-base2-f32-v1`. The probe has 16 query heads, 8 KV heads, dimension 128
and page size 8. Synthetic immutable operands use real runtime bucket geometry.

One Nsight Compute 2025.3.1 capture executes Q2048/R2/H28672 and records source
counters, warp states and launch resources. Source-correlated excessive shared
wavefronts and local sectors are zero. They do not imply that all kernels or
all hardware-level bank metrics are zero.

| Opcode | Not issued fixed-wait samples | Executed instructions |
| --- | ---: | ---: |
| HMMA | 106,674 | 119,668,736 |
| NOP | 41,782 | 25,242,624 |
| FADD | 9,920 | 67,313,664 |
| BRA | 9,456 | 10,464,256 |
| FMUL | 5,623 | 123,408,384 |
| MOV | 5,062 | 97,258,496 |
| FMNMX | 4,407 | 35,526,656 |
| MUFU.EX2 | 902 | 31,787,008 |

The largest individual sampled load-dependency site is a metadata predicate
`ISETP.GE.U32.AND` with 58,610 samples. Many tensor dependency sites each have
roughly 1,600 fixed-wait samples. Ranking only the largest PC would miss their
aggregate importance. The analyzer now aggregates `stall_wait (Not Issued)`
by opcode, separately from all `stall_wait` samples, and retains instruction
windows without asserting producer edges.

## Lowering experiment

The immutable plan has explicit `Read(column, slot)` and `Product(column, slot)`
actions. It reads columns 0 and 1 into two slots, consumes column 0, reuses its
slot for column 2, consumes column 1, then continues in ascending column order.
Every slot remains owned until its previous product consumes it. QK arithmetic,
BF16 operand mapping, reduction order, softmax, PV and shared publications stay
unchanged. The terminal CUDA lowering emits the corresponding PTX sequence.

This tests column operand lifetime, not the previously rejected paired-half
product ordering or larger-tile schedules. It adds no model-name rule, runtime
JIT, request-path validation or production strategy API. The exact selected
baseline comes from the pinned exporter, with an identical baseline CUBIN;
stable candidate IDs alone are not considered proof of equal lowering.

Offline compilation uses CUDA 13.0.88, O3, `fmad=false`, FTZ off, precise
division/square root and a 255-register ceiling. Emitted code changes, with a
higher register count and additional move and NOP sites:

| Static resource or instruction sites | Baseline | Rolling lookahead |
| --- | ---: | ---: |
| Registers per thread | 234 | 235 |
| Shared bytes | 1,024 | 1,024 |
| Local bytes | 0 | 0 |
| HMMA sites | 128 | 128 |
| LDSM sites | 72 | 72 |
| MOV sites | 532 | 543 |
| NOP sites | 38 | 48 |

These are static disassembly counts, not dynamic counts for the alternative.
There is no alternative counter capture assigning its modest timing change to
those extra sites. They show that this is not an instruction-count reduction.

## Paired timings

Five alternating pairs per cell, 30 CUDA event repetitions per member; all
pairs are retained. Positive gain means faster. Median paired gain and the
ratio of latency medians are different statistics; the short control shows why
they must not be substituted for one another.

| Workload | Baseline median µs | Candidate median µs | Median paired gain | Worst pair gain | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Q1792 R1 H30720 tail | 6,278.315 | 6,133.622 | +2.34% | +0.20% | Inconclusive |
| Q2048 R2 H28672 | 6,593.860 | 6,491.312 | +1.56% | +0.98% | Inconclusive |
| Q2048 R16 H2048 short control | 915.967 | 923.465 | +0.08% | −4.61% | Inconclusive |

All 15 pairs have complete-output bitwise equality and maximum absolute
baseline difference zero, with the sampled oracle within the declared 0.003
limit. Candidate-only memcheck, racecheck and synccheck pass at Q129/R2/H128:
zero errors/hazards, empty stderr, bitwise equality and FP64 oracle maximum
absolute error 0.000330008. This sanitizer scope is not full long-context or
whole-model qualification.

GPU work is serialized with an 8 GiB/no-swap process scope, 900-second timeout
and a 32 GiB MemAvailable reserve. The minimum available memory recorded during
timing is 121,390,092 KiB. The terminal GPU is idle. Later controller hardening
adds an explicit pinned-probe hash and per-sanitizer UUID, idle and reserve
checks; that revision is preserved separately from the executed commands.

## Remaining dependency problem

Operand lookahead alone does not establish a robust win. A next experiment
must identify which QK/PV accumulator, fragment-transfer or output-rescale
dependencies serialize the consumer, and test that exact chain under an
explicit live-register budget. More buffering, wider tiles and a new IR layer
are not justified by this result. The separate concurrent numerical mismatch
also remains a correctness priority before any broader serving claim.

The functional architecture stays semantic/numerical contract → pure schedule
and ownership → explicit effects and lifetime → device lowering. The prototype
remains diagnostic until an executable schedule demonstrates a repeatable gain
and its actual serving dispatch is verified.

## Checks and preserved identities

All three changed offline helpers pass warning-denied native checks. Three
tests pass: two analyzer regressions and one ownership regression spanning two
through eight operand columns. No production source or API changes are made.

- Baseline source SHA-256:
  `3738eb47bb25c18c6b9cb621d01725f45979fd5cb72b1b5ccbaa9be2f5aebd78`.
- Baseline CUBIN SHA-256:
  `616ecc0f88c192e59c3134bfc9390a57d59984cb3b31543956b587858e132cdb`.
- Candidate source SHA-256:
  `6ec31282938f9f6a372fca59018ae978c6406d0e321e5d23a919f04f91d88c26`.
- Candidate CUBIN SHA-256:
  `8e625385d4a314e0fb2a31ec7683fabab8b061b99fbf7186a4d0d6c4a81b2811`.
- Probe SHA-256:
  `78901fc5e3222e4394b3fb35746d60ad36a0181f22472c0397c5dee1e5079536`.

Counters and instruction windows:
`/home/wlc004s/lunaflux-prefill-dependency-20261007.W9uC55Al`.
Rolling alternative, paired samples and sanitizers:
`/home/wlc004s/lunaflux-prefill-operands-20261007.8clxHFEs`.

Archive SHA-256 values are respectively
`550b726f404cff1cfe9c1a1af7e36b64a28a077f06fabc2295b54c48fea94b21`
and `727ac444c597e2376ec98c151ee84446a98e75ccc744d6967aba1820c2e39e22`.
Distinct downloads under `/tmp/lunaflux-prefill-dependency-20261007.nbIakcBC`
match these hashes. Both extracted `MEASUREMENT.sha256` manifests verify
locally. No prior archive is overwritten.
