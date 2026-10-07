# Prefill key read ownership and dependency scheduling

Permuting QK shared-read ownership removes 94.18% of executed register moves
without changing any tensor arithmetic or output bits. On its own, this gives
inconsistent gains and does not pass the frozen performance gate. Additional
device-compiler NOPs absorb part of the instruction savings. The problem is
therefore not merely excess source instructions: physical operand ownership
and the read-to-product dependency schedule must be optimized together.

Production sources and selection remain unchanged. This is an offline
kernel experiment, not a new token/s measurement or a completed framework
performance repair. The independent batched projection numerical question
also remains open.

## Ownership change

The original non-transposed `ldmatrix.x4` supplies four matrix words in the
order `[00,10,01,11]`. Its two ordered MMA consumers require `{b0,b2}` and
`{b1,b3}`. PTX forwarding preserves the values by name, but hardware register
allocation inserts moves to form the physical operands.

The candidate swaps address-provider lane bits 3 and 4 before reading the
same published shared tile. The matrix word order becomes `[00,01,10,11]`,
allowing adjacent `{b0,b1}` and `{b2,b3}` consumers. Producer storage, Q/PV
loads, causal traversal, arithmetic order, probability rounding, online
rescale, copy publication, and all barriers remain frozen. This permutation
belongs to terminal device ownership realization, not model semantics.

Pure MoonBit tests check the address permutation at dimensions 64, 128 and
256, every reduction fragment, four row origins, all 32 providers, and both
packed BF16 components. They also reject ambiguous source sites. Generated
source alone is not accepted as evidence of fewer physical moves; the
selected cubin and dynamic SASS counters below establish that change.

## Paired timing for ownership alone

Each workload uses five alternating baseline/candidate pairs and 30 CUDA
event repetitions per measurement. Queries count new tokens; history counts
prior KV positions. Runtime launch capacity remains 2048 queries and 32 rows,
with grid `63 × 16`, block 128 and Q64/K64/D128.

| Queries | Active rows | History | Baseline median µs | Candidate median µs | Median paired gain | Worst pair |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1792 | 1 | 30720 | 6265.54 | 6105.12 | 2.90% | −0.16% |
| 2048 | 2 | 28672 | 6592.49 | 6466.17 | 1.89% | 0.82% |
| 2048 | 16 | 2048 | 938.18 | 914.09 | 0.03% | −1.50% |

Median paired gain is the median of paired ratios, not a ratio of independent
median times. The short case illustrates the difference clearly. None meets
the unchanged rule requiring at least 3% gain in every pair.

## Matched machine counters

Fresh captures select exactly one launch of
`lunaflux_attention_prefill_tile_compiler_exp2_v1` at queries 2048, rows 2,
history 28672. Unprofiled timing above remains the timing authority.

| Dynamic instruction | Baseline | Ownership candidate |
| --- | ---: | ---: |
| Total | 891,530,112 | 785,607,552 |
| MOV | 97,258,496 | 5,662,720 |
| NOP | 25,242,624 | 41,136,128 |
| HMMA | 119,668,736 | 119,668,736 |
| FMUL | 123,408,384 | 123,408,384 |
| FADD | 67,313,664 | 67,313,664 |
| LDSM ordinary | 37,396,480 | 37,396,480 |
| LDSM transposed | 29,917,184 | 29,917,184 |
| CTA barrier | 2,808,832 | 2,808,832 |
| Warp synchronization | 3,739,648 | 3,739,648 |

The ownership change removes 91,595,776 MOVs and 11.88% of total executed
instructions, but NOPs grow by 15,893,504. These counts confirm the specific
register-pairing inefficiency; they do not show that all moves were unnecessary
or that their removed issue slots must translate proportionally into time.

| Resource or issue metric | Baseline | Ownership candidate |
| --- | ---: | ---: |
| Registers per thread | 234 | 236 |
| Allocated registers | 240 | 240 |
| Register-limited resident blocks | 2 | 2 |
| Dynamic shared bytes | 49,168 | 49,168 |
| Local bytes | 0 | 0 |
| Tensor activity | 65.10% | 66.35% |
| Active warps | 16.03% | 15.99% |
| Average warp latency per issued instruction | 6.52 | 6.81 |
| Fixed-wait contribution | 2.19 | 2.54 |
| Long-scoreboard contribution | 0.72 | 0.85 |
| Short-scoreboard contribution | 0.45 | 0.43 |
| Math-throttle contribution | 0.99 | 1.04 |
| Barrier contribution | 0.17 | 0.20 |

Issue-normalized warp contributions are not additive completion-time shares.
Removing instructions changes their denominator. The increased NOP count and
fixed-wait contribution support testing operand-read scheduling next, but do
not identify one sampled consumer as proof of every preceding producer stall.
Neither occupancy nor mathematical work changed enough to explain a broad
engine speedup.

## Correctness and resource bounds

All 15 pairs preserve full baseline output equality, `bitwise=true, maxabs=0`.
The largest sampled scalar-oracle error is 0.000420981, below the unchanged
0.003 ceiling. Memcheck, racecheck and synccheck report zero errors or hazards
and empty stderr at queries 129, rows 2, history 128. This is diagnostic
kernel coverage, not whole-model numerical admission.

Jobs are serialized, limited to 8 GiB with swap disabled, and require a
32 GiB MemAvailable reserve. The lowest observed trial value is
121,301,480 KiB; the GPU is idle at the terminal check. The current automation
passes warning-denied native check and five focused tests, including read-ahead
lifetime and exact source-replacement coverage. The first frozen archive
retains its three-test driver; the follow-up retains its four-test driver.

## Frozen identities and retained results

- GPU UUID: `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, Spark GB10, sm121.
- NVCC 13.0.88 SHA-256: `fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
- Probe SHA-256: `78901fc5e3222e4394b3fb35746d60ad36a0181f22472c0397c5dee1e5079536`.
- Baseline source: `3738eb47bb25c18c6b9cb621d01725f45979fd5cb72b1b5ccbaa9be2f5aebd78`.
- Baseline cubin: `616ecc0f88c192e59c3134bfc9390a57d59984cb3b31543956b587858e132cdb`.
- Ownership source: `ce322de8cd2325c8c8b087f26e55bffbbcd5a9d3e2134bb5fc98150240c0e3e2`.
- Ownership cubin: `222cc7691e986b96ddb16df595abad95d7ce61c16c3f28fd343a8aa804d29f3a`.

Remote results are sealed at
`/home/wlc004s/lunaflux-prefill-key-read-20261007.fpLSOjRj`.
The local copy is
`/tmp/lunaflux-key-read-result-20261007.6pOW5QoV/verified`.
All 143 file-manifest entries verify locally. The downloaded archive SHA-256
is `0035b06497cb3d42ad6593cc9a973d1a90f76cb6c88ba73d168e3150fcfc5ba7`.

Implementation is
`benchmarks/gpu_pipeline/prefill_key_read_ownership_20261007.mbtx`.
The frozen first experiment uses its default ownership-only mode. Optional
`read-ahead` selects a distinct experiment in a fresh directory. The generic
profiling and counter comparison helpers are reused unchanged; the historical
filename `identity-comparison.json` does not mean identity rescaling was
enabled here. Both original and transformed artifacts, raw timing pairs,
sanitizer logs and Nsight reports are retained.

## Follow up with two read ahead slots

The second finite experiment combines the same ownership permutation with
two RHS read-ahead slots. It loads the next published key fragment before
consuming the current one, retires a slot only after its ordered products,
and preserves ascending reduction order. A single assembly region contains
the four column products. No global copy, publication, barrier or numerical
law changes. Its frozen baseline is still the original artifact, not the
ownership-only candidate; the two experiments are not a direct paired
comparison with each other.

| Queries | Active rows | History | Baseline median µs | Read ahead median µs | Median paired gain | Worst pair |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1792 | 1 | 30720 | 6313.90 | 5944.91 | 6.05% | 3.10% |
| 2048 | 2 | 28672 | 6598.63 | 6375.45 | 3.68% | 1.94% |
| 2048 | 16 | 2048 | 937.72 | 935.53 | 0.16% | −1.07% |

The one-row cell passes every pair. The two-row and short cells do not clear
the unchanged gate, so the overall verdict remains inconclusive and selection
unchanged. All 15 additional pairs are bitwise identical with zero maximum
absolute difference and sampled oracle error at most 0.000420981. All three
short sanitizer gates pass again. The minimum observed MemAvailable is
121,534,012 KiB.

Fresh selected Q2048/R2/H28672 counters:

| Dynamic instruction | Baseline | Read ahead candidate |
| --- | ---: | ---: |
| Total | 891,530,112 | 762,274,496 |
| MOV | 97,258,496 | 4,792,320 |
| NOP | 25,242,624 | 49,550,336 |
| Branch | 10,464,256 | 6,739,968 |
| HMMA | 119,668,736 | 119,668,736 |
| LDSM ordinary | 37,396,480 | 37,396,480 |
| LDSM transposed | 29,917,184 | 29,917,184 |
| CTA barrier | 2,808,832 | 2,808,832 |
| Warp synchronization | 3,739,648 | 3,739,648 |

Total instructions decrease 14.50%; MOVs decrease 95.07%. The schedule still
introduces additional NOPs, and matrix work remains unchanged. Registers are
236, allocated registers 240, resident blocks two, dynamic shared memory
49,168 bytes, and local/stack usage zero. Tensor activity is 65.08% versus
66.14%; active warps are 16.03% versus 16.00%.

Average issue-normalized warp latency is 6.64 versus 7.53, fixed wait 2.19
versus 2.61, long scoreboard 0.72 versus 0.93, short scoreboard 0.44 versus
0.36 and barrier 0.19 versus 0.34. These higher normalized contributions do
not negate the unprofiled long-history gains and cannot be interpreted as
elapsed time added by the change. They also do not establish that the
remaining dependency chain is resolved. Further work must demonstrate
robust whole-chain and serving gains rather than simply fewer MOVs.

Read-ahead source SHA-256:
`76f21ee7a1159b122b3cf3f9da4fd0d5306d8f88affe0b57d7c54ed822900119`.
Read-ahead cubin SHA-256:
`301f8fd9db1e0221e4b0b5048d08c03d55ee49f5f50d8ea04e5df33e4582ebf3`.
Remote results are sealed at
`/home/wlc004s/lunaflux-prefill-key-lookahead-20261007.e3cOZlOj`.
Downloaded archive SHA-256:
`d3d4d9582a57648c4a978b71ca8e037d62c866e759bfa142d6eea5d55a46876a`.
The local verified copy is
`/tmp/lunaflux-key-lookahead-result-20261007.owN92qSY/verified`;
all 143 file-manifest entries verify locally.
