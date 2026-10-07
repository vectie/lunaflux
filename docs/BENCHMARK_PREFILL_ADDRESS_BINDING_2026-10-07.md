# Selected prefill address hoisting and benchmark path repair

The address-basis rewrite does not accelerate the selected prefill kernel.
Fresh paired runs reject it, and matched counters show identical executed
opcode counts. Its SASS changes initialization but leaves every instruction
and encoding after `0x5d80` unchanged. Production lowering and selection remain
unchanged. This experiment does not close the vLLM/SGLang serving gap.

A benchmark-harness bug is fixed: cloning a contract with `String::replace`
rebound only its first path. The first timing attempt therefore executed the
previous operand candidate, not the new address candidate. Those timings are
invalid for this hypothesis and are preserved separately. The repaired helper
rebinds every occurrence, checks that no frozen root survives before execution,
and tests all three workload argument lists and the artifact inventory.

## Experiment and functional boundary

The frozen reference is candidate 30322, symbol
`lunaflux_attention_prefill_tile_compiler_exp2_v1`, Q64/K64/D128, on Spark .179
GB10 sm121, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
The numerical law remains `approx-base2-f32-v1`.

The hypothesis was that hoisting the invariant PV shared-address basis out
of the four key-fragment iterations would expose useful integer work without
the restrictive joined PTX region rejected in the
[PV lifetime experiment](BENCHMARK_PREFILL_VALUE_PIPELINE_2026-10-07.md).
The pure rewrite preserves every lane/key/column address, ascending product
order, fragment mapping, and separate load/MMA regions. It does not change QK,
softmax, publications, or the shared layout. Exhaustive address tests cover
32 lanes, four key fragments and eight output columns.

This is an offline terminal-lowering prototype, not a model-specific branch,
new production IR layer, or request-path JIT. Only a measured winner would
justify integrating the corresponding pure plan into general lowering.
The budget was one candidate, three cells and matched counters after a non-win.

## Valid paired measurements

The repaired contract uses the new root for every spec, baseline, candidate
and hashed artifact path. Five alternating pairs per cell, 30 CUDA-event
repetitions per member. Admission requires at least 3% gain in every pair in
every cell. Positive gain means faster; paired medians are not ratios of
latency medians. These are synthetic operands with real runtime bucket launch
geometry, not whole-model token/s measurements.

| Q/R/H | Baseline median µs | Candidate median µs | Median paired gain | Worst pair | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| 1792/1/30720 | 6263.748 | 6288.001 | −0.39% | −2.75% | Regression |
| 2048/2/28672 | 6575.018 | 6590.718 | −0.24% | −1.18% | Regression |
| 2048/16/2048 | 911.829 | 933.610 | −2.48% | −4.69% | Regression |

All 15 pairs report complete-output bitwise agreement, maximum absolute
difference zero and sampled FP64 oracle error within 0.003. Memcheck,
racecheck and synccheck at Q129/R2/H128 report zero errors/hazards and empty
stderr. This boundary check is not whole-model or long-context sanitizer
qualification.

Each GPU process is serialized, limited to 8 GiB, no swap, 900 seconds, with
a 32 GiB MemAvailable reserve. Minimum recorded timing MemAvailable is
121,818,616 KiB. Terminal GPU is idle. Both artifacts use 234 registers/thread,
240 allocated registers/thread, two resident blocks/SM, no local allocation,
and no stack.

## Matched counters and generated code

Fresh Nsight Compute captures use Q2048/R2/H28672, grid 63×16, block 128.
Both artifacts execute exactly the same per-opcode counts, not merely the same
total instruction count.

| Counter | Baseline | Candidate |
| --- | ---: | ---: |
| Executed warp instructions | 891,530,112 | 891,530,112 |
| HMMA | 119,668,736 | 119,668,736 |
| Transposed LDSM | 29,917,184 | 29,917,184 |
| MOV | 97,258,496 | 97,258,496 |
| IMAD | 14,876,672 | 14,876,672 |
| IMAD.SHL | 15,681,536 | 15,681,536 |
| NOP | 25,242,624 | 25,242,624 |
| Warp publications | 3,739,648 | 3,739,648 |
| CTA barriers | 2,808,832 | 2,808,832 |
| Tensor activity | 65.04% | 65.01% |
| Active warps | 16.04% | 16.04% |
| Long scoreboard wait per active issue | 0.72 | 0.71 |
| Short scoreboard wait per active issue | 0.45 | 0.44 |
| Fixed wait per active issue | 2.19 | 2.19 |

Source-correlated excessive shared wavefronts are zero for both; local-sector
metrics were not collected. Fixed-wait not-issued samples on HMMA are
106,597/107,290 and on NOP 42,527/42,364. These samples cannot be added into
elapsed-time attribution.

The disassembly differs only before `0x5d80`: register initialization,
address register assignment, one commuted integer addition and a reuse flag.
The verified suffix comparison includes instruction encodings and scheduling
control words. Thus this source-level hoist does not improve the repeated
consumer schedule. A different CUBIN hash alone is not evidence of an improved
kernel. The timing differences do not establish that equivalent address
algebra fundamentally costs 2.48% more on all machines or shapes.

## Remaining scope

The latest serving result remains the
[October 6 report](BENCHMARK_DECODE_ROUTE_AND_FAIRNESS_FIX_2026-10-06.md):
4096/64/C16 is 225.13 tok/s versus vLLM/SGLang 243.03/241.85;
32512/64/C2 is 16.95 versus 17.58/18.85. These references are earlier
same-session measurements, not newly interleaved baselines for this prototype.
No rejected candidate is installed or advertised as a serving improvement.

The separate solo/paired numerical question also remains open. The next
performance change must alter an executed dependency or reduce actual work,
not repeat algebra already handled by the device compiler. The one-candidate
budget is exhausted; no additional speculative variant was tested in this
round.

## Reproduction and verified evidence

`benchmarks/gpu_pipeline/prefill_value_addresses_20261007.mbtx` prepares,
executes, validates and profiles the candidate. Its stale-path invalidation
mode preserves a bad timing attempt without promotion. Three regression
tests pass. `summarize_address_basis_20261007.mbtx` retains the metric/opcode
comparison and verifies the SASS suffix; two tests pass. Both helpers pass
warning-denied native checks.

Baseline CUBIN SHA-256:
`616ecc0f88c192e59c3134bfc9390a57d59984cb3b31543956b587858e132cdb`.
Candidate CUBIN SHA-256:
`57fd6dfb4d642b07f20706597e5490002352cec38b1140c8376ec3aa1a079980`.

Valid evidence: `/home/wlc004s/lunaflux-prefill-address-repair-20261007.tv0CXXrv`.
Archive SHA-256:
`ee4a8e74ad91745bb4de1508f1f75139a4a9d55dbc4a2df24d89896ef56ce81a`.
Invalid timing attempt: `/home/wlc004s/lunaflux-prefill-address-20261007.WsCZ9IRT`.
Its separate archive SHA-256 is
`496a72e6a88c5118c624d80b2dd1610659eba871cbf41f66204117065f05c068`.

Both archives are downloaded without overwrite to
`/tmp/lunaflux-address-final-20261007.L4kX2kNV`. Their hashes and all
138 valid-run / 82 invalid-run manifest entries verify locally. Raw timings,
contracts, command arguments, SASS, counters, sanitizers and executed controller
revisions are retained. Build caches are excluded; no original evidence was
deleted.
