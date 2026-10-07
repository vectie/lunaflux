# Prefill identity rescale measurement

Skipping identity output rescaling removes real executed instructions, but does
not give a reliable performance win on the frozen workload vector. Production
selection remains unchanged. The long-history paired median gains are 1.43%
and 1.07%, with losing pairs; the short control gains 0.46% at the paired
median and also has losing pairs. No cell meets the existing admission rule
of at least 3% gain in every pair.

This corrects a possible interpretation of the earlier address-basis result:
that rewrite did not change the repeated machine-code loop. This experiment
does change it. Even a 6.07% reduction in total executed instructions does not
by itself resolve the remaining attention pipeline cost.

## Change and numerical scope

The frozen c30322 kernel unconditionally multiplies all register-owned output
accumulators by the online-softmax rescale factor on each KV tile. The
diagnostic candidate adds one warp-wide `any` vote. When both factors are
exactly one in every lane, it skips the scale loop. Otherwise the entire warp
executes the original loop, in the original order. QK, probability rounding,
PV, denominator update, copy publication, and CTA synchronization are unchanged.

The experiment uses finite probe inputs and the existing
`approx-base2-f32-v1` law. It does not admit a new general floating-point
identity law: signaling NaNs, exceptional intermediate states, and the full
accepted production domain require separate numerical analysis before any
integration. There is no production pass, request-path branch, model-specific
selection, or runtime compilation added by this experiment.

## Paired timing

Each cell retains five alternating pairs with 30 event repetitions per
measurement. Query counts are new query tokens, and history is prior KV
history, not a total-context label. All use the runtime bucket launch,
2048 query-token capacity, 32-row capacity, grid `63 × 16`, and block 128.

| Queries | Active rows | History | Baseline median µs | Candidate median µs | Median paired gain | Worst paired gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1792 | 1 | 30720 | 6297.44 | 6207.60 | 1.43% | −0.17% |
| 2048 | 2 | 28672 | 6639.87 | 6567.67 | 1.07% | −1.08% |
| 2048 | 16 | 2048 | 936.68 | 934.36 | 0.46% | −4.42% |

Median paired gain is the median of `1 − candidate / baseline` for paired
samples; it is not calculated from the two independent median times. All
15 pairs preserve full baseline output equality, `bitwise=true, maxabs=0`.
The largest sampled scalar-oracle absolute error is 0.000420981, below the
unchanged 0.003 ceiling. This is kernel correctness, not model-quality parity.

## Exact selected kernel counters

Fresh Nsight captures compare both pinned artifacts at queries 2048, rows 2,
history 28672. Timing above is unprofiled; counter replay duration is not
substituted for paired timing.

| Executed instruction | Baseline | Candidate |
| --- | ---: | ---: |
| Total | 891,530,112 | 837,436,288 |
| FMUL | 123,408,384 | 63,705,088 |
| HMMA | 119,668,736 | 119,668,736 |
| MOV | 97,258,496 | 97,258,496 |
| FADD | 67,313,664 | 67,313,664 |
| LDSM ordinary | 37,396,480 | 37,396,480 |
| LDSM transposed | 29,917,184 | 29,917,184 |
| NOP | 25,242,624 | 27,112,448 |
| Branch | 10,464,256 | 11,399,168 |
| CTA barrier | 2,808,832 | 2,808,832 |
| Warp synchronization | 3,739,648 | 3,739,648 |

The guard removes 59,703,296 FMULs, or 48.38% of the original FMUL count.
New vote and comparison instructions each execute 934,912 times. The
instruction reduction is therefore real, unlike address hoisting, but both
matrix-load families and tensor products retain exactly their original work.

| Resource or warp metric | Baseline | Candidate |
| --- | ---: | ---: |
| Registers per thread | 234 | 236 |
| Allocated registers per thread | 240 | 240 |
| Register-limited resident blocks | 2 | 2 |
| Dynamic shared bytes | 49,168 | 49,168 |
| Local bytes reported by resource dump | 0 | 0 |
| Tensor activity | 65.02% | 65.27% |
| Active warps | 16.03% | 16.01% |
| Average warp latency per issued instruction | 6.77 | 6.51 |
| Fixed-wait contribution | 2.19 | 2.35 |
| Barrier contribution | 0.19 | 0.41 |
| Long-scoreboard contribution | 0.72 | 0.75 |
| Short-scoreboard contribution | 0.45 | 0.50 |

Warp contributions are normalized per issue-active event, not additive shares
of kernel completion time. More barrier waiting with unchanged barrier count
does not prove a new barrier was inserted. It is consistent with less uniform
warp arrival after conditional work, but that causal edge has not been isolated.
Fixed-wait samples remain concentrated at HMMA and NOP: HMMA not-issued wait
samples are 106,221 versus 103,839; NOP samples are 42,243 versus 46,053.
These sampled consumers do not identify every preceding dependency.

The measured conclusion is limited: identity rescaling is not the main remedy
for this selected loop. The next hypothesis must change dependency scheduling
or matrix/copy work rather than merely remove an algebraic operation. No new
vLLM/SGLang capture or end-to-end speedup is implied, and the independent
batched projection numerical question remains open.

## Correctness and bounded resources

Memcheck, racecheck, and synccheck pass with zero errors or hazards and empty
stderr at queries 129, rows 2, history 128. Those bounds do not constitute a
whole-model or long-context sanitizer campaign. GPU jobs are serialized;
memory is limited to 8 GiB with swap disabled and a 32 GiB MemAvailable reserve.
The lowest before/after trial observation is 121,338,572 KiB. The GPU is idle
after completion. The two new MoonBit automation files pass warning-denied
native checks and all five focused tests.

## Reproduction identities

- GPU UUID: `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, Spark GB10, sm121.
- NVCC 13.0.88 SHA-256: `fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
- Probe SHA-256: `78901fc5e3222e4394b3fb35746d60ad36a0181f22472c0397c5dee1e5079536`.
- Baseline source SHA-256: `3738eb47bb25c18c6b9cb621d01725f45979fd5cb72b1b5ccbaa9be2f5aebd78`.
- Baseline cubin SHA-256: `616ecc0f88c192e59c3134bfc9390a57d59984cb3b31543956b587858e132cdb`.
- Candidate source SHA-256: `7e60cede0c61790ccac825be0ce6376b738b107cf553c5fb1dd8fbdbe9eb8fef`.
- Candidate cubin SHA-256: `d68d04abed54397547a728ef49613fb1716215a6feba21ecc56484a3697696ce`.

Sealed remote evidence is
`/home/wlc004s/lunaflux-prefill-identity-20261007.s4XgSxDG`.
The local verified copy is
`/tmp/lunaflux-identity-result-20261007.Bw3vfsMI/verified`, including every
workload argument, artifact identity, raw counter report, SASS, and all 15
paired samples. All 142 file-manifest entries verify locally. The downloaded
archive SHA-256 is
`b742c8e807dbf09ec812c06cbc7cb15ee62ab0b44b9bc52804bf770772479633`.

Implementation:
`benchmarks/gpu_pipeline/prefill_identity_rescale_20261007.mbtx` and
`benchmarks/gpu_pipeline/summarize_identity_rescale_20261007.mbtx`.
The shared AKO trial, operand finalizer, SASS reader, and terminal sealer are
retained verbatim with the experiment. Complete contract rebinding rejects
stale baseline-root references before any trial.
