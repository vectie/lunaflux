# Selected prefill counters and exporter consistency

## Result

The new selected-kernel captures do not support treating long prefill as
primarily global-load-bound. Tensor activity is 63–65%, long-scoreboard share
9–11%, barrier share 5–7%, and fixed instruction/dependency wait share about
33%. Both captures have zero local-memory sectors. These are warp-latency
fractions, not an additive explanation of the framework completion gap.

A benchmark propagation defect was found and corrected: the saved export
command pointed to an older executable that omitted page-batched historical
lookup lowering. Its stable candidate IDs were unchanged. Comparing its
alternatives to the newer serving binary confounded the lowering revision with
geometry. The corrected controller first regenerates the selected control,
checks numerical and geometry fields, and requires an identical CUBIN before
timing alternatives. A regression rejects matching candidate IDs with different
source hashes.

The corrected three schedule alternatives and one query-width alternative all
regressed. None is selected. Production inference source and serving artifacts
remain unchanged; this round fixes experiment consistency, not the remaining
performance gap or concurrent numerical-parity concerns.

## Fixed workload and selected artifacts

Spark .179, GB10 sm121, 48 SMs, UUID
`GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, driver 580.178.04. Qwen3-0.6B BF16
attention geometry: 16 query heads, 8 KV heads, dimension 128, page size 8,
maximum query tokens 2048, maximum rows 32. The isolated probe uses synthetic
immutable current/history operands, not live serving inputs.

The selected kernel is `lunaflux_attention_prefill_tile_compiler_exp2_v1`,
candidate 30322, Q64/K64, single-stage split-copy lifetime, explicit
`approx-base2-f32-v1` law. Observed grid is 63×16×1, block size 128, registers
234. The recipe's larger admitted grid is not substituted for this observed
runtime grid.

- Selected source SHA-256:
  `3738eb47bb25c18c6b9cb621d01725f45979fd5cb72b1b5ccbaa9be2f5aebd78`.
- Selected and freshly regenerated CUBIN SHA-256:
  `616ecc0f88c192e59c3134bfc9390a57d59984cb3b31543956b587858e132cdb`.
- Selected recipe SHA-256:
  `750750f6cdfc87fde42bbbe48efd61d3a1b494a17e6f3dd7409b869cb2e0bc0a`.
- Probe SHA-256:
  `78901fc5e3222e4394b3fb35746d60ad36a0181f22472c0397c5dee1e5079536`.
- CUDA 13.0.88 nvcc SHA-256:
  `fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

The corrected exporter is built from committed source `715ad886`, not the
unrelated dirty working tree. It reproduces the selected control exactly.
Compilation uses O3, `fmad=false`, FTZ off, precise division/square root and
the recipe register ceiling. No production JIT or hot-path validation was added.

## Fresh selected counters

One selected invocation per workload, Nsight Compute 2025.3.1. Counter replay
duration is not an unprofiled latency or end-to-end tok/s measurement.

| Counter | Q1792 R1 H30720 tail | Q2048 R2 H28672 |
| --- | ---: | ---: |
| Replay duration ms | 6.983 | 7.181 |
| Executed warp instructions | 844,413,696 | 891,530,112 |
| Tensor active elapsed percent | 63.41 | 65.03 |
| Long-scoreboard warp-latency share percent | 9.17 | 11.22 |
| Barrier share percent | 6.63 | 5.11 |
| Short-scoreboard share percent | 6.94 | 7.01 |
| Fixed dependency wait share percent | 32.76 | 33.28 |
| Local load and store sectors | 0 / 0 | 0 / 0 |

GB10 DRAM-byte/throughput metrics requested in this capture are unavailable;
the report does not substitute L2 sectors or async instruction counts for DRAM
bytes. There is no new matched reference counter capture in this round.

## Corrected paired timing decisions

Five alternating unprofiled pairs per cell, 30 event repetitions per member,
full-output error checks and sampled FP64 oracle. The fixed maximum error is
0.003; differing KV fold boundaries need not be bitwise equal. The performance
gate requires every pair to improve by at least 3%. All samples are retained.

Positive paired time change below means slower. Percentages are medians of
paired ratios, not ratios of the two latency medians. The short control has
Q2048/R16/H2048, rather than changing the long inputs to find a win.

| Alternative | Tail baseline / candidate µs | Tail change | R2 baseline / candidate µs | R2 change | Short control change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q64 K64 double-buffer 30320 | 6285.36 / 12979.61 | +109.64% | 6562.70 / 13467.22 | +105.36% | +18.32% |
| Q64 K32 split-copy 30324 | 6304.01 / 6940.27 | +11.07% | 6669.80 / 7897.02 | +18.33% | +2.98% |
| Q64 K128 split-copy 32004 | 6355.73 / 7525.58 | +18.92% | 6661.79 / 7889.24 | +18.66% | +0.75% |
| Q128 K64 split-copy 31022 | 6322.66 / 6856.29 | +8.94% | 6602.98 / 7690.49 | +16.24% | +0.63% |

The Q128 extension uses the existing pure frontier, subgroup query ownership,
metadata bucket and ordered physical fold. It introduces no CUDA renderer or
model-specific rule. Its production edits and fixtures were removed after the
loss; the exact two-file experimental overlay is preserved with its evidence.
The [earlier Q128 experiment](BENCHMARK_AKO_QUERY_REUSE128_2026-10-05.md) already
found the same tradeoff. This exact-envelope recheck is confirmation, not a new
successful optimization.

No new alternative counters were collected here. The earlier counter results
show why merely widening is not sufficient: Q128 halves async copies but
retains almost all tensor work, moves to one resident CTA, and increases
collective/math-pipeline waiting. The current timings confirm the regression;
they do not prove that each earlier counter contribution is unchanged.

## Remaining work

The [final October 6 serving benchmark](BENCHMARK_DECODE_ROUTE_AND_FAIRNESS_FIX_2026-10-06.md)
remains the end-to-end authority: 4K/64/C16 is 225.13 tok/s versus 243.03/241.85,
and 32512/64/C2 is 16.95 versus 17.58/18.85. Completion gaps are respectively
8.0%/7.4% and 3.7%/11.3%. They are not replaced by these kernel-event timings.

Against SGLang the last traced long-C2 deficit is mainly aggregate prefill
attention; against vLLM it includes decode and other kernels, with faster Luna
prefill offsetting part of them. A new experiment must compare exact executed
operator chains and independently quantify the ordered fragment/softmax
dependency path. Repeating bigger tiles, query retention or more pipeline
stages alone is unsupported by these rejected results. Reference chunking and
current/history decomposition also differ; isolated tail timing cannot assign
all aggregate prefill cost to one instruction.

Functional layering remains semantic/numerical contract → pure ownership and
schedule → explicit effects and storage lifetime → device lowering. The new
checks are offline measurement orchestration; no model/backend details enter
the scheduler or token path. No additional IR layer is justified by this round.

## Checks and preserved evidence

GPU work was serialized. Probe/controller owners use an 8 GiB/no-swap cap,
900-second bound and 32 GiB MemAvailable reserve. Corrected timing samples stay
above 121,020,512 KiB available. The terminal GPU is idle. Existing host swap
usage is not described as zero. No rejected variant was deployed, so no new
production sanitizer admission is claimed.

Experimental native tests passed 31 strategy and 89 source tests. Restored
production passed 189 affected native tests: 30 strategy, 88 source, 39 tile
schedule and 32 attention physical IR. All four offline helpers passed
warning-denied native checks; their two regressions passed. They cover CSV grids
with commas and rejection of a stale lowering behind an unchanged candidate ID.
Existing MoonBit migration warnings are excluded only at affected-package
command boundaries; no clean whole-tree claim is made.

Remote roots, including all rejected/confounded attempts:

- Counters and rejected stale-exporter comparisons:
  `/home/wlc004s/lunaflux-prefill-recheck-20261007.ZXGzWEev`.
- Corrected pinned-source alternatives:
  `/home/wlc004s/lunaflux-prefill-pinned-20261007.kvU0yviC`.
- Rejected Q128 overlay and timing:
  `/home/wlc004s/lunaflux-query128-20261007.Dk9kKWIH`.

Measurement archives retain raw outputs, commands, cubins, recipes, manifests
and numerical samples without build caches or model weights. Archive SHA-256:

- Confounded attempt and fresh counters:
  `905463a2d539c00e9d97d86e089de156c2f6d53ca2b1ef2366cffea81e6b042e`.
- Corrected geometry:
  `dcc476b7466a921697f9106d9733ba5c1407a1f1527d852ee0f2cf7a2c6b7084`.
- Wide-query rejection:
  `d3d9aec9df3559014256be5f16e83e01e8876f710f760d72caa20f0d9bcc16e5`.

Local destination: `/tmp/lunaflux-prefill-recheck-20261007.gSegtQ6r`.
All three downloaded archive hashes match the values above, and all extracted
`MEASUREMENT.sha256` manifests verify locally. Downloads used distinct new
filenames without overwriting existing evidence.
