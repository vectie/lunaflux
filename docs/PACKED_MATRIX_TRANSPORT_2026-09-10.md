# Packed matrix transport: fresh full-runtime benchmark

## Result

Against the preceding tested LunaFlux `987db7a`, this iteration improves
Qwen3-0.6B 59-input/256-output throughput **1280.80 -> 1492.35 tokens/s at C8
(+16.5%)**, and **1978.49 -> 2529.45 at C16 (+27.8%)**.
The same-day vLLM/SGLang throughput ratios are now **1.24x/1.17x at C8** and
**1.28x/1.23x at C16**. These are narrower gaps, not overall parity.

The head's selected cold-cache GPU time is **744.320 us**, versus the
matched vLLM replay's **744.288 us**. DRAM throughput reaches **95.97%**
of sustained peak, versus 55.84% in the preceding implementation. Head
register spill instructions are zero in the final selected replay.
QKV cold replay is also close to the baseline. Output and down still lag.

Long-input serving remains substantially behind, mainly before the first
token in this workload. This iteration does not claim to finish long-prefill
attention, all shapes, all models, or production promotion.

## Frozen comparison

- Production implementation source: `ed6f572a6d6ca39bf6bb365c4e8b0e634b0a55e2`.
  Archive SHA-256:
  `17a82c87b0ddc5f8da73f730b3ed7b59152dad0a47c44f870d4d1aa91e1de8a6`.
  Subsequent commits add regression tests, measured tuning and this report,
  not different production kernel logic. Unrelated dirty work is excluded.
- Final runtime: `/dev/shm/lunaflux-concurrency-packed-matrix-r5/runtime-final-r2`.
  All standard and fused artifacts are regenerated from that clean source.
- Exact tuning input:
  [projection-tuning.v1](../benchmarks/operand_supply_20260910/projection-tuning.v1).
  Head buckets 8/256 and composite MLP bucket 16 have fresh measured records;
  the other three entries retain their historical measurements.
- Same Qwen3-0.6B BF16 model, token-ID prompts, greedy selection, ignored EOS,
  prefix reuse disabled, same 5060 Ti 16 GB / 36 SMs.
  GPU UUID `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`,
  PCI `00000000:17:00.0`, CUDA 13.1.115.
- vLLM 0.24.0 and SGLang 0.5.2 (graph maximum batch 32) retain the existing
  same-day measurements/configuration. They were **not rerun in this iteration**.
  The installed linear paths execute captured PyTorch/cuBLAS BF16 GEMMs;
  neighboring source repositories are not assumed to match every installed
  optional branch. See the [matched diagnosis](CONCURRENCY_GAP_CAUSES_2026-09-10.md).
- Four input/output vectors x C1/2/4/8/16; one warmup and three timed
  synchronized finite bursts per cell. No profiler or other GPU workload
  runs concurrently with ordinary throughput measurement.
  This is not a saturation, independent quality, or energy-efficiency test.

## What changed, and why it remains functional

The existing pure compiler already expresses independent output maps over
ordered K16 folds, row demand, sparse gather/scatter and operand lifetimes.
This change improves **CUDA lowering of those plans plus offline strategy
selection**. It does not introduce another language-level optimization pass
or claim that every compiler operation is now optimal.

1. Use the same compact, row-preserving transport permutation for all eligible
   matrix pipelines. Delete the obsolete padded-layout abstraction/allocation.
   Physical storage now agrees with the producer and consumer address map.
2. Lower regular QKV/output fragments through packed `ldmatrix` loads, as
   selected-row projections already could. Each accumulator element has a
   unique owner and is stored directly, eliminating shared output staging,
   rereads and their synchronization.
3. Render input and weight copy domains separately. There is no runtime
   input-versus-weight selector inside every copy iteration. The masked input
   prime is synchronous; subsequent immutable operand transfers overlap the
   ordered fold through the existing two-slot pipeline.
4. Measure existing head output-map distributions. Two workgroup tiles use
   shared transport instead of the single-tile register-gather path. Select
   this through the existing offline tuning contract for buckets 8 and 256.
5. Measure the **complete** bucket-16 MLP, including the dependent down
   consumer. Select the two-group strategy (64-thread launch), not a
   gate-only winner; the compiler owns the final output-map partition.

Floating-point K16 update order, sparse row addresses and final BF16 rounding
stay unchanged. There is no reassociation, atomic accumulation, model-name
branch, runtime JIT, token-step allocation or cryptographic/file-system work.
CUDA instructions, warp mapping and shared-memory effects stay in the CUDA
backend; model builders and scheduling acquire no NVIDIA details. Device- and
shape-specific tuning records are normal backend data, not universal timings.

### Isolated experiments

- Output: packed loads/direct stores/compact storage alone improved C8 only
  about 6% (27.38 -> 25.75 us). Specializing the producer domains brought it
  to about 15.06 us. The larger gain is not attributable to layout alone.
- Head: removing unused upper-row register fragments reduced registers from
  128 to 117 and removed a 24-byte stack, but slowed C8 roughly 18%.
  That experimental source change was reverted. Low register counts alone
  are not a performance goal.
- Head two/four-tile distributions both measured about 740 us at eight rows.
  Two tiles use less shared storage. The wider envelope was separately
  measured at 256 tokens/32 selected rows; no C8 time is relabeled as a
  measurement of the wider bucket.
- C16 MLP: gate/up alone changes 32.118 -> 13.566 us, while down changes
  16.229 -> 17.415 us (a small regression). The actual dependent two-kernel
  graph changes **48.039 -> 30.915 us (1.55x)**. That complete operation is
  the tuning target. The four-group composite is about 31.32 us.
- A one-group C16 alternative makes gate/up still faster but down worse.
  Fresh dependent composite means: one group 32.606 us, two groups 30.913 us,
  four groups 31.302 us. More CTAs or fewer warps alone did not improve down.
  The one-group option was not selected.
- One scratch export failed because the newly appended tuning record was
  not canonically ordered. Its empty stdout/stderr and exit -6 are retained.
  The corrected snapshot was sorted with the existing codec's ordering and
  exported into a new directory. No failed output was relabeled as a pass.

## Ordinary throughput

Output tokens/s, arithmetic means of three trials. Previous LF is `987db7a`;
final LF includes the measured C16 MLP selection. Full per-trial data is
preserved. Small C1 fluctuations are not credited as gains.

| Input / output | C | Previous LF | Final LF | Gain | vLLM | SGLang | vLLM / LF |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 59 / 256 | 1 | 247.58 | 247.74 | 0.1% | 277.76 | 268.46 | 1.12x |
| 59 / 256 | 2 | 351.25 | 414.58 | 18.0% | 520.15 | 494.21 | 1.25x |
| 59 / 256 | 4 | 673.09 | 790.33 | 17.4% | 976.17 | 947.12 | 1.24x |
| 59 / 256 | 8 | 1280.80 | 1492.35 | 16.5% | 1850.05 | 1746.46 | 1.24x |
| 59 / 256 | 16 | 1978.49 | 2529.45 | 27.8% | 3229.44 | 3117.26 | 1.28x |
| 128 / 128 | 1 | 240.91 | 239.85 | -0.4% | 268.72 | 258.59 | 1.12x |
| 128 / 128 | 2 | 340.88 | 402.73 | 18.1% | 496.45 | 476.20 | 1.23x |
| 128 / 128 | 4 | 646.20 | 753.68 | 16.6% | 933.17 | 896.25 | 1.24x |
| 128 / 128 | 8 | 1216.15 | 1420.92 | 16.8% | 1744.47 | 1655.24 | 1.23x |
| 128 / 128 | 16 | 1843.95 | 2312.39 | 25.4% | 2968.13 | 2864.40 | 1.28x |
| 512 / 64 | 1 | 208.93 | 209.15 | 0.1% | 244.95 | 242.45 | 1.17x |
| 512 / 64 | 2 | 290.91 | 334.51 | 15.0% | 433.41 | 422.91 | 1.30x |
| 512 / 64 | 4 | 485.77 | 556.52 | 14.6% | 744.19 | 719.21 | 1.34x |
| 512 / 64 | 8 | 791.35 | 905.13 | 14.4% | 1183.36 | 1133.58 | 1.31x |
| 512 / 64 | 16 | 1017.56 | 1206.13 | 18.5% | 1650.73 | 1604.18 | 1.37x |
| 1528 / 32 | 1 | 133.89 | 136.80 | 2.2% | 180.46 | 175.52 | 1.32x |
| 1528 / 32 | 2 | 166.67 | 187.33 | 12.4% | 268.54 | 255.70 | 1.43x |
| 1528 / 32 | 4 | 216.95 | 242.12 | 11.6% | 367.46 | 344.71 | 1.52x |
| 1528 / 32 | 8 | 267.78 | 298.02 | 11.3% | 444.70 | 419.90 | 1.49x |
| 1528 / 32 | 16 | 288.83 | 322.96 | 11.8% | 494.05 | 471.02 | 1.53x |

The intermediate run with head/QKV/output improvements but without the C16
MLP record measured 2244.39 tokens/s at short C16. Adding that record brings
it to 2529.45 (+12.7% over the intermediate). C8 remains approximately
1493-1497 tokens/s, consistent with an unchanged bucket.

### The long-input gap is concentrated before first token

1528 input / 32 output tokens, client-observed means:

| C | Metric | Final LF | vLLM | SGLang |
| ---: | --- | ---: | ---: | ---: |
| 8 | TTFT, ms | 500.63 | 222.50 | 246.33 |
| 8 | Decode, ms/output token | 11.13 | 10.84 | 11.60 |
| 16 | TTFT, ms | 943.40 | 408.40 | 436.23 |
| 16 | Decode, ms/output token | 19.59 | 18.97 | 20.83 |

The decode portion is close to vLLM and slightly faster than SGLang in these
long-input cells, while TTFT is more than twice theirs. That localizes the
remaining serving gap to the pre-first-token phase; it does **not** prove
which attention/projection/host operation owns every millisecond. A fresh
phase-aligned prefill trace is needed before attributing the entire TTFT
difference to one kernel or to scheduling. TTFT is not a pure GPU counter.

## Selected-kernel counters, not warm-cache serving estimates

Cold-cache NCU replay at C8, using the actual compiled-set / recipe / release
joined CUBINs and launch geometry. Time is microseconds. Previous LF and vLLM
are the matched campaigns; final LF is newly captured.

| Kernel | Previous LF | Final LF | vLLM | Final LF / vLLM |
| --- | ---: | ---: | ---: | ---: |
| QKV | 24.800 | 22.912 | 22.656 | 1.01x |
| Output | 32.736 | 22.752 | 13.152 | 1.73x |
| Gate/up | 32.960 | 32.960 | 34.464 | 0.96x |
| Down | 29.440 | 29.568 | 20.256 | 1.46x |
| Head | 1279.040 | 744.320 | 744.288 | 1.00x |

Gate/up includes activation; the baseline replay is GEMM only. This is not a
claim that the complete baseline MLP has more work or worse serving latency.

Final C8 DRAM throughput is QKV 83.57%, output 42.25%, gate/up 86.85%,
down 48.76%, head 95.97%. Head executes more warp instructions than the
preceding implementation (39.18M vs 25.29M), yet is much faster because it
supplies memory nearly at bandwidth limit. Head is no longer the place to
expect another 1.7x bandwidth improvement on this GPU with the same bytes.

Warm repeated-graph replay gives QKV 19.16 -> 10.28 us, output 27.37 -> 15.03 us,
and head 1284.28 -> 740.44 us. QKV's warm speedup is much larger than its
cold speedup: its weights benefit from cache reuse. Neither warm replay
numbers nor NCU timings are substituted for end-to-end throughput.

The final C16 cold times are QKV 22.464, output 19.680, gate/up 33.184,
down 29.376, head 752.064 us. Final C16 gate/up barrier stalls are 2.10%;
the new head uses 88 registers with no reported spill instructions.
Output/down remain priority projection gaps, but changing workgroup size
  again needs instruction/memory measurements; the one-group down experiment
already disproves a simple "more blocks must be faster" explanation.

### Conflict accounting remains explicit

Kernel IDs: 0 QKV, 1 output, 2 gate/up, 3 down, 4 head.
All ten new replays have zero source-attributed excessive shared wavefronts.
Hardware aggregate counters are **not** all zero and are not the same metric.

| Kernel ID | Rows | Source excess | Hardware load conflicts | Hardware store conflicts |
| --- | ---: | ---: | ---: | ---: |
| 0 | 8 | 0 | 62 | 8428 |
| 1 | 8 | 0 | 0 | 3114 |
| 2 | 8 | 0 | 77 | 1657 |
| 3 | 8 | 0 | 0 | 0 |
| 4 | 8 | 0 | 10210 | 32126 |
| 0 | 16 | 0 | 0 | 7 |
| 1 | 16 | 0 | 0 | 65 |
| 2 | 16 | 0 | 265 | 0 |
| 3 | 16 | 0 | 0 | 0 |
| 4 | 16 | 0 | 21185 | 3283 |

The head switches from register gather to shared transport and its hardware
totals become nonzero, even while the source excess stays zero and measured
bandwidth reaches baseline performance. We do not call the hardware totals
zero, hide them, or use them interchangeably with source-attributed excess.

## Verification and remaining limits

- Clean Linux source: `moon info`, format check, warning-denied native check
  and **2974/2974 tests** passed for the tested implementation commit.
- Updated local projection regression package: **64/64** passed. New tests
  cover head row-8/row-256 transport and medium-row MLP companion/workspace
  contracts. New test-only commits do not change the frozen GPU source.
- Final actual CUBINs: **55 simple + 55 varied-exponent BF16 pairs** are
  bitwise equal to the preceding version, including scalar boundaries,
  tails, sparse head row selection and 1024-token cases.
- First runtime: **75 sanitizer checks** (memcheck with full leak checking,
  racecheck, synccheck). After adding the C16 MLP record, **6 additional**
  checks cover the two changed C16 kernels. All passed.
- Two complete 20-cell ordinary LF campaigns passed, with expected output
  token counts and deterministic resource release. No production cutover.
- Final versus previous named measured sequences: **365/372 exact**.
  The seven differences are C4 runs with already-observed batch-dependent
  sequences. All **496** final sequences including warmups occur in the
  previous same-prompt sequence pool. This is not proof of batch-invariant
  generation or an independent model-quality qualification.

Remaining performance work is not "all solved": pre-first-token processing
for long inputs, output/down operand supply, and non-projection serving costs
still need targeted measurement. Head and QKV parity in these replays is not
universal engine parity, multi-GPU validation, or production readiness.

## Artifacts

Both benchmark generations, raw NCU/SASS, paired launches, sources/CUBINs,
build/test logs, rejected experiments, tuning records and orchestration are
archived together. Production model weights are not duplicated into the archive.
Downloaded without overwrite to
`benchmarks/qwen3_comparison/results/packed-matrix-ed6f572-20260910/`:

`lunaflux-packed-transport-ed6f572-20260910-r1.tar.gz`

Remote and local SHA-256 agree:

`a5092506fa9f9c720ba533330c94c8e70f697a528ca8b88aeaf8cee9109692fb`

The archive retains the failed export and rejected experiments, not just the
winning paths. The only removed intermediates were this campaign's rebuildable
debug cache and an unused duplicate numeric-weight file; the matching original
weight artifact, source archives, release binaries, logs and receipts remain.
