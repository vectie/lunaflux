# AKO: fragment lookahead and rejected joint sibling binding

## Outcome

The two-slot r64-C8 MLP route lowers the complete 8192-token GPU chain by
**6.38–7.33%** in five fresh distinct-weight processes. All 25 whole-chain
pairs improve by at least **5.59%**. The original one-slot alternative had
missed the all-pairs 3% threshold in one confirmation process.

This is a **larger-query-chunk route result**, not a new end-to-end speedup.
Serving still uses its existing 2048-token chunk and unchanged selected
artifacts. Five 2048-token confirmation processes do not establish a robust
win. Longer retained history does not itself make an MLP query chunk larger.
No fresh vLLM/SGLang throughput comparison was run.

The follow-up joint-register binding did not deliver its intended MOV
elimination. It regressed the coowned C16 chain and was removed completely
from production source. Its exact CUDA sources, cubins, timings and instruction
captures remain in a separate rejected experiment archive.

## Fixed experiment contract

GPU: GB10, sm121, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.
CUDA: 13.0.88; nvcc executable SHA-256
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

Query vector: **[32,129,512,2048,4096,8192]**. Runtime row metadata: 32;
hidden/intermediate widths: 1024/3072; BF16 with ordered accumulation and
unchanged activation arithmetic. Every AOT module admits at most 8192 query
tokens. Each probe uses the module's actual primary and down launch geometry,
not a common synthetic launch.

Weight modes are cached and 28 distinct gate/up/down allocations. Those
allocations contain identical deterministic values: an address-working-set
test, not diverse real layer values or model-quality validation.

Each cell has five alternating paired samples for primary, down, and complete
chain, after warmup; each timing averages 20 repeats. Complete intermediate
workspace and output compare bitwise, followed by a sampled scalar oracle.
Only whole-chain time determines the verdict. A robust win requires every
paired chain reduction to be at least 3%; profiler replay timing is separate.

The predeclared budget was two ownership layouts with two-slot lookahead,
each compared to the selected control and its one-slot parent, followed by
at most one counter-guided lowering change. This produced **48 + 48 initial
cells**, plus **10 confirmation processes** for the first experiment. No
additional schedule sweep follows the rejected experiment.

## Experiment 1: existing pure fragment lifetime choice

`ProjectionFoldChoice.fragment_stages=2` supplies a bounded register lifetime
through `FragmentProgram` into CUDA lowering. The shared operand-ring stages,
transfer size, down program and numerical fold remain unchanged.

The exporter checks both primary/down capacity guards, emitted priming and
read-next actions, unchanged down source, and retained bounded row variants.
All one-slot references and the probe are reused from the previously sealed
8192-capacity campaign; matching source bytes and cubin hashes are checked.

### Complete-chain median paired reductions

Positive means lower time. Each entry is **cached / distinct-weight**, in
percent. These are medians of paired reductions, not ratios of independently
sorted latency medians. Small-cell noise and every losing pair remain in raw
records; a favorable median alone is not a robust win.

| Query tokens | r64-C8 F2 vs selected control | r64-C8 F2 vs F1 parent | coowned C16 F2 vs selected control | coowned C16 F2 vs F1 parent |
| ---: | ---: | ---: | ---: | ---: |
| 32 | +2.92 / −0.01 | −0.24 / +0.13 | −1.19 / +1.86 | −0.32 / −0.47 |
| 129 | +2.42 / +2.06 | +0.28 / +0.08 | +11.41 / +0.06 | +0.42 / +0.54 |
| 512 | +4.74 / +8.39 | +6.76 / +9.76 | +13.24 / +10.42 | −0.82 / −0.32 |
| 2048 | +5.77 / +5.18 | +1.09 / +0.51 | +1.99 / +1.38 | +0.20 / +1.16 |
| 4096 | +6.46 / +6.63 | +0.52 / +1.01 | +1.42 / +1.74 | +0.66 / +0.87 |
| 8192 | +7.04 / +7.58 | +1.28 / +0.81 | +1.51 / +1.83 | +1.60 / +1.44 |

Only r64-C8's 4096/8192 cells against the selected control clear the all-pairs
threshold. None of the F2-versus-F1 complete-chain cells does. In particular,
the large coowned 129/512 median gains do not qualify: primary results do not
support those large chain gains, and weak/regressing pairs remain.

### Five new distinct-weight confirmations at 8192 tokens

| Process | Control / F2 median ms | Median paired reduction | Weakest pair |
| ---: | ---: | ---: | ---: |
| 0 | 2.743 / 2.542 | 7.33% | 6.12% |
| 1 | 2.730 / 2.548 | 6.68% | 5.77% |
| 2 | 2.726 / 2.548 | 6.38% | 5.59% |
| 3 | 2.734 / 2.545 | 6.91% | 5.67% |
| 4 | 2.756 / 2.545 | 6.57% | 5.64% |

The five 2048 processes have median paired chain reductions
**[6.07%,5.87%,5.10%,−0.39%,5.73%]**, with weakest pairs
**[0.94%,0.83%,−4.36%,−1.76%,0.59%]**. Thus serving selection is unchanged.
The matched incremental F2-versus-F1 cells also prevent attributing the whole
6–7% route advantage solely to register lookahead: ownership already supplied
most of the earlier gain.

### Separate 8192-primary counters: F1 → F2

| Metric | r64-C8 | coowned C16 |
| --- | ---: | ---: |
| Warp instructions | 230,031,360 → 231,604,224 | 278,495,232 → 301,252,608 |
| Registers/thread | 102 → 116 | 60 → 64 |
| Active warps/SMSP | 3.94 → 3.96 | 7.87 → 7.89 |
| Eligible warps/SMSP | 0.41 → 0.34 | 0.72 → 0.80 |
| Issue-active | 28.65% → 24.25% | 33.19% → 35.22% |
| Tensor-active | 46.02% → 48.27% | 43.84% → 46.73% |
| Short-scoreboard/issue ratio | 1.64 → 1.57 | 1.72 → 2.20 |
| Long-scoreboard/issue ratio | 1.67 → 2.61 | 3.86 → 1.88 |
| Barrier/issue ratio | 1.55 → 2.60 | 5.04 → 5.63 |
| Local spilling requests | 0 → 0 | 0 → 0 |

The counters do not establish a universal dependency-stall reduction.
Coowned lookahead adds **8.17%** instructions; its source-correlated MOV count
increases from **40,058,880 to 43,008,000**, with identical MMA count
**25,165,824**. That motivated the one allowed follow-up.
Ratios are normalized diagnostic statistics, not additive milliseconds or
proof that a sampled waiting instruction identifies its upstream producer.

## Experiment 2: joint MMA constraints, rejected

Hypothesis: bind both siblings' carried accumulators and operand fan-out in
one inline-assembly region, preserving half-major/product-minor ordering,
to avoid repeated native register-tuple copies. This is terminal CUDA register
binding, not a semantic fusion or a new IR layer. The production change has
been removed; the archived `.cu` files own the measured executable identity.

References are the exact F2 cubins from experiment 1, rather than regenerated
parents. Every measured selected primary entry point retains its input bounds,
launch envelope, ordered arithmetic and down companion. Whole-module source
also contained experimental changes to unselected bounded row entry points;
this campaign does **not** qualify those entry points or a serving bundle.

### Joint vs frozen F2 parent: complete-chain reductions

| Query tokens | r64-C8 cached / distinct | coowned C16 cached / distinct |
| ---: | ---: | ---: |
| 32 | −0.04% / −0.34% | −3.83% / −0.63% |
| 129 | +0.16% / +0.06% | −1.85% / −1.23% |
| 512 | −0.32% / −0.09% | −1.02% / −4.54% |
| 2048 | +5.26% / +3.73% | −1.84% / −1.41% |
| 4096 | +0.02% / +0.15% | −1.56% / −1.77% |
| 8192 | −0.06% / +0.08% | −2.90% / −1.60% |

None establishes a robust improvement. Complete raw cells against the selected
control also remain in the archive, but they cannot turn inherited ownership
gains into evidence that the new joint binding works.

### Why the lowering change failed on coowned C16

At 8192 tokens, separate selected-primary counters show **301,252,608 →
323,026,944 instructions**, or **+7.23%**, still 64 registers/thread with zero
spilling. Tensor activity decreases **43.94% → 42.86%**; profiler replay time
increases 2.254 → 2.315 ms. Unprofiled paired chain times, not replay, determine
the rejection.

The source-correlated instruction capture explains why the intended copy
elimination was ineffective:

| Executed opcode | Separate bindings | Joint binding |
| --- | ---: | ---: |
| MOV | 43,008,000 | 42,860,544 |
| HMMA BF16 | 25,165,824 | 25,165,824 |
| Matrix shared load | 18,874,368 | 18,874,368 |
| LEA | 8,110,080 | 17,645,568 |
| IADD3 | 15,040,512 | 17,891,328 |
| WARPSYNC | 12,582,912 | 15,728,640 |

MOV count falls only **0.34%**. Address/control/support work grows while
mathematical work and matrix-load count remain fixed. Broader constraints do
not force an optimal native register allocation, especially under the same
64-register residency ceiling. This is an observed failed hypothesis, not a
claim that all remaining latency has been causally assigned.

## Validation, safety and reproducibility

All 96 initial cells and 10 confirmation processes pass full output/workspace
bitwise comparison plus the sampled scalar oracle (reported error zero).
For the retained F2 experiment, **129 and 8191 query tokens** each pass
memcheck, racecheck and synccheck with zero errors/hazards/warnings and no leaks.
No extra sanitizer or serving admission is claimed for the discarded lowering.

Final projection tests: **107/107**. The paired-trial parser passes 3/3;
geometry, frozen-record renaming, reserve/bitwise sealing and CSV parsing each
pass 1/1. Affected native checks pass with existing migration exclusions
`-79-29-25-20-92-14`. Scoped `moon info` completes with 912 existing warnings
and zero errors. This is not a warning-clean whole-repository release.

GPU jobs are serialized, with MemoryMax=8G, MemorySwapMax=0 and
RuntimeMaxSec=1200. Measured before/after MemAvailable is at least
122,047,404 KiB across the initial campaigns (over 116 GiB), above the 32-GiB
reserve. Process-memory peaks are not asserted to be CUDA allocation peaks.
GPU compute processes are absent at completion.

Experiment 1 remote: `/home/wlc004s/lunaflux-ako-fragment-20261005.Ss6IOvJd`.
Local: `benchmarks/qwen3_comparison/results/ako-fragment-window-20261005.I6SZSod5/measurement.tar.gz`.
SHA-256: `3b0eb5759f48ed3fd1ff2e9dbb8a2b60a9dacdcacab378f3b95b482b4eafd1c1`.
All **479 manifest files** verified locally.

Rejected experiment remote: `/home/wlc004s/lunaflux-ako-joint-20261005.jID3EUKR`.
Local: `benchmarks/qwen3_comparison/results/ako-joint-binding-20261005.FYgmhMvy/measurement.tar.gz`.
SHA-256: `3a826ae90ec2bc4642038866c6a7c572ce8c1e49f5d74e1fc0ac2a794072c3eb`.
All **377 manifest files** verified locally. Archives were created outside
their input directories and downloaded without overwriting prior campaigns.

The repeat counter-driver invocation initially hit a create-new filename
collision before any GPU launch. A separate invocation label and versioned
driver resolved it; measurement outputs were not overwritten. Native helper
builds report an existing async C `write` warning; local helper test builds
also report empty-library warnings. Neither is a GPU correctness failure.

Committed changes are offline exporter/measurement support and this report.
The existing immutable fragment IR and CUDA lowering remain the production
implementation; no request-path JIT, model-name special case, profiling,
cryptography, filesystem validation or new heap allocation was introduced.
Before increasing serving chunks, qualify the entire larger-chunk graph,
workspace and attention chain, then measure selected-symbol end-to-end latency.
