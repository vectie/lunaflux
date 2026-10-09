# Reference prefill schedule: smaller storage, measured complete execution

## Scope and hypothesis

The previous whole-service comparison left two different gaps: short C8 decode
and long C1 prefill. Retuning the already-fast long C1 decode partition cannot
close the latter. This bounded AKO iteration tests three **prefill** terminal
schedules, not another semantic IR layer or a model-name heuristic.

The selected reference wrapper uses the pinned paged split-KV device function
with one partition. Its Q64/KV128 specialization reserves 81,920 shared bytes
per block. The alternatives are Q64/KV64 (49,152 bytes), Q128/KV64 (65,536), and
Q64/KV32 (32,768). Less storage may allow better latency hiding, but additional
KV iterations or register pressure may erase the gain. This is a falsifiable
schedule hypothesis, not an occupancy-based performance promise.

The reference's regular prefill function is not a drop-in replacement for the
engine's paged CSR input. No dense cache gather or unvalidated address-space
reinterpretation was introduced.

## Functional architecture

`ReferencePrefillSchedule` is a typed immutable **CUDA terminal** choice. It
jointly determines source specialization, key rounding and shared-memory
requirements. Generated static assertions check the pinned library's storage
trait. Models, scheduler, semantic/numeric IR, KV effects and runtime dispatch
remain unchanged. No JIT, profiling or tuning was added to a token step.

The integrated alternatives are Q64/KV128 and Q64/KV64; the exporter emits both
and keeps the old default. Both bind the same output continuation and explicit
BF16-probability/base-two-exponential law. The bundle requires identical module
and launch resources in its two equivalent prefill slots. The independently
defined decode adapters reject a prefill-only schedule change.

This numerical permission is not a bitwise-equivalence claim. Changing reduction
tiles can change BF16 outputs. Kernel tolerance checks and fixed output token
counts also do not prove model-quality parity.

## Fixed experiment

- Clean source `e5b93652` plus the narrowly scoped schedule/export/probe changes;
  unrelated dirty-tree work is excluded.
- Spark .179, GB10 sm121, GPU `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`,
  PCI `0000000F:01:00.0`.
- Qwen3-0.6B, BF16 Q/K/V/output, F32 accumulation, BF16 probabilities, head
  dimension 128, GQA 16:8, page size 8. Reference arithmetic/compile flags stay
  unchanged: CUDA 13.0.88, O3, sm121, relaxed/extended CUDA templates, lineinfo.
- GPU workloads serialized; 32-GiB live host reserve; serving 64-GiB memory cap
  with no extra swap; bounded user-systemd experiment stages.
- Five alternating paired kernel trials. Independent sampled scalar referee,
  pairwise maximum absolute difference ceiling 0.003, deterministic rerun,
  read-only operands and inactive-output preservation. History includes 32K,
  tail queries and fragmented pages.
- Whole service: fixed varied token-ID vectors, 64 generated tokens, cells
  512/C8, 8192/C8, 32512/C1; fresh-start control/candidate/candidate/control,
  one warmup plus three measured waves per cell/start.

The baseline service is the prior four-partition package at
`/home/wlc004s/lunaflux-overall-best-20261009.b9EitVKQ/serving/long`.
The parent runtime, vendor output/down projection, residual normalization and
decode module are reused. Both A/B arms use the same newly rebuilt worker with
the typed reference-resource admission fix below. New bundle-scoped captured decode measurements must
be regenerated rather than copying an authenticated scope from another bundle.

## Evidence and validation ledger

Remote campaign:
`/home/wlc004s/lunaflux-prefill-choice-20261009.vIZCtp8x`.

The Q64/KV64 candidate passes the independent exact-size CSR oracle, ragged
mixed/pure-decode writer checks, memcheck/full leak, racecheck, initcheck and
synccheck. Q128/KV64 compiles with 156-byte spill loads/stores; it is retained
as an experiment, not integrated. Q64/KV32 remains diagnostic as well.

Preserved preparation failures: the initial archive carried an AppleDouble
metadata file which MoonBit correctly rejected as invalid source; the clean
archive excludes those records. The initial serving harness confused the
test argv indices (including argv[0]) with executable argument indices; bundle
assembly rejected it before GPU serving. Both failed attempts and corrected
commands are retained. No production deployment was changed.

The first complete package passed offline release validation but failed worker
startup. Debugging exposed a missed propagation boundary:
`engine/device_step/qwen_fused_attention_admit.mbt` still required exactly
81,920 shared bytes. The exporter alone had been updated. Worker startup now
checks both immutable schedule geometries, and its regression tests admit
49,152 while rejecting 49,151 / 49,153 / 32,768 / 65,536. The admission runs
once, not in token execution. Both service arms use this identical rebuilt
worker to isolate the schedule change. The aborting run remains preserved;
it has no valid candidate service timing.

The calibration helper also invoked an older exporter through its frozen
runtime source link. All 42 captured route measurements and their memory/race/
sync checks completed; only the final binding failed. The current exporter
successfully bound those same genuine, current-bundle measurements. The failed
binding logs were not relabeled or overwritten. A later benchmark preflight
also rejected an incomplete launch-overlay directory before serving; its
missing preparation metadata and worker-identity link were supplied before the
complete four-session rerun. None of these failed attempts contributes timings.

## Results against previous best

Five alternating paired kernel trials, median microseconds, **after correcting
the probe to use the compacted Q64 launch domain**:

| Rows × query tokens; history | Q64/KV128 | Q64/KV64 | Time reduction |
| --- | ---: | ---: | ---: |
| 8 × 64; 512 | 44.023 | 44.593 | −1.29% |
| 8 × 256; 8,192 | 2,478.202 | 2,118.487 | 14.52% |
| 1 × 2,048; 8,192 | 1,702.336 | 1,484.079 | 12.82% |
| 1 × 2,048; 32,768 | 7,465.924 | 6,442.398 | 13.71% |
| 1 × 7; 19 | 6.180 | 5.758 | 6.83% |

The initial exploratory sweep launched the 64-CTA-X maximum envelope, including
inactive tiles. It overstated the short kernel benefit. Those observations stay
in `kernel-summary.json`; the table above uses `exact-probe/summary.json` and
the actual selected cubin, not the exploratory build. Its uniform-row Q64
domains have X=8/32/32/32/1. The short 512-history probe is slightly slower;
there is no clear short-serving improvement to claim. The independent
end-to-end A/B already used the real runtime and is unaffected by this probe
correction (`9a0d4445`). Tiny diagnostic domains are not serving-bucket proof.

In the maximum-envelope exploration, Q64/KV32 wins the tiny cases but is slower than KV64 on both single-row long
cases. Q128/KV64 loses 55.47% on the short case and 5.18% on 32K; its spill
and weaker complete-workload tradeoff exclude it. These alternatives remain in
the evidence, not selected by a model-specific heuristic.

Unprofiled fresh-start control/candidate/candidate/control, six measured waves
per arm and cell, 64 generated tokens per request:

| Input / concurrency | Previous wall ms | KV64 wall ms | Completion reduction | Previous → KV64 output tok/s |
| --- | ---: | ---: | ---: | ---: |
| 512 / C8 | 713.5 | 707.5 | 0.84% | 717.59 → 723.67 |
| 8,192 / C8 | 4,695 | 4,579 | 2.47% | 109.05 → 111.81 |
| 32,512 / C1 | 3,879.5 | 3,677 | 5.22% | 16.50 → 17.41 |

The short-context change is too small for a strong improvement claim: candidate
session medians were 712 and 703 ms, versus control 714 and 713 ms. There is no
observed material short-context regression. The long-context result repeats in
both starts: 3,667/3,684 ms candidate versus 3,872/3,891 ms control.

32K TTFT falls from 2,426 to 2,223.5 ms; TPOT stays 22.762 versus 22.706 ms.
The approximately 202-ms completion saving is therefore a prefill improvement,
not a decode speedup. The exact 32K generated-token vectors match across arms;
C8 vectors do not, so this is not generation-quality or batch-invariance proof.
Minimum observed available host memory stayed above 99 GiB in both arms.

Implementation commit: `49336e36`. Clean native full suite: **3,477/3,477**;
Linux device-step regression suite: **228/228**. Warning-denied checks retain
the repository's existing migration exclusions `-79-20-29-25-92-14`; this is not
a claim that all repository deprecations are resolved.

## Fresh vLLM / SGLang comparison

Same GPU, model, varied token-ID vectors and fixed 64-token output; two fresh
starts per engine in reversed order, one warmup plus three measured waves per
cell/start. Frozen reference images/configurations are those from
[the immediately preceding comparison](BENCHMARK_WORKLOAD_DECODE_SELECTION_2026-10-09.md).
All framework timings below are from this new campaign, not copied baselines.

| Input / concurrency | LunaFlux output tok/s | vLLM output tok/s | SGLang output tok/s | Luna completion overhead vs vLLM / SGLang |
| --- | ---: | ---: | ---: | ---: |
| 512 / C8 | 722.65 | 776.93 | 760.21 | +7.51% / +5.20% |
| 8,192 / C8 | 111.50 | 113.71 | 112.43 | +1.99% / +0.83% |
| 32,512 / C1 | 17.30 | 16.14 | 17.34 | −6.70% / +0.24% |

Wall medians (Luna / vLLM / SGLang) are 708.5 / 659 / 673.5 ms,
4,592 / 4,502.5 / 4,554 ms, and 3,699 / 3,964.5 / 3,690 ms. Negative
overhead means less time. The 0.24% long-context difference is a practical tie,
not evidence that one engine is consistently faster.

Per-request median TTFT / TPOT in milliseconds:

| Input / concurrency | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| 512 / C8 | 82.5 / 9.651 | 86 / 8.897 | 95 / 8.810 |
| 8,192 / C8 | 1,339 / 47.548 | 1,376 / 45.897 | 1,164.5 / 53.413 |
| 32,512 / C1 | 2,239.5 / 22.881 | 2,448.5 / 23.651 | 2,124.5 / 24.492 |

Short C8 still loses primarily after the first token. At 32K, SGLang retains
an approximately 115-ms TTFT advantage, offset by LunaFlux's lower TPOT. Do not
add pooled C8 request medians as if they were a critical-path decomposition.
The A/B and three-framework tables are distinct campaigns; the later Luna
32K median is 3,699 rather than 3,677 ms. Keep both rather than cherry-picking.

The first 32K output vector matches SGLang, not vLLM; all three are repeatable
on C1. C8 output vectors differ across engines and some repeat waves. This
limits generation-quality claims, despite equal measured token work and passing
bounded kernel correctness. Minimum available memory is 98.94 GiB for Luna,
65.13 GiB for vLLM and 65.44 GiB for SGLang: all exceed the 32-GiB reserve.

## Selected dispatch and hardware explanation

Nsight Systems observes the selected mixed-prefill symbol with **49,152 dynamic
shared bytes, 232 registers/thread and block size 128** in all three serving
request windows. Long C1 uses grid `(32,16,1)` and 448 prefill launches; decode
still uses the separate four-partition partial/merge chain. This is real
serving selection, not merely an exported source file.

Nsight Compute replays the same baseline/selected cubins on a shape-matched
full long-prefill chunk: C1, 2,048 queries, 30,720 history tokens, fragmented
pages, grid `(32,16,1)`, block 128. Operands are deterministic synthetic BF16,
not captured model activations. This controlled replay explains a selected
schedule; it is not whole-service latency. Clock/cache control are disabled.

| Counter | KV128 | KV64 |
| --- | ---: | ---: |
| Dynamic shared bytes/block | 81,920 | 49,152 |
| Shared-memory resident-block limit | 1 | 2 |
| Register resident-block limit | 2 | 2 |
| Registers/thread | 242 | 232 |
| Active-warps occupancy | 8.74% | 19.42% |
| Eligible warps/cycle | 0.17 | 0.30 |
| Issue-active percentage | 16.97% | 24.36% |
| Tensor-active percentage | 60.18% | 72.16% |
| Warp instructions | 589,928,448 | 641,360,896 |
| Profiled replay duration | 6.842 ms | 5.865 ms |

The storage geometry removes a one-block shared-memory constraint. Instruction
count rises **8.72%**, yet the replay is **14.27% shorter**, with more eligible
warps and higher tensor activity. This supports improved latency hiding and
shows why minimizing instruction count alone is the wrong objective. The
unprofiled exact-domain pairs and the complete service A/B independently
confirm the timing direction. The earlier maximum-envelope counter capture is
also preserved, but `hardware-serving-shape` is the matched-grid explanation.

## Decision

Keep Q64/KV64 as the new measured GB10 long-context package, retain Q64/KV128
as a legal alternative, and do not change unmeasured portable defaults or
production deployments. This is the best measured package across this bounded
matrix so far, not an exhaustive optimum or a claim to beat both frameworks
on every workload. Short C8 decode remains the clearest remaining gap.

## Reproduction and preservation

The selected benchmark package is
`/home/wlc004s/lunaflux-prefill-choice-20261009.vIZCtp8x/candidate-ready`.
It retains the immutable model/kernel/policy roots and uses the rebuilt worker;
`control-ready` changes only the same worker binding on the previous-best
package. The production deployment remains untouched.

```text
implementation  49336e36
compacted probe 9a0d4445
selected.cu    b987815505fa60062a786b08246aa70ab1bf44fe7c3aae4b77acdb39ff5dc51e
selected.cubin 3c5e993fd06299ab7d1c66a1ad65f5590267628ab2879b889a385adb9d40c8fa
worker         4f9cf22d1c41e8f55a493ddbc483103d9237c6f572af442f9ab4dd1f855df3a1
launch         7981c957e9fa93c5e54729e7c73205437d2500c554e66ee8bdd359e24fa7aba7
decode module  0a0ce21bf82edf8e4c82248f8f0e9ffa1f465d3550c25ac3e653a886948ea051
evidence-v1.tar.gz
59b7922530bdfcb632ef8bec049f514707ecbfcc00581a554f65dd233c2d5fac
```

The archive was downloaded without overwrite to
`/tmp/lunaflux-prefill-evidence-20261009.ciwo1J`. Its hash and all **3,613**
extracted `FILES.sha256` entries verify locally. Source archives/overlays,
selected worker/modules, commands, route measurements, complete request-token
vectors, sanitizers, both counter captures, dispatch traces and failed attempts
are retained. Reproducible model payloads, dependency/build caches and toolchain
directories are excluded. Test containers are stopped and the GPU is idle.
