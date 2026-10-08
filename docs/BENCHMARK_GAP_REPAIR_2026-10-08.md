# One bounded repair iteration after the final-gap investigation

This iteration implements and tests alternatives to the bottlenecks in
[the preceding investigation](BENCHMARK_FINAL_GAP_2026-10-08.md). It does **not**
establish that the remaining framework gap is closed. Rejected schedules are
preserved, not selected merely because their synchronization counters improve.

## Scope and reproducibility

- Spark .179, GB10 `sm_121`, 48 SMs; CUDA UUID
  `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`, PCI `0000000F:01:00.0`.
- Qwen3-0.6B BF16, fixed varied token-ID inputs, greedy/ignore-EOS, 64 output
  tokens. No model-name specialization was added to the compiler.
- Isolated Linux build: `c2fa074d` plus the owned overlay; unrelated dirty-tree
  changes are excluded. Source archive SHA-256
  `ae26abe73d5e449f7fdf9838aae2bbe1f8f847daa558349d9853abde05c6ff55`;
  overlay `63d0b600d88ba5db6f76ce2149c893b0142aa5b96a3193ba2ac4b2eb2deb00ad`.
- Final subset overlay (no completion experiment):
  `39b4ebde8cf41dbdc522092bf7a9bf15853b1538eb8bbd8d668cab2edeb45edc`.
- Experimental overlap worker SHA-256
  `494a1d98c6f886134f271943557a01fb8d9de76ba7ccf199a21ecda8acc369fc`.
- Final source worker (overlap removed) SHA-256
  `15876e2b86b025730bbad22245ae834e21d5694a4e8754214547c0958c222e4d`.
- New production normalization cubin SHA-256
  `c6cfd123edd43d973f704dc24cae48f09a3a69ee3902b8e01acb279391376189`.
- Remote evidence: `/home/wlc004s/lunaflux-gap-repair-20261008.sCCySOLe`.
  All attempts have separate non-overwriting directories. Controllers/probes
  are bounded; serving runs use 64 GiB with no swap and a 32 GiB host reserve.
  GPU workloads are serialized. No production deployment was changed.

## Implemented compiler/runtime changes

### Physical storage is checked after ownership refinement

`ProjectionPhysicalPipeline::static_storage_bytes` accounts for actual padded
operand allocation and lifetime aliasing. A coowned epilogue does not pay for
a nonexistent shared result plane; a split epilogue does. Static plus dynamic
storage is checked for both primary and row-variant plans. The 128×128 coowned
ring at 49,152 bytes is legal with zero dynamic bytes, but not with four more
bytes. This fixes offline candidate admission; it does not make such a
candidate fast or guarantee its compiled register usage is launchable.

Commit: `6d2e2050`.

### Exact-order retained residual normalization

The backend-neutral `RetainedNormalizationTree` redistributes the existing
logical addition tree without reassociation. CUDA lowering retains the original
128 parallel producers and BF16-rounded values, publishes partials once, then
replicates the ordered tree suffix within each consumer subgroup. This removes
seven of eight CTA barriers and the residual global-memory reread. Wider rows
outside the bounded retention budget retain the existing implementation.

Commit: `a2e851c6`. The pure numerical tree/ownership plan remains separate
from CUDA shuffle and shared-memory lowering; no scheduler/model CUDA branch
or request-path compilation is introduced.

Symbolic non-associative tree tests check every addition edge. The production
ABI keeps the same symbol, block geometry and launch contract; module identity
changes. Compiled registers increase from 17 to 32, with no spills. Exact
bitwise comparison covers five patterns and two alias modes. Memcheck with
full leak checking, racecheck, initcheck and synccheck pass.

| Width 1024 / live rows | Old median µs | New median µs | Change |
| --- | ---: | ---: | ---: |
| 1 | 3.52832 | 3.34016 | -5.33% |
| 8 | 3.52736 | 3.35584 | -4.86% |
| 32 | 3.58976 | 3.41888 | -4.76% |
| 2048 | 26.05664 | 25.33568 | -2.77% |

Each has five alternating paired samples with 100 captured launches. The
2048-row new samples include **29.60704 and 30.12672 µs regressions**; the median
must not hide that variability. This is isolated GPU time, not serving speedup.
An earlier one-subgroup producer design was correct but much slower and was
rejected. Only the parallel-producer variant enters the serving comparison.

All diagnostic width/row cells are included below; these use the small
`lf_norm` wrapper, not the production runtime-count ABI timed above. Values
are median duration changes (negative is faster), not whole-engine gains.

| Width / live rows | 1 | 8 | 32 | 2048 |
| --- | ---: | ---: | ---: | ---: |
| 127 | -8.22% | -7.87% | -9.20% | -34.58% |
| 128 | -9.46% | -7.99% | -8.18% | -36.76% |
| 1024 | -5.63% | -5.05% | -5.18% | -0.46% |
| 1025 | -5.15% | -5.60% | -5.65% | -5.58% |
| 2048 | -4.81% | -4.67% | -4.89% | -0.81% |

All 20 cells pass the bitwise check. Five paired samples per cell, including
individual regressions, are in `norm-shapes-summary.json`; tiny large-row
median differences are not evidence of a robust speedup.

### Completion metadata overlap: implemented, tested, then removed

The executor separates private enqueue and completion effects. It writes only
token-independent completion metadata to an exclusive startup-allocated buffer
after enqueue and before waiting. Actual token slots are validated and filled
after successful execution. A pending frame cannot be published; abort clears
it, and stale/foreign owners remain rejected. Failure after enqueue drains and
poisons the executor. The mixed/full-batch allocation gate passes.

This **does not** implement SGLang's one-batch-ahead GPU scheduling. The next
graph still waits for current token/KV ownership retirement. It is a bounded
CPU-work overlap, not a claim that all submission serialization is solved.
The ablation below demonstrated no whole-engine benefit, so this experiment
was removed from production source, including its otherwise-unused APIs and
tests. Its exact code is retained in `overlay.tar`, `source-v2`, and
`completion-overlap-rejected.patch`. No inactive optimization flag was added.

## Alternatives that did not solve their bottleneck

### Decode register exchange: correct, but slower

The experiment replaced shared score/probability exchange with a register and
shuffle realization while preserving the strict F32 law. All five workload
cells passed bitwise comparison and the sampled FP64 oracle. Nevertheless:

| Rows / history | Old µs | Experimental µs |
| --- | ---: | ---: |
| 1 / 127 | 8.229 | 16.407 |
| 8 / 127 | 10.260 | 16.414 |
| 8 / 8192 | 1151.34 | 1249.52 |
| 16 / 4096 | 1152.30 | 1290.62 |
| 1 / 32768 | 1332.12 | 3300.63 |

The C8/8192 counter capture explains why fewer barriers did not help:

| Metric | Selected old | Register exchange |
| --- | ---: | ---: |
| Warp instructions | 41,451,264 | 47,751,296 |
| Registers/thread | 148 | 141 |
| Short-scoreboard / issue-active | 2.03 | 3.40 |
| Long-scoreboard / issue-active | 2.83 | 0.99 |
| Barrier / issue-active | 0.85 | 0.35 |

These stall ratios are not percentages of total completion time. More
instructions and short dependencies outweighed lower barrier/load waits. The
production decode renderer was restored; the attempted source/IR and raw
counter evidence remain under `rejected-register-exchange` and `decode-counter`.

### Wider coowned gate/up: also slower

All comparisons retain the old down projection, stream 28 layers of weights,
and time the complete gate/up/activation/down chain. Three larger geometries
pass bitwise and sampled scalar correctness but regress:

| Gate tile / consumer groups | Old 2048-row chain µs | New chain µs |
| --- | ---: | ---: |
| 64×128 / 8 | 749.25 | 1099.26 |
| 128×64 / 8 | 708.36 | 1231.52 |
| 128×128 / 8 | 706.57 | 978.54 |

Their 128-row chains regress too. The 128×128 / 16-group variant fails launch
with CUDA 701: 150 registers × 512 threads exceeds the SM register budget.
No experimental gate geometry is selected. Larger tiles and lower CTA counts
alone are not a fix; executable pipeline/epilogue costs still matter.

## Validation and integration

- Isolated Linux `moon info`, formatting check, native compile, release worker
  and exporter build pass. The experimental set passes **3,456/3,456 tests**;
  the final subset after removing overlap passes **3,454/3,454 tests**.
- Existing toolchain migration warning exclusions
  `-79-20-29-25-92-14` are retained; this is not a claim of zero legacy warnings.
- Device-worker mixed/full-batch dynamic and static allocation checks pass.
- Four production-normalization sanitizer modes and full leak checking pass.
- The failed AppleDouble-contaminated initial source extraction is preserved;
  the successful build excludes `._*` entries. A private unpacked ripgrep
  package supplies the static checker without altering host installation.
- Changing normalization invalidates the bundle-bound attention table. Fresh
  measurements were generated in `materialize-v2/both/decode-calibration-v2`;
  old timings were not relabeled under a new hash.
- The first serving attempt passed control but rejected a deployment directory
  alias before GPU admission. The repeat uses the original canonical runtime
  path. Both attempts remain available.

## End-to-end ablation

Six fresh starts ran control → runtime → both → both → runtime → control.
Each start discarded one warmup and measured three 8192/64/C8 waves.

| Arm | Median completion ms | Output tok/s | Request TTFT ms | Request TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| Control | 4691.5 | 109.13 | 1392.0 | 48.47 |
| Completion overlap only | 4697.5 | 108.99 | 1398.5 | 48.64 |
| Overlap + retained normalization | 4702.0 | 108.89 | 1402.0 | 48.50 |

This is **no demonstrated end-to-end improvement**. The changes are +0.13% and
+0.22% in completion time, respectively, versus approximately 1.1% between
control-start medians (4670 / 4723 ms). Two starts are insufficient to call
these small differences either a stable win or a stable regression.

All 144 measured requests have identical input bodies across arms and complete
64-token timing/output vectors. Output sequence identity is **not** universal:
control itself matches its first-trial token vectors in 42/48 requests,
runtime in 39/48 and combined in 40/48. The isolated normalization operation is
bitwise-equal, but these varying-schedule serving runs do not establish exact
sequence identity or model-quality parity. Raw vectors are retained.

## Did the changes actually execute?

The profiled combined arm records the new residual symbol with **32 registers**
per thread, matching the new cubin, not the old 17-register kernel. Both traces
contain 96 steps with no unmatched kernels or unresolved categories.

| Single profiled wave | Control ms | Combined ms |
| --- | ---: | ---: |
| Client window | 4718.287 | 4696.618 |
| GPU activity union | 4588.834 | 4577.567 |
| GPU-inactive interval | 129.453 | 119.051 |
| Between step kernel envelopes | 107.821 | 96.539 |
| Multi-query normalization | 88.172 | 87.431 |
| Single-query normalization | 23.797 | 22.841 |

The normalization difference is only **1.697 ms / about 0.036% of the request
wave**. Submission still overlaps the preceding graph in **zero** transitions.
The single-capture inactive-time improvement does not establish an ordinary
timing win; it was not reproduced by the repeated ablation above. One initial
combined profiling attempt hit a reused transient bridge-unit name; it is
preserved separately from the successful uniquely named retry.

## Final selected-source framework comparison

The final worker is rebuilt from the two committed source fixes above;
completion overlap is absent. The serving bundle selects the new normalization
cubin and retains the prior opt-in reference mixed-attention owner and vendor
output/down companions. These are **opt-in benchmark runtime rates**, not a
claim that a default deployment or every compiler-generated path has them.

Order is LunaFlux → vLLM → SGLang → SGLang → vLLM → LunaFlux. Each fresh start
discards one warmup per cell and measures three waves; thus each median has
six waves from **two starts**, not six independent starts.

| Input / output / concurrency | LunaFlux tok/s | vLLM tok/s | SGLang tok/s | LunaFlux completion vs vLLM / SGLang |
| --- | ---: | ---: | ---: | ---: |
| 512 / 64 / 8 | 718.60 | 776.35 | 759.64 | +8.04% / +5.71% |
| 8192 / 64 / 8 | 109.11 | 113.88 | 112.50 | +4.37% / +3.11% |
| 32512 / 64 / 1 | 16.35 | 16.22 | 17.36 | -0.82% / +6.14% |

Throughput counts generated tokens only. The 32K result is C1, not a scaled-up
C8 claim. The 0.82% vLLM advantage is smaller than LunaFlux's approximately
0.92% start-to-start median drift in that cell, so this is not a stable-win
claim. Short-context vLLM start medians drift 651 → 667 ms; the complete starts
and token-delay vectors remain in `frameworks-summary.json`.

| Input / concurrency / engine | Completion ms | Request TTFT ms | Request TPOT ms |
| --- | ---: | ---: | ---: |
| 512 / 8 / LunaFlux | 712.5 | 84.5 | 9.67 |
| 512 / 8 / vLLM | 659.5 | 78.5 | 8.79 |
| 512 / 8 / SGLang | 674.0 | 80.0 | 9.04 |
| 8192 / 8 / LunaFlux | 4692.5 | 1405.0 | 48.56 |
| 8192 / 8 / vLLM | 4496.0 | 1376.5 | 45.77 |
| 8192 / 8 / SGLang | 4551.0 | 1168.0 | 53.21 |
| 32512 / 1 / LunaFlux | 3914.0 | 2448.0 | 22.93 |
| 32512 / 1 / vLLM | 3946.5 | 2441.0 | 23.62 |
| 32512 / 1 / SGLang | 3687.5 | 2128.5 | 24.40 |

All 306 measured requests pass identical input-vector and sampling-parameter
checks. Each has 64 output token IDs and 64 timestamps. Cross-engine and even
some same-engine output sequences differ; no quality-parity claim is made.
The minimum sampled host MemAvailable is 68,901,840 KiB (65.71 GiB), above the
32 GiB reserve. The capped benchmark controllers report zero swap; serving
containers/services have 64 GiB limits with no swap. The bridge units' exit
143 records correspond to the harness's explicit post-campaign stop, not a
failed generation. Runtime stderr is empty and normal drain is recorded.

The final 8K result remains effectively flat against the preceding control
ablation. This iteration does **not** demonstrate closure of the remaining
gap. It must not be described as “all fixed.”

## What remains, and what the failed experiments rule out

1. **Decode attention remains the largest vLLM GPU-time target.** The rejected
   register exchange trades barrier/load stalls for more instructions and
   short dependencies. It does not test or implement a tensor-core partial/
   merge alternative under an explicitly validated numerical policy.
2. **Prefill gate/up remains slower.** Wider coowned tiles alone regress or
   exceed compiled register limits. The storage-accounting bug is fixed, but
   an efficient GEMM plus separately costed epilogue/activation alternative
   is still needed; a larger legal tile is not that implementation.
3. **Runtime batch overlap remains incomplete.** Preparing a few completion
   bytes while the same batch runs is not SGLang's one-batch-ahead execution.
   It was removed after the non-win. Bounded in-flight batch state with
   device-resident token dependencies and ordered cancellation/KV retirement
   remains architectural work.
4. **Normalization has a real but small kernel improvement.** Its selected
   trace saves only 1.697 ms. This cannot close a roughly 200 ms serving gap;
   the remaining QKV and mixed-attention costs also remain targets.

The AKO bounded experiment loop therefore stops after preserving the non-wins,
not by quietly selecting slower schedules or declaring success from reduced
barrier counters. No additional IR layer is justified by these measurements
alone. The retained changes preserve pure planning → explicit ownership/
storage effects → device lowering.

## Sealed evidence

Archive SHA-256:
`f39b84862680654ef1f561c4cbf889803e5c797193f806f8c66b4425d373f793`.
The archive contains 4,072 manifest-checked files, including rejected attempts,
source/overlay archives, final worker copies, selected cubins, raw counters,
timing vectors and build/sanitizer logs. Expanded source trees, build caches
and repeated model copies are omitted; excluded model hashes are retained.
The downloaded archive and every internal file hash were independently
verified locally. [Verification receipt](/tmp/lunaflux-gap-repair-verified-20261008.yT8toIIe/VERIFIED.txt)
and [full framework summary](/tmp/lunaflux-gap-repair-verified-20261008.yT8toIIe/evidence/frameworks-summary.json)
are available in the new non-overwriting evidence directory. No previous
evidence, failed experiment, or unrelated workspace change was deleted.
