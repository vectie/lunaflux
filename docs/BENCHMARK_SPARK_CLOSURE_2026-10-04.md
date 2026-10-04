# Selected compiler-path repairs on Spark

This campaign implements the four remaining workstreams, checks their actual
AOT/runtime integration, and measures them against a fresh vLLM/SGLang control.
It does not assume that implementing an alternative makes it profitable.

## Implementation and functional boundaries

| Workstream | Implemented change | Runtime consequence |
| --- | --- | --- |
| Decode compute ownership | Immutable owned-score/blockwise folds; owner counts 2/4/8; distinct grouped-head matrix alternatives | Numeric identity, tile geometry and split companions travel through the bundle into startup binding |
| Independent operand lifetimes | K0/V0/K1/V1 rings, future page lookahead, retained checked page origins, publication subsumption | K can be retired/refilled after QK independently of V; consumers retain the checked origin rather than repeatedly reconstructing it |
| Prefill supporting work | Pair-stream probability consumption, factored addressing, separately named approximate-base2 law | Strict c322 remains selected; approximate arithmetic is explicit and is not relabeled as bitwise-equivalent |
| Ingress/rotary reuse | Split-pair register retention; immutable F32 sin/cos pairs prepared once before all layers | One explicit preparation effect per step, shared by every layer; no repeated per-head/per-layer trigonometry |

The changes retain the existing semantic → strategy → physical/effect → device
lowering architecture. Ownership, lifetime, legality and numeric laws are pure
compiler values. CUDA async instructions, subgroup realization and register
mapping stay in device lowering. Runtime allocation/binding occurs at startup;
GPU submission and cache preparation are explicit effects. There is no request-
path JIT, tuning, filesystem authentication or qualification scan.

Relevant implementations include
[`PagedCopyOrigin`](../compiler/attention_physical_ir/paged_copy_origin.mbt),
[`PagedTileEpochs`](../compiler/attention_physical_ir/paged_tile_epochs.mbt),
[`retained copy ownership`](../compiler/attention_physical_ir/retained_copy_ownership.mbt),
[`step rotary cache lowering`](../kernels/luna_cuda_fused_parallel_aot/source_step_rotary_cache.mbt),
and [`startup split binding`](../engine/device_worker/decode_split_runtime.mbt).

### Propagation bugs repaired

The extended decode descriptor now determines both ordinary and partitioned
entry geometry and symbols. Its selected c468 module exports partial/merge
`v3_ep_3903`/`v3_ep_3904`; startup must not request the legacy v1 symbols.
The numeric descriptor is preserved through assembly and startup binding.

A real serving startup exposed a second, packaging-level bug: the refreshed
source overlay included uncommitted changes and the prior manifest, but omitted
an already-committed change to `engine/device_worker/decode_split_runtime.mbt`.
GDB showed CUDA error 500 at function lookup: the stale binder requested
`lunaflux_paged_attention_bf16_decode_split_partial_v1_ep_3903`, while the
qualified module correctly exported v3. The worker's exit was not an OOM.

`prepare_closure_overlay.mbtx --refresh-committed` now merges every eligible
committed path since the explicit pre-work base with the existing manifest and
explicit additions. A regression covers the omitted binder. A new 166-file
source overlay was uploaded and checked byte-for-byte; the failed 164-file
bundle and startup attempts were left intact. The new archive SHA-256 is
`87805e8af466987727e55aed179e13e180b24f5bfe72635624f37422f90725cf`.
The final 170-file overlay additionally includes retained page-validity ownership;
its SHA-256 is `6eeec569f5f905da75751e35162425302f350b5e46e6db6892553b3f133bf3b8`.
That final source was rebuilt, assembled with fresh route measurements, and
used in the final matched comparison below. Its worker SHA-256 is
`ea8d9982c2381cca35210c72aa83918aa9c4f84d1819111001ad5696caecf597`.

## Isolated kernel results

GB10, sm121, 48 SMs, UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`;
CUDA 13.0.88. Ordinary timings are five alternating samples, not profiler
replay durations. All GPU work was serialized. Kernel diagnostic containers
were limited to 8 GiB without swap; serving to 64 GiB without swap, retaining
at least 32 GiB MemAvailable.

| Exact qualified path / workload | Before µs | After µs | Less time |
| --- | ---: | ---: | ---: |
| Cached ingress consumer, C16/history4096 | 484.079 | 436.334 | 9.86% |
| Strict prefill c322, C16/history4096 | 1796.313 | 1797.206 | −0.05% |
| Ordinary c468 decode, C1/history4096 | 323.629 | 171.342 | 47.06% |
| Ordinary c468 decode, C8/history4096 | 566.797 | 584.482 | −3.12% |
| Ordinary c468 decode, C16/history4096 | 1242.220 | 1159.851 | 6.63% |
| Ordinary c468 decode, C16/history8191 | 2450.212 | 2348.850 | 4.14% |
| Partitioned c468, C1/history4096, before/after origin retention | 47.206 | 40.787 | 13.60% |
| Partitioned c468, C16/history4096, before/after origin retention | 1171.089 | 1170.374 | 0.06% |

Ingress preparation costs approximately 4.25 µs once per step, outside the
consumer timing; multiply neither that cost nor its saving by the head count.
The ingress consumer and strict prefill preserve tested bitwise outputs. The
new owned-score reduction has an explicit different numeric law: its checks
use the independent FP64 oracle, not a false bitwise claim against c454.
Ordinary and partitioned c468 are qualified on the **same composed CUBIN**.

Decode qualifications cover 21 cells across C1/C8/C16, fragmented physical
pages, tails and histories through 8191. Ingress and each tested prefill
alternative cover six cells including mixed rows and ragged query tails.
Memcheck/leak, racecheck and synccheck passed for the selected modules and
reported alternatives. Matrix decode c480–483, explicit-exp2 prefill and
KV128/Q64 prefill were also tested; none justified replacing the selected
schedule. KV128/Q64 was about 54% slower in the long C16 example and used
255 registers/one resident block. Short-history owned8 decode also regresses.
These are reasons for workload selection, not hidden results.
The grouped matrix alternatives pad this workload's two query heads per KV
head to a 16-row MMA tile. They are executable generic alternatives, but their
large inactive-row fraction is a structural cost on this shape; merely changing
SIMT operations into MMA does not guarantee less work.

### Why the strict prefill rewrite is neutral

The old pair loop was already unrolled, force-inlined and spill-free, and NVCC
already factored much of the addressing. The rewrite does not reduce the
mathematical exponential, conversion, MMA, rescale or synchronization counts.
Registers increase from 232 to 235. c322 already overlaps future K with current
PV and future V after reader publication; this was not a wholly missing async
pipeline. It retains two async waits and three CTA publications per epoch.
Its retained score/output state and repeated Q fragment loads remain costly.
Wider KV tiles increase live state; simply widening them or enabling exp2 was
not faster in these probes. This is implementation/resource pressure, not
evidence that adding another IR layer alone would help.

## Selected-kernel counters

These are exact C16/history4096 invocations under cold-cache Nsight Compute
replay. They diagnose supporting work; they are not whole-serving timings or
an additive wall-time budget.

| Metric, ordinary decode c454 → composed c468 | Before | After |
| --- | ---: | ---: |
| Executed warp instructions | 46,240,256 | 41,571,072 |
| Registers/thread | 92 | 144 |
| Long-scoreboard stalls per active issue | 4.85 | 2.50 |
| Barrier stalls per active issue | 0.99 | 0.41 |
| Short-scoreboard stalls per active issue | 2.02 | 3.90 |
| Source-correlated excessive shared wavefronts | 0 | 0 |
| Local spilling requests | 0 | 0 |
| Profiled duration, µs | 1353.440 | 1276.352 |

The independent ring initially reduced global-load/barrier waiting but added
addressing work and short-scoreboard pressure. Retaining checked page origins
then reduced instructions about 10.1% versus c454. Short-scoreboard pressure
still increases; it is not valid to say every stall is eliminated.

Cached ingress's counter capture reduced warp instructions 62,970,048 →
56,563,904, with 116 registers unchanged. Strict prefill instructions barely
changed: 191,119,808 → 190,887,040; its replay was slower. Neither profiler
duration replaces the alternating ordinary timings above.

### Retained page-validity follow-up

The next measured dependency was an `LDS` of shared stage validity feeding an
`ISETP` predicate. A pure page-epoch ownership proof now permits replicated
subgroup validity registers for aligned, uniformly owned pages. K0/K1 and
V0/V1 validity have distinct lifetimes; V validity survives early K refill.
Unproven/unaligned cases retain the checked shared fallback. This does not
remove required async waits, CTA publications, tail commits or error draining.

Both ordinary and partitioned entries passed all 21 correctness cells and
memcheck/leak, racecheck and synccheck on their exact composed module. Against
the preceding origin-retaining c468, ordinary C1/history4096 improved
175.539 → 171.123 µs; C8 regressed 580.372 → 597.236 µs; C16 was approximately
neutral, 1180.703 → 1171.851 µs. These are workload-dependent changes, not a
blanket speedup.

The paired C16 counter capture reduced short-scoreboard stalls 4.03 → 3.48,
but registers increased 144 → 148 and long-scoreboard stalls 2.50 → 2.77.
Instructions increased slightly, 41,571,072 → 41,646,848. Replay duration was
1222.752 → 1224.288 µs: essentially neutral. Both captures had zero spilling
and zero source-correlated excessive shared wavefronts. Removing the identified
dependency worked; it did not remove the overall instruction/load bottleneck.
The new SASS hot short-dependency samples instead include the `FMNMX` max-fold
chain (2,416 and 1,377 samples at two PCs); the hottest load-dependency PC is
a metadata `SHFL.IDX` (4,096 samples). A sampled instruction is the dependency
consumer, not necessarily the original memory instruction that caused waiting.
The [next dependency experiments](BENCHMARK_DECODE_DEPENDENCY_EXPERIMENTS_2026-10-04.md)
inspect the neighboring SASS and correct the interpretation further: the hot
`FMNMX` follows a shared score load. Removing that exchange moves the waiting
sample but does not improve C16 time. Scoped FMA removes 20.56% of instructions
with neutral C16 time; no new experimental path was enabled in serving.
This is why deleting the previous validity reload does not imply deletion of
the next critical dependency.

## Whole-serving comparison

The final complete-source comparison used three fresh Latin-square rounds,
identical token-ID requests, identical warm-up per cell, C16 and fresh starts
of pinned vLLM/SGLang 26.01 containers. Rates are aggregate **output** tokens
per second, not input-plus-output throughput. Reported times are medians of
the three measured request-set completion times.

| Input/output tokens per request | LunaFlux ms / tok/s | vLLM ms / tok/s | SGLang ms / tok/s | LunaFlux extra completion time, vLLM / SGLang |
| --- | ---: | ---: | ---: | ---: |
| 128/32 | 371 / 1380.05 | 326 / 1570.55 | 330 / 1551.52 | 13.80% / 12.42% |
| 4096/64 | 4635 / 220.93 | 4201 / 243.75 | 4219 / 242.71 | 10.33% / 9.86% |
| 4096/256 | 12819 / 319.53 | 11835 / 346.09 | 12019 / 340.79 | 8.31% / 6.66% |

Against the frozen LunaFlux control, completion time decreased 2.62%, 1.00%
and 1.13%, respectively. Three rounds do not establish statistical significance
for small gains. They do establish that the large isolated C1 decode gain does
not translate into an equally large C16 serving gain. Long workloads' token
outputs agreed across engines and with the control. Minimum MemAvailable was
55,791,408 KiB, above the 32 GiB reserve.
The preceding complete-source bundle measured 371/4649/12791 ms. The final
validity transport is neutral in serving: 371/4635/12819 ms. It is not a large
new improvement masked by the benchmark. Within the final short-prompt rounds,
both LunaFlux and vLLM varied at the 624/382 boundary; SGLang also differed
from LunaFlux on short outputs. Full all-cell token agreement is therefore
**false**, despite agreement on both long cells. Fixed output counts are still
matched; do not present this as universal output reproducibility.

### Actual selection and effect count

An Nsight Systems trace of the preceding complete bundle, before the final
validity transport, used two 4096/256 C16 trials (warm-up plus measurement) and
verified the new cached ingress, strict prefill, ordinary c468 and its composed
v3 partial/merge entries. This trace is instrumented, not another latency sample.

- Cached ingress: 16,128 calls across 28 layers.
- Rotary preparation: 576 calls, exactly 16,128 / 28: once per execution step.
- Ordinary decode: 13,832 C16 calls plus 336 C8 calls.
- Partitioned decode: 1,176 partial and 1,176 merge calls.
- Strict prefill: 1,848 calls across two observed grids.

Ordinary plus partial decode occupied about 17.08 seconds of accumulated GPU
kernel time across both trials, versus roughly 1.14 seconds of prefill. These
totals are not an additive causal completion-time budget against the references.
The measured remaining problem is chiefly the selected decode implementation
on this workload, not a failure to propagate the new modules or rotary reuse.
Strict prefill remains neutral despite its source changes.

### Short-prompt numerical variation

The preceding comparison had one short LunaFlux round producing token 382
instead of 624 for a repeated body. In the final comparison, two rounds did so.
A separate two-trial diagnostic capture of the preceding bundle correlated
requests using complete 32-token response vectors, not an assumed mapping from
request ID to client row. All **1,024** GPU selections agreed with CPU argmax
of the actual produced BF16 logits.

At the divergent position 129, both tokens had BF16 value 22.0 (bits 16816),
so choosing lower token ID 382 was correct tie-breaking. The identical-body
observation on another execution schedule had 624=22.0 and 382=21.875 (bits
16815), one BF16 ULP apart. Differences already existed at position 127 before
token divergence. This demonstrates schedule-correlated numerical variation,
not a stale sampling readback. It does **not** exclude an upstream arithmetic,
memory or race fault, nor establish identical-schedule replay. No epsilon-based
tie override or production arithmetic change was introduced to hide it.

### Scope and remaining limits

The four requested implementation tracks are implemented, integrated and
physically exercised. The throughput gap is **not eliminated**. Grouped matrix
decode, wider KV128 prefill and approximate exp2 were tested and rejected as
slower/neutral rather than installed merely because they exist. Decode still
has substantial supporting address/reduction instructions; prefill retains
large score/output state, repeated fragment loads and required publication
costs. These are executable schedule/resource constraints, not missing IR names.
The current workload matrix is three vectors on one GB10/model; it does not
prove an optimal schedule for every device/model, production readiness or full
numerical reproducibility across batching schedules.

## Software validation

Local native full suite: **4,353/4,353**. `moon info`, formatting check and native
warning-denied check passed with the existing toolchain migration exemptions
`-79-20-29-25-92-14`; this is not an unsuppressed warning-denied claim.
The new overlay omission regression and bench-helper checks passed.
Unrelated dirty worktree changes were not staged into these implementation
commits or uploaded through the scoped overlay.

## Preserved results

The final serving comparison is
`/home/wlc004s/lunaflux-closure-final-current-20261004.ztDeaWpy`;
the preceding comparison is
`/home/wlc004s/lunaflux-closure-current-20261004.JpGYa6y4`.
Final isolated qualification/source is
`/home/wlc004s/lunaflux-closure-validity-20261004.4N0GE4e7`;
its paired counter capture is
`/home/wlc004s/lunaflux-closure-validity-counter-20261004.KgF8YVCx`.
The launch trace and short diagnostic capture are respectively
`/home/wlc004s/lunaflux-closure-trace-20261004.Ute9ki6P` and
`/home/wlc004s/lunaflux-closure-logits-20261004.j5ClwKY5`.

Twenty-five explicit campaign roots, including slower alternatives and failed
startup diagnosis, were archived without replacing their originals. The compact
archive inventories 32,157 readable files, with 186,483 exclusions explicitly
listed (build/toolchain/source trees, large deployment/release copies and any
unreadable privileged output). It is a results snapshot, not a full machine or
release backup. Archive SHA-256:
`8c7c1fc8e66aa1a26cf8ba04aa40e828635ad37b6679492ea884043049020f4c`.

The byte-identical downloaded archive and local analysis are under
[`benchmarks/results/closure-20261004.k7d52exN`](../benchmarks/results/closure-20261004.k7d52exN/).
That ignored results directory also holds `summary.json`,
`control-comparison.json` and `short-logits-analysis/analysis-final.json`.
The complete 32,157-file inventory verified against the original host files.
Local ordinary extracted files also verified. macOS tar consumes the 126
AppleDouble `._` sidecars as metadata rather than ordinary files; those bytes
remain preserved in the byte-verified archive and verified host inventory.
Original failed runs were preserved; no production deployment was performed.
