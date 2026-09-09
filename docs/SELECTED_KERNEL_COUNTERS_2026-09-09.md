# Selected-kernel instruction and memory diagnosis

## Result and measurement boundary

No compiler or production kernel was changed for this investigation. The
selected matrix pipeline is running, but its on-chip operand movement and
synchronization are inefficient. Attention additionally executes substantial
local-memory traffic. Adding pipeline stages alone is not an established fix.

This extends [matched serving attribution](MATCHED_SERVING_ATTRIBUTION_2026-09-09.md),
which measured the long-C8 serving gap on source `d3d81ca`. That trace attributed
approximately 91% of the GPU-family time difference to linear operations and
attention. The counters below diagnose five selected LunaFlux kernels; they
are not a new end-to-end benchmark or matched baseline counter comparison.

Nsight Compute injection into the serving process failed before usable kernel
capture (supervised child exit -6; direct attempt exit 6). Those attempts are
excluded. Instead, a standalone CUDA-driver replay loaded the **unchanged,
serving-selected CUBINs**, with matching launch dimensions, register counts,
and shared-memory sizes. Only the diagnostic driver was compiled.

Replay inputs are synthetic finite BF16 tensors: 1,024 query tokens, hidden
size 1,024, 16 query heads, 8 KV heads, head dimension 128, intermediate 3,072;
attention has one request with context 1,528 and 191 pages of 8 tokens.
Outputs passed finite/nonzero checks, not an independent numerical oracle.
This reproduces kernel geometry and layout, not live C8 concurrency, all
production data values, the 504-token tail, or the serving cache history.

Device: RTX 5060 Ti, 36 SMs, UUID
`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI `00000000:17:00.0`.
Nsight Compute 2025.4.1 collected `--set full`, 40 passes per kernel, once with
`--cache-control none` and once with `--cache-control all`. Both used
`--clock-control none`; these are two cache conditions, not a statistical
confidence interval. Profiling ran with sudo; driver access policy and
deployment were not changed.

## Counters

Unless stated otherwise, values below use cache-control all. Tensor active is
`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`, not achieved
FLOP/s. Eligible warps are per scheduler per active cycle.

| Selected operation | Warm / flushed ms | Tensor active | L1/TEX throughput | Eligible warps | Shared-load bank conflicts |
| --- | ---: | ---: | ---: | ---: | ---: |
| QKV | 0.643 / 0.644 | 26.1% | 81.7% | 0.370 | 44.15 million |
| Attention | 1.158 / 1.153 | 14.7% | 59.7% | 0.236 | 45.66 million |
| Output projection | 0.329 / 0.330 | 25.4% | 79.2% | 0.331 | 22.07 million |
| MLP gate/up | 0.654 / 0.665 | 36.9% | 32.3% | 0.317 | 14.16 million |
| MLP down | 0.247 / 0.248 | 53.2% | 68.8% | 0.444 | 11.86 million |

Cache flushing changes measured duration by at most 1.6%. Flushed DRAM reads
are respectively 10.52, 10.49, 8.41, 14.72, and 12.60 MB, versus much smaller
warm reads. This strongly disfavors external DRAM bandwidth as the dominant
constraint **in these replays**. It does not establish the complete serving
working set's DRAM behavior. L1/TEX throughput is a subsystem metric, not a
claim that global bandwidth reaches 82% of peak.

### 1. QKV/output: conflicting operand loads and CTA synchronization

Selected scalar `LD.E` operand loads have **8 actual shared wavefronts per
ideal wavefront**. For example, QKV SASS address `0x7da94f76af20` executes
`LD.E R78, desc[UR8][R38.64+0x1000]`: actual 1,048,576, ideal 131,072,
excess 917,504. Output `0x7da94f76a9b0` has actual 524,288, ideal 65,536.
This is measured shared-memory serialization, not an inference from tile size.

Barrier stalls account for 32.1% and 34.0% of their respective average
warp-latency-per-issued-instruction totals. QKV's largest sampled not-issued
location is the K-loop branch at `0x7da94f76b5b0`: 16,941 barrier samples.
The preceding `BAR.SYNC.DEFER_BLOCKING` explains why the wait is sampled on
the branch; this is not evidence of a branch-prediction bottleneck.
Output has the same pattern at `0x7da94f76b040`, 8,743 barrier samples.

Generated QKV code does prefetch the next K64 tile, but then executes
`cp.async.wait_group 0` and `__syncthreads()` each iteration. WMMA operand
loads use a stride-64 shared layout. Counter/SASS evidence identifies the
serialized loads and repeated block rendezvous as real problems. A future
layout/synchronization change needs a controlled counter and timing ablation;
these percentages do not predict an equivalent wall-time speedup.

### 2. Gate/up: block rendezvous with little latency-hiding capacity

Barrier stalls are **47.0%** of average warp latency; the loop branch at
`0x7da94f781b70` has 27,206 barrier samples. A following category of stalls
waits on memory operands (`WARPSYNC.ALL` at `0x7da94f781b20`, 5,852
long-scoreboard samples).

The launch uses 512 threads, 122 registers/thread and 40 KiB total shared
memory. Nsight reports **one resident block per SM**, limited by both the
register and shared-memory budgets. Achieved occupancy is 31.6%, but only
0.317 warps/scheduler/cycle are eligible to issue. Resident warps are mostly
waiting; nominal occupancy alone would obscure this.

Its matrix loads also show 4 actual shared wavefronts per ideal. The
`LDGSTS.E.BYPASS.128.ZFILL` instruction at `0x7da94f781210` has 9,142,272
actual versus 4,571,136 ideal wavefronts. Its reported N-way value is 16,
but the excess is **2x ideal**, not a justified claim of 16x slowdown.

### 3. Attention: local-memory state plus shared-memory dependencies

Attention executes 2,195,456 local-load sectors and 2,293,760 local-store
sectors: **143.65 MB of 32-byte local-sector traffic**. This is requested
local-memory traffic, not 143.65 MB of DRAM transfers. The other four kernels
have zero local sectors in these captures.

Concrete examples include `STL [R80], R37` at `0x7da94f76c230` and
`LDL R38, [R82]` at `0x7da94f76b650`, each with 67,584 executed warp
instructions and 270,336 theoretical local sectors. The generated online
softmax stores per-query `fold_maximum` and `fold_denominator` arrays indexed
by `state_slot`. Those dynamically indexed states are a source-level suspect;
without a source-line mapping or isolated ablation, do not assign every local
instruction to one array. Nsight's derived spill-request metric is zero:
**local-memory allocation is established; register spilling is not**.

Average warp latency divides into long-scoreboard 24.4%, short-scoreboard
20.4%, MIO throttle 13.6%, and barrier 11.6%. The selected matrix loads also
show 8x actual/ideal shared wavefronts, e.g. `LD.E` at `0x7da94f76cd70`
(540,672 actual / 67,584 ideal). Only 0.236 warps/scheduler/cycle are eligible;
issue activity is 18.2% and tensor activity 14.7%.

These jointly identify memory-dependency and operand-movement stalls rather
than saturated Tensor Core arithmetic. Long scoreboard includes global/local
memory dependencies; short scoreboard includes MIO/shared dependencies. The
aggregate counters do not assign all long-scoreboard time to local arrays.
See NVIDIA's [counter and stall definitions](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

### 4. Down projection: a different, less barrier-dominated mix

Down reaches 53.2% tensor activity. Barrier contribution is only 8.2% of
average warp latency, versus fixed-latency wait 27.4%, LG throttle 15.4%, and
math-pipe throttle 14.9%. Its copy instruction at `0x7da94f769a30` has
6,225,920 actual / 3,112,960 ideal shared wavefronts; selected operand loads
have 4x actual/ideal wavefronts. Do not apply the gate/up diagnosis wholesale.

## What this justifies next, and what remains unproven

The measured targets are shared operand layout/lowering, block synchronization
and producer/consumer balance, and attention local-state representation. They
are not a generic shortage of compiler passes. Functional IR transformations
can address them, but only if the selected machine code and counters improve.

No compiler change was made. Next experiments should change one mechanism at
a time, retain numerical checks, repeat these exact counters, and then run
unprofiled serving. A causal speedup attribution still requires those ablations.
The 504-token tail, vocabulary head, decode and baseline selected-kernel NCU
counters remain outside this capture; this report does not allocate every
millisecond of the 2.23x serving gap to individual stall reasons.

## Reproduction and retained artifacts

Remote root: `/run/user/1000/lunaflux-selected-counters-20260909-r1`.
Replay source `replay.cu`; build: CUDA 13.1 `nvcc -std=c++17 -O2 -arch=sm_120
replay.cu -lcuda -o replay`. Invocation accepts the unchanged runtime root and
`1024`. NCU options: `--config-file off --set full --cache-control all
--clock-control none --launch-count 5`; warm capture replaces `all` with `none`.

Selected module SHA-256 identities:

| Module | SHA-256 |
| --- | --- |
| QKV | `e4ab8f87499ae264aa7c1331f0bf4f3c399bf39311bf9e54035eda8935b7d807` |
| Output | `d5ef8dfe464d18736b1395d59cbdb0d7dd8e7de23692954967266ac68f92be48` |
| Gate/up and down | `4b21a8acea0b34af8de8717f1099a6d22e13dadbd79e397045becf8abd1ac4c2` |
| Attention | `3c8f741850cf30b79597b58be3bbbb0f092ac77102db1a9b0eb567ad13088690` |

Both complete `.ncu-rep` files, raw metrics, SASS counters, summaries, profiler
logs and replay source are archived in `selected-counter-results-r1.tar.gz`.
Remote and downloaded archive SHA-256 agree:
`789c200b4edb9ba2ce30264bec481b0620fe7eb46452559d852fed37fa6e5aad`.
Local copy: `/private/tmp/lunaflux-selected-counter-results.qvpSKH/selected-counter-results-r1.tar.gz`.
SASS addresses above are from the cache-flushed report and are module-qualified;
absolute addresses are not stable across runs and can overlap between modules.
