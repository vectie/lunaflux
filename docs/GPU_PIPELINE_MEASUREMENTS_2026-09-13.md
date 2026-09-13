# Pipeline measurements and half-baseline feasibility

Status: diagnostic coverage is **partial**, not a completed T0–T5 campaign.
No production kernel was changed for these measurements.

## Scope and provenance

GPU: RTX 5060 Ti, UUID GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6,
PCI 00000000:17:00.0. CUDA 13.1. New attention measurements use the clean
4896771 runtime built with MoonBit v0.10.12. Selected GEMM counters and saved
service traces use the earlier f9df976 kernel implementation; these are not
new end-to-end baseline runs. See TOOLCHAIN_RETEST_2026-09-13.md for the
same-day old/new LunaFlux runtime comparison and C2 output caveat.

Raw local directories:

- `/private/tmp/lunaflux-pipeline-capacity-20260913-r1`
- `/private/tmp/lunaflux-pipeline-selected-issue-20260913-r1`
- `/private/tmp/lunaflux-attention-details-20260913`
- `/private/tmp/lunaflux-static-trace-20260912-r1`

Attention archive downloaded and local/remote SHA-256 matched:
`48d7557b359536a5db1f7418b26c005653878739db312472b407ce6227bec227`.
Remote archive: `/run/lunaflux-toolchain-4896771-20260913/attention-details-20260913.tar.gz`.
Each case retains command, exit status, stdout, stderr and module identity;
four cases additionally retain NCU reports, raw metrics and SASS samples.
NCU uses cold-cache replay and uncontrolled clocks; ordinary event timings
are separate. Profiled and ordinary durations must not be mixed as exact
end-to-end savings.

## T0: measured device calibration

Seven-trial medians: 256 MiB read 425.15 GB/s; read+write copy 385.98 GB/s;
4096-cubed BF16 GEMM 51.57 TFLOP/s. L2 capacity is 32 MiB. System cuBLAS
is version 130201, not necessarily the baseline's bundled library.
These are empirical reference rates, not mathematical hardware ceilings.

M=1528 shape-matched cuBLAS calibration is 49.48 TFLOP/s for QKV,
48.80 for output, 49.95 for gate projection and 49.23 for down. Gate excludes
activation. Small/large matrix calibration covers M=1/8/128/1528/2048.
Repeated weights can fit L2: DRAM-only roofline reasoning is insufficient.

## T1: selected M=1528 GEMM counters

Times are cold-cache NCU microseconds; tensor percentage is elapsed-cycle
activity. Eligible warps are per active scheduler cycle, not occupancy.

| Family | LF / vLLM / SGLang μs | LF / vLLM tensor % | LF / vLLM eligible warps | LF/vLLM instructions |
| --- | --- | --- | --- | --- |
| QKV | 357.50 / 262.30 / 280.48 | 73.87 / 95.44 | 0.67 / 0.20 | 3.07× |
| Output | 180.10 / 138.75 / 147.20 | 72.58 / 93.31 | 0.57 / 0.18 | 2.77× |
| Gate projection* | 526.24 / 388.96 / 395.58 | 72.46 / 96.58 | 0.56 / 0.20 | 2.44× |
| Down | 261.34 / 199.39 / 216.80 | 73.68 / 94.58 | 0.36 / 0.18 | 2.53× |

*LunaFlux gate includes activation; baseline GEMM does not. Do not present
this row as a fully equivalent operator chain comparison.

LunaFlux has **more** eligible warps and higher instruction issue rate, yet
lower tensor activity. Increasing occupancy alone is not the demonstrated
solution. More issue slots are spent on non-matrix work; PC-level dependence
and opcode attribution must identify which work can actually be removed.

Actual L2 read sectors (32 bytes/sector): QKV 18,706,636 vs vLLM 12,735,354;
output 9,420,964 vs 6,442,237; down 14,135,148 vs 9,661,047. DRAM reads are
similar between engines. Gate is a counterexample to a blanket bandwidth
explanation: LF has fewer L2 read sectors than vLLM but remains slower.
Resource block limits are not measured resident-block counts; stall sample
percentages are not additive wall-clock savings.

## T2: independent query and history sweep

Mean of five event-timed trials, 30 launches/trial, μs. `q` is total query
tokens spread across `rows`, not query tokens per row. Multi-row cases have
the probe's small per-row context jitter. The same compiled prefill module
is invoked directly: this does not prove actual runtime dispatch or baseline
equivalence. Paired runs are bitwise equal with finite outputs and unchanged
KV, but there is **no independent numerical reference** in this probe.

| q / rows | past 0 | past 512 | past 4096 | past 8192 |
| --- | --- | --- | --- | --- |
| 32 / 1 | 22.56 | 47.85 | 231.62 | 447.91 |
| 128 / 1 | 30.74 | 67.94 | 256.07 | 493.43 |
| 1528 / 1 | 561.03 | 869.58 | 3023.12 | — |
| 2048 / 1 | 940.67 | — | 4247.66 | — |
| 1528 / 8 | 161.47 | 481.90 | 2739.55 | — |

Tail cases: q33/past513 = 50.13 μs; q129/past4097 = 403.98 μs.
The q128/past4096 to q129/past4097 increase is 57.8%, but **two inputs
changed**. A fixed-history 128/129 sweep is required before assigning the
increase to query tiling alone. The subsequent 2×2 control below separates
these inputs. Multi-row initial attention does less causal
work than a single long row; its lower time is not an 8-row batching speedup.

Follow-up fixed-history control (five trials, same module):

| Query tokens | History 4096 | History 4097 |
| --- | --- | --- |
| 128 | 255.69 μs | 258.11 μs |
| 129 | 401.10 μs | 403.52 μs |

At fixed history, the extra query token increases time by about **57%**;
the extra history token adds only about 1%. This isolates the discontinuity
to the query-count change, though distinguishing extra tile work, CTA waves
and a different resource/dependency pattern still needs per-case counters.
Raw follow-up: remote `attention-boundary` under the same scratch root;
local `/private/tmp/lunaflux-attention-boundary-20260913.tar.gz`.

| Counter case | NCU μs | Tensor activity | Eligible warps | Instructions |
| --- | --- | --- | --- | --- |
| q32/r1/past4096 | 247.97 | 8.53% | 0.34 | 10,754,824 |
| q128/r1/past4096 | 262.53 | 33.18% | 0.49 | 34,414,976 |
| q1528/r1/past0 | 561.12 | 35.75% | 0.54 | 80,715,064 |
| q1528/r8/past4096 | 2704.86 | 39.23% | 0.50 | 386,407,788 |

Derived local/shared spilling requests report zero in all four cases.
That does not establish zero total local-memory traffic or zero bank conflicts.
The profiler's derived shared-nway aggregate is not interpreted as a
literal per-instruction bank-conflict degree. Source samples are retained
for instruction-level attribution rather than claiming another conflict fix.
Short query/long history clearly underutilizes tensor execution; whether
split-K pays for its merge must be measured, not assumed.

## T4 and the half-time budget

Saved long C8 trace: kernel sum 715.027 ms, busy union 714.881 ms, GPU envelope
753.184 ms, request timestamp window 769.377 ms. Envelope minus busy is
38.303 ms (5.09%). Eliminating every external GPU gap gives at most 1.054×
within that envelope. Existing graph replay and continuous batching are
present; internal dependency stalls are a separate target.

The matched saved vLLM GPU envelope is 564.418 ms. Its half is 282.209 ms.
On this **profiled GPU-envelope** basis, LF must save 470.976 ms, or 62.5%.
Removing all LF attention (261.009 ms) and all external gaps still leaves
453.872 ms—well above that target. Thus attention plus host-bubble work
alone cannot deliver half-time, even under this unrealistic elimination.

The four projection families consume 373.006 ms. An illustrative 25%
reduction saves 93.252 ms; halving attention saves 130.505 ms; eliminating
all external gaps saves 38.303 ms. Combined, the optimistic scenario leaves
491.125 ms, **not** 282.209 ms. These are sensitivity calculations, not
predicted speedups; fusion and overlap savings must not be double-counted.

vLLM's large GEMM performance is already close to this device's measured
calibration. Twice its same-work GEMM speed is unsupported by these data.
Half end-to-end time is not proven impossible, but it requires substantial
work/traffic elimination or a different execution strategy beyond merely
matching baseline kernels. Quantization, prefix reuse and speculation remain
separate workload/quality contracts, not shortcuts to the identical-BF16 goal.

## Remaining measurements, in order

1. Per-case counters for the isolated 128/129 query discontinuity; matched baseline
   initial/continuation attention, independent numerical reference, actual
   mixed and pure decode dispatch.
2. Selected small/tail GEMM counters, complete gate+activation chains, source
   PC instruction/transaction attribution and measured L2/shared traffic.
3. One-factor compiler pipeline experiments only after that attribution.
4. Complete C1–16 batch/bucket/padding/metadata timelines and ragged arrivals.
5. Fresh counterbalanced three-engine raw-duration matrix, independent
   input/output extension, sustained samples and C2 deterministic replay.

The new-toolchain anchor has five repeats per cell, but three paired outputs
out of 620 differ at C2, including variation in the old runtime. Root cause
is unresolved. Do not label the performance retest as full correctness pass.
