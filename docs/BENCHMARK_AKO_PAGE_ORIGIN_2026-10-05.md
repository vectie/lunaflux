# Retained page origins — 2026-10-05

## Decision

Reject this experimental production renderer. Six paired cells on both Sparks
are effectively flat: median paired reductions range from −0.37% to +0.08%.
None meets the fixed 3% minimum across every pair. The accepted page-batch
production source and tests are restored exactly; only offline experiment
support and this report remain. No serving bundle was rebound, and these are
kernel measurements, not new end-to-end or framework-comparison results.

One finite candidate, three workloads per host, five alternating pairs per
workload. Both devices were used concurrently, one GPU workload per device.
Compilation finished before unprofiled timing. .179 ran the matched hardware
capture while .178 ran correctness and sanitizer checks.

## Pure plan and terminal lowering

The experiment uses the existing portable `PagedCopyOrigin` relation to prove
the entire page/row/head/component affine extent fits a narrow origin. The
page owner computes page base plus KV-head offset once, and vector consumers
borrow that origin through the existing subgroup shuffle. UINT_MAX is reserved
for absence only when no legal address can equal it. Overflow, wide layouts
and unproved ownership retain the original checked map.

Numerical law, reduction order, copies, barriers, invalid-page publication,
retained V lifetime, partial/mixed paths and public APIs are unchanged. No
model-specific policy, runtime JIT or token-path allocation was introduced.
The experimental source is preserved externally, not as an unselected
production branch or permanent feature flag.

## Unprofiled paired results

Q2048; runtime envelope rows32, grid63×16×1, block128. Each pair has30 GPU-event
repeats. Times below are medians in microseconds; reduction is the median of
individual paired ratios, not the ratio of medians.

| Host | Rows / history | Page-batch µs | Origin µs | Paired reduction | Worst pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| .178 | 1 / 28,672 | 6388.683 | 6411.307 | −0.37% | −1.95% |
| .178 | 2 / 28,672 | 6355.762 | 6359.790 | −0.08% | −1.13% |
| .178 | 2 / 8192 | 1913.887 | 1912.785 | 0.06% | −0.83% |
| .179 | 1 / 28,672 | 6613.088 | 6617.073 | −0.33% | −2.02% |
| .179 | 2 / 28,672 | 6605.955 | 6592.716 | −0.01% | −1.56% |
| .179 | 2 / 8192 | 1979.462 | 1976.992 | 0.08% | −2.48% |

## Matched executed counters

.179 Q2048/R2/H28672, baseline then candidate. These are executed warp
instructions and average warp-latency fractions, not additive elapsed-time
attributions. Instrumented duration is excluded from the timing decision.

| Metric | Page-batch | Origin |
| --- | ---: | ---: |
| Total instructions | 891,530,112 | 879,743,872 |
| `IMAD.WIDE.U32` | 7,342,080 | 14,682,112 |
| `IMAD` | 14,876,672 | 15,794,176 |
| `IADD3` | 12,696,320 | 20,038,400 |
| `LOP3.LUT` | 33,782,784 | 26,442,752 |
| `MOV` | 97,258,496 | 99,128,320 |
| `NOP` | 25,242,624 | 30,852,096 |
| `LDG.E` | 960,128 | 960,128 |
| `SHFL.IDX` | 7,340,032 | 7,340,032 |
| Tensor `HMMA` | 119,668,736 | 119,668,736 |
| Async `LDGSTS` | 14,958,592 | 14,958,592 |
| CTA barriers | 2,808,832 | 2,808,832 |
| Registers/thread / allocated | 234 / 240 | 238 / 240 |
| Register/shared-limited resident blocks | 2 / 2 | 2 / 2 |
| Long-scoreboard / warp latency | 16.21% | 17.71% |
| Barrier / warp latency | 3.06% | 3.22% |
| Issue active | 31.27% | 30.91% |

The source-level simplification does not become uniformly simpler SASS:
instructions fall1.32%, but wide addressing doubles and additions increase.
The hottest load-dependent bounds-check PC remains `ISETP.GE.U32.AND`:
45,641→45,432 not-issued long-scoreboard samples, almost unchanged. The
candidate still waits for the same page producer before issuing dependent
copies. Residency and spilling do not explain the flat timing. Source arithmetic
count alone is insufficient; the next hypothesis must change a proven producer
readiness interval or memory-level overlap, and verify the resulting SASS rather
than merely changing the expression's integer width.

## Validation and preservation

All30 timing pairs are bitwise equivalent and pass the sampled BF16 oracle
ceiling0.003. Q129/R2/H128 memcheck, racecheck and synccheck pass with zero
errors/hazards, oracle maxabs0.000330008. Long Q2048/R2/H28672 memcheck also
passes, oracle maxabs0.000377474. Candidate reports238 registers, two resident
blocks and zero local bytes. Affected native tests pass121/121 experimentally
and120/120 after restoration, using existing warning exclusions20/79/29/25.

GPU178 UUID `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`;
GPU179 UUID `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`. Both GB10/sm121.
GPU job caps16GiB, zero new swap, bounded tasks and600-second deadlines;
paired before/after MemAvailable readings exceed122,230,856KiB against a32GiB
reserve. Both devices are idle at completion. Outer helper unit memory peaks
are not asserted as whole serving-process peaks.

Numerical law remains `approx-base2-f32-v1`, c30322/Q64/K64, stage1.
Baseline cubin `7b42b3531466f5fe22c59e5aac2883db68984907c49a5637ac5f1358ad2eab57`.
Experimental source `99e7a3bbe5e7f852a6a92cc6216f10287946cd7e5ccaf4509346edc09ea11b74`.
Experimental cubin `59be5cca7fa2071ca30214dd37fd371efce2cc7b241f1afe5822a52d741e5abd`.
Independent offline compilations match. CUDA13.0.88 nvcc remains pinned to
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.

Remote roots:

- .178 `/home/wlc003s/lunaflux-ako-page-origin-20261005.bK8FcRJh/experiment-ready`
- .179 `/home/wlc004s/lunaflux-ako-page-origin-20261005.FfAJgTv3/experiment`

Local `/tmp/lunaflux-ako-page-origin-20261005.TSbgJKSn` preserves the source
patch, exact artifacts, raw timings, NCU capture and parsed report. Downloaded
archives and every manifest entry were verified:

- .178 `bce64f7c982412bbff9dcccfc98deb7b1ac3a221b577d047879ea9446742c253`
- .179 `e041f893dc2aa092ac346f125bd97fbe5dc664cc4754c642f16228962d868408`

An initial .178 extraction started before transfer completion; that partial
directory was preserved. A second setup used a nonexistent trial binary path
and failed before GPU execution. After the transfer finished and its hash was
verified, a fresh extraction and exact copied ARM trial binary completed the
experiment. Neither setup failure is counted as a GPU result.
