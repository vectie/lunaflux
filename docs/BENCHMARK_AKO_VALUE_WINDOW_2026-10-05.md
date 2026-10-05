# Independent PV consumer windows — dual Spark, 2026-10-05

## Decision

Reject the source-level two-column PV consumer-window change. The selected
kernel's entire 78,208-byte executable text is identical to the accepted
baseline. Its launch geometry, registers, shared memory and residency also
match. This is an effective selected-kernel code-generation no-op, not a new
pipeline whose small speedup was hidden by another bottleneck.

All six paired timing cells are flat: median paired reductions range from
−0.59% to +0.14%, below the unchanged acceptance gate of five paired reductions
each at least 3%. Production source, fixtures and generated public interfaces
are restored exactly. No serving bundle was rebound, and no new end-to-end or
vLLM/SGLang comparison is claimed.

Both Sparks ran unprofiled timing cells concurrently. Afterwards .179 collected
selected-kernel counters while .178 ran short sanitizer checks and long-context
memcheck. There was only one GPU job per host; CPU compilation did not overlap
unprofiled timings. Both GPUs were idle at completion and at a subsequent SSH
check. This parallel split reduces turnaround without contaminating measurements
by running competing workloads on one GPU.

## Bounded hypothesis and functional compiler boundary

One candidate family, keeping accepted Q64/K64, D128, one pipeline stage,
page-batch addressing and `approx-base2-f32-v1` numerical law fixed.

The experimental generic physical IR grouped two independent output products
into immutable consumer windows. CUDA terminal lowering staged both right-hand
fragments before interleaving their independent MMA instructions. Each output
retained its instruction and reduction order; no new loads, barriers or
arithmetic law were intended. The proposal was to shorten supporting dependency
chains without increasing CTA geometry or register residency pressure.

This was not a Qwen-name or machine-name branch. The pure plan and its ragged
window/coverage tests were preserved in the external source snapshot, then
removed from production after rejection. A new public abstraction that does not
change executable behavior is not justified by aesthetically different source.

## Executable identity, not just instruction totals

The bounded offline `ako_code_identity.mbtx` parser extracts the unique named
ELF64 little-endian executable PROGBITS section. It compares the selected
symbol's actual bytes, excluding debug names and build paths. It is not part of
runtime loading or the token path. Synthetic tests cover valid extraction,
non-executable sections, missing symbols and out-of-range text.

Symbol: `lunaflux_attention_prefill_tile_compiler_exp2_v1`.

| Identity | Baseline | Candidate |
| --- | --- | --- |
| Executable text bytes | 78,208 | 78,208 |
| Executable text SHA-256 | `62e5de8dd753c52009f7b9385a6b14a2350024fbd422cadc684edbc75719f65c` | same |
| Whole cubin SHA-256 | `7b42b3531466f5fe22c59e5aac2883db68984907c49a5637ac5f1358ad2eab57` | `edd95c4a9bdf64242a6c153b1901f9fd3934cc4de4fa9292a1775499e073df04` |

Different container/source digests do not prove a different executed kernel.
Opcode totals alone would not prove identical instruction order either; the
executable-text comparison supplies that missing check. These observations
establish that the proposed selected schedule was already produced by the CUDA
toolchain. They do not establish that its current schedule is optimal.

## Unprofiled paired results

Q2048, runtime row envelope 32, block128 and grid63×16×1 for both artifacts.
Five alternating pairs per cell, 30 CUDA-event repeats per pair. Reduction is
the median of paired ratios, not a ratio of independently summarized medians.
Negative reduction is slower.

| Host | Rows / history | Baseline µs | Window µs | Paired reduction | Worst pair |
| --- | ---: | ---: | ---: | ---: | ---: |
| .178 | 1 / 28,672 | 6385.210 | 6389.375 | +0.026% | −1.404% |
| .178 | 2 / 28,672 | 6391.883 | 6402.317 | −0.588% | −0.900% |
| .178 | 2 / 8192 | 1912.785 | 1910.138 | +0.138% | −0.064% |
| .179 | 1 / 28,672 | 6624.205 | 6607.306 | −0.196% | −1.835% |
| .179 | 2 / 28,672 | 6576.943 | 6609.615 | −0.497% | −2.509% |
| .179 | 2 / 8192 | 1978.630 | 1977.807 | +0.034% | −0.087% |

The apparent difference between host timings is not itself a framework effect.
Acceptance compares paired artifacts on the same host.

## Matched executed counters

.179 Q2048/R2/H28672, baseline then candidate. Every entry of the executed
opcode-total map matches, not merely the following selected rows.

| Metric | Baseline | Window |
| --- | ---: | ---: |
| Total warp instructions | 891,530,112 | 891,530,112 |
| LDSM.16.MT88.4 | 29,917,184 | 29,917,184 |
| HMMA | 119,668,736 | 119,668,736 |
| MOV | 97,258,496 | 97,258,496 |
| NOP | 25,242,624 | 25,242,624 |
| FMUL | 123,408,384 | 123,408,384 |
| CTA barriers | 2,808,832 | 2,808,832 |
| Registers/thread / allocated | 234 / 240 | 234 / 240 |
| Dynamic shared bytes | 49,168 | 49,168 |
| Register/shared-limited resident CTAs | 2 / 2 | 2 / 2 |
| Active warps, percent of peak | 16.04% | 16.03% |
| Average warp latency per issued instruction | 6.161 | 6.148 |
| Long-scoreboard / warp latency | 11.87% | 11.21% |
| Barrier / warp latency | 3.28% | 3.07% |
| Math-pipe throttle / warp latency | 16.02% | 16.11% |
| Wait / warp latency | 35.51% | 35.57% |
| Issue active | 31.63% | 29.02% |

Instrumented durations, 7.190 versus 7.204 ms, are excluded from the decision.
Warp-latency fractions are not elapsed-time attribution. With identical code,
small counter/timing differences cannot support a claim that the new source
reduced barriers or load dependency. Zero stack/spills and local bytes were
reported; residency did not change.

## Correctness, memory and validation

All 30 timing pairs passed bitwise comparison, maxabs zero against baseline and
the sampled BF16 oracle ceiling 0.003. Short Q129/R2/H128 memcheck, racecheck and
synccheck passed with zero errors/hazards and oracle maxabs 0.000330008. Long
Q2048/R2/H28672 memcheck passed with zero errors and oracle maxabs 0.000377474.
This is kernel correctness, not whole-model quality admission.

Minimum observed MemAvailable was 121,792,852 KiB against a 33,554,432-KiB
reserve. GPU jobs used user-systemd MemoryMax16GiB, no swap, TasksMax64 and a
600-second timeout. CPU jobs were limited to 8GiB, no swap, TasksMax128 and
300 seconds. Unified-memory GPU usage is not reported by nvidia-smi on these
hosts; this report does not invent a GPU-memory number.

Experimental affected tests passed 182/182: physical IR16, source88, lowering7,
attention physical IR32 and tile schedule39. After restoring production, the
same groups passed 181/181 (physical IR15). Offline helpers passed 8/8: schedule1,
report2, trial3 and executable identity2. Scoped warning-denied native checks use
existing exclusions 20/79/29/25; this is not a clean-whole-tree claim. Generated
owned public interfaces are unchanged after restoration.

## Reproduction and preserved artifacts

CUDA13.0.88 nvcc SHA-256:
`fbb111f057786ddd10ba723d993cc7dd43abf978b6baa32fedd3c9d806dc79e1`.
GB10 sm121 GPUs: .178 `GPU-9c3d3cf0-439a-5da2-67e9-20414255879f`,
.179 `GPU-5b603331-c778-d6ef-7ffa-7c6b774fe4a6`.

- .178 root: `/home/wlc003s/lunaflux-ako-value-window-20261005.XE1IcPvW`.
- .179 root: `/home/wlc004s/lunaflux-ako-value-window-20261005.CDxV3xXJ`.
- Local root: `/tmp/lunaflux-ako-value-window-20261005.tZ8QmOWi`.
- Input archive SHA-256: `7fd984b946c7a5807597bd7ff5aac78ca95fe283bad11a1ac8211ed994f11fb5`.
- Candidate CUDA source SHA-256: `1a8e2e66588bc0cb28b6ac32f9763c7559c1fa6c832367edbb8e6caa5298ffd1`.
- .178 downloaded archive SHA-256: `d0592f98f0fcf27e7c748dc3245aedd723ada7f8f944361cbd31e4ad172026e6`.
- .179 downloaded archive SHA-256: `94184f63117c57a4708d1c6c78e3df1af64d25bfd2e6adafcea9abe045750a4a`.

Both downloads used new non-overwriting paths. Archive hashes and every sealed
measurement-manifest entry verify locally. The local root retains both extracted
campaigns, `report.json`, `code-identity.json` and
`final-experimental-source.tar.gz`; no rejected production implementation is
needed in the active tree to reproduce this result.

## Next direction

Before timing another scheduling variant, compare selected executable text,
launch geometry and resource metadata. Skip timing a truly identical executable
configuration; source digests alone are insufficient. This is an offline
developer-loop check, never a request-path validation.

The earlier Q128 experiment genuinely changed instructions but lost independent
CTA overlap. This window experiment did not change selected instructions at all.
The next useful hypothesis must alter the actual fragment ownership/dataflow or
collective dependency plan while retaining independent-CTA overlap, not merely
reorder equivalent source statements. No second candidate is launched in this
bounded round, and no new speedup is claimed.
