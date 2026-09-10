# Why LunaFlux is still about 2x behind: matched execution and hardware counters

## Result

The short-input concurrency gap is primarily **selected GPU kernels**, not
failure to batch. At C8, head, MLP down, and gate/up explain **85.8% of the
steady-decode GPU-time difference from vLLM**. Output projection adds 10.2%.
QKV adds only 1.5%; decode attention does not explain this short-input gap.

This investigation freezes optimizer changes. It does **not** integrate the
uncommitted head-lookahead or bounded-grid metadata candidates, nor claim
that those candidates solve the measured problems.

## Scope and reproducibility

- LunaFlux worker/kernels: `f3d57a2875d878bba34881053dc3e5c7efc93cf9`.
  This retains bounded-row/selected-head work from `3aeeef7` and reverts the
  decode-softmax reassociation experiment. No dirty working-tree code is used.
- Same Qwen3-0.6B BF16 files, token-ID inputs, greedy selection, ignored EOS,
  disabled prefix reuse, same RTX 5060 Ti 16 GB / 36 SM GPU.
- GPU UUID: `GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`.
- vLLM 0.24.0; SGLang 0.5.2 with CUDA graph maximum batch 32, so C16 does
  not unfairly fall back to eager. Their benchmark environments are unchanged.
- Ordinary measurements: four input/output vectors, C1/2/4/8/16, one warmup
  and three measured synchronized bursts per cell. These are finite-burst
  measurements, not steady-arrival saturation tests.
- Fresh NSYS profiles: 59/256 and 1528/32 at all five concurrencies, one
  warmup and one measured burst. Profiler timings are used for attribution,
  not substituted for ordinary throughput.
- LF profiling uses the previously instrumented diagnostic parent to emit
  submitted-batch markers; the committed worker and AOT kernels stay frozen.
- NCU: 13 cold-cache selected-kernel replays, clock control disabled, including
  source/SASS, launch, occupancy, DRAM/L2, issue and scoreboard counters.
  Synthetic deterministic operands are used for these replays, not model
  output-quality validation.

All five profiled LF projection CUBINs were independently hash-joined to the
fresh frozen serving runtime. Baseline replays use the actual installed
cuBLAS libraries, with captured call contracts, and match the serving kernel
symbols, grids, blocks, registers, and software shared-memory allocation.

Results, analysis scripts, summaries and raw archive are under:

`benchmarks/qwen3_comparison/results/gap-diagnosis-f3d57a2-20260910/`

Raw archive `lunaflux-gap-diagnosis-f3d57a2-20260910-r1.tar.gz` was downloaded
without replacing an existing archive; remote and local SHA-256 agree:

`499dea3720315388011d5131555650d1199cab862ad59a30a2a1909634948aab`

Authored orchestration and analysis are MoonBit `.mbtx`; the standalone CUDA
file is a diagnostic library-call replay, not production runtime code.

## Ordinary throughput: current frozen implementation

Output tokens/s, arithmetic mean of three trials. Baselines are the same-day
ordinary runs; the LF column is a fresh run after reverting reassociation.

| Input/output tokens | C | LunaFlux | vLLM | SGLang | vLLM / LF |
| --- | ---: | ---: | ---: | ---: | ---: |
| 59 / 256 | 1 | 248.46 | 277.76 | 268.46 | 1.12x |
| 59 / 256 | 2 | 210.41 | 520.15 | 494.21 | 2.47x |
| 59 / 256 | 4 | 408.62 | 976.17 | 947.12 | 2.39x |
| 59 / 256 | 8 | 789.21 | 1850.05 | 1746.46 | 2.34x |
| 59 / 256 | 16 | 1426.02 | 3229.44 | 3117.26 | 2.26x |
| 1528 / 32 | 1 | 131.87 | 180.46 | 175.52 | 1.37x |
| 1528 / 32 | 8 | 238.29 | 444.70 | 419.90 | 1.87x |
| 1528 / 32 | 16 | 272.83 | 494.05 | 471.02 | 1.81x |

The full 20-cell vectors and trial values are in the three engine summary
JSON files. Token counts are checked; this is not an assertion of identical
generated sequences or independent model-quality qualification.

## 1. Separate GPU work from scheduling and gaps

Short 59/256 C8, fresh profiled burst:

| Quantity | LunaFlux | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| Timestamp window, ms | 2664.986 | 1133.079 | 1169.563 |
| Union of GPU kernel/copy/memset intervals, ms | 2448.206 | 1078.857 | 1082.517 |
| Window minus GPU busy time, ms | 216.780 | 54.222 | 87.046 |

Against vLLM, **1369.349 / 1531.907 = 89.4%** of the observed window difference
is additional GPU busy time. Against SGLang it is 91.3%. The remaining time
includes host execution, gaps, transport and profiler effects; it must not
all be labeled scheduler overhead. LF marker profiling affects host timing
more than the ordinary run, so these shares are diagnostic, not a universal
non-profiled latency decomposition.

LF emits 2048 output tokens and 2040 pure-decode row executions in this
cell. Of those decode rows, **2024 belong to full eight-row steps**
(253 steps); all 57,937 kernels in those steps have a nonzero graph ID.
There is no evidence for a steady-state failure to combine these requests
or a wholesale CUDA graph fallback.

### Attribution avoids a previous analysis trap

Baseline projections are grouped into **complete head-terminated forwards**:
112 ordered decoder projections (28 layers times QKV/output/gate-up/down),
followed by one vocabulary head. No global modulo-113 ordinal is carried
across an incomplete forward. Full-graph classification uses individual
graph launch correlations, not reusable graph IDs alone; steady comparisons
select the dominant complete graph in each cell.

All vLLM measured forward groups are complete. SGLang long-C16 has one
101-projection tail without a head at the client window boundary; it stays
explicitly unattributed. GPU-busy accounting still includes it. This is why
request-output counts cannot simply be equated to backend forward counts.

## 2. The actual per-operation gap

Stable C8 decode, **GPU kernel milliseconds per whole model step**:

| Functional group | LF, short | vLLM, short | SGLang, short | LF / vLLM |
| --- | ---: | ---: | ---: | ---: |
| QKV projection | 0.700 | 0.619 | 0.617 | 1.13x |
| Output projection | 0.889 | 0.343 | 0.350 | 2.59x |
| Gate/up including activation | 2.162 | 0.979 | 0.997 | 2.21x |
| MLP down | 2.254 | 0.558 | 0.556 | 4.04x |
| Vocabulary head | 2.472 | 0.741 | 0.741 | 3.34x |
| Attention | 0.596 | 0.615 | 0.496 | 0.97x |
| Other kernels | 0.420 | 0.265 | 0.520 | — |
| Sum | 9.494 | 4.120 | 4.278 | 2.30x |

LF minus vLLM is 5.374 ms per step: head **1.731 ms / 32.2%**, down
**1.696 ms / 31.6%**, gate/up **1.183 ms / 22.0%**, output **0.546 ms / 10.2%**.
These are measured deltas, not percentages of LF's own runtime. Baseline
gate/up includes its separate activation kernel for a fair functional group.

C16 changes the ranking: LF down is 0.787 ms versus vLLM 0.565 ms, while
LF gate/up rises to 3.311 ms versus 0.995 ms. A C8 diagnosis must therefore
not be generalized to every row bucket. The earlier same-CUBIN gate grid
experiment is relevant to C16, but its uncommitted admission-metadata fix is
not part of this run.

## 3. Selected-kernel counters explain what is slow

Cold-cache replay, C8. LF versus the **actual vLLM-selected library kernel**;
SGLang's head/down/gate counters closely reproduce the latter. Replay times
are not mixed with whole-serving times.

| Kernel | Time, microseconds LF / vLLM | DRAM read, MB LF / vLLM | DRAM throughput % peak LF / vLLM | Warp instructions LF / vLLM |
| --- | ---: | ---: | ---: | ---: |
| Head | 2483.936 / 744.288 | 311.316 / 311.270 | 28.79 / 95.98 | 15,506,968 / 20,444,888 |
| Down | 92.640 / 20.256 | 6.351 / 6.356 | 15.54 / 71.24 | 532,608 / 361,024 |
| Gate/up | 79.392 / 34.464 | 12.614 / 12.613 | 36.03 / 83.05 | 2,519,616 / 588,288 |
| Output | 32.448 / 13.152 | 4.246 / 4.242 | 29.68 / 73.35 | 1,096,640 / 249,408 |
| QKV | 26.304 / 22.656 | 8.421 / 8.419 | 72.70 / 84.38 | 2,514,176 / 392,192 |

NCU gate/up compares the projection body: LF additionally fuses activation,
while the library projects into 6144 concatenated channels. Whole-serving
group accounting above includes baseline activation. SASS shows the extra
LF instruction volume is dominated by addressing, selection, permutation,
shared loads and control flow, not just that activation epilogue.

**All five paired kernels execute exactly the same number of
`HMMA.16816.F32.BF16` warp instructions**:

| QKV | Output | Gate/up | Down | Head |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | 16,384 | 49,152 | 24,576 | 1,215,488 |

This rules out several-fold excess Tensor Core matrix work as the C8
explanation. It does not establish identical floating-point accumulation
order or full numerical equivalence between libraries.

### Head: global-load dependency and insufficient memory-level parallelism

- Actual DRAM read volume is essentially equal. LF does **not** read three
  times the weights from DRAM. Its achieved read rate is about 125 GB/s,
  versus 418 GB/s for the baseline.
- LF long-scoreboard stalls are **98.24%**, issue-active **1.53%**; baseline
  values are 77.78% and 6.77%.
- LF active-warps occupancy is **49.91% versus 10.31%**, and LF executes
  **fewer** total instructions. Neither low occupancy nor more total
  instructions explains this kernel's regression.
- LF SASS uses 4,899,936 scalar `LDG.E` warp instructions; the baseline's
  dominant global load is `LDG.E.LTC128B.128` (1,367,424), with only 18,992
  scalar `LDG.E` instructions. The MMA count is identical.
- The hottest LF sampled PC is the first `HMMA` consuming the newly loaded
  fragments: 242,886 long-scoreboard samples at offset `0x560` from the
  first instruction. The committed inner fold loads one K16 fragment and
  consumes it before progressing; there is no lookahead window in this
  frozen source.
- LF global-load L1 sector requests are 29.210 million versus 14.586 million,
  but DRAM reads remain equal. This is transaction/dependency inefficiency
  **inside the memory hierarchy**, not proof of double DRAM traffic.

The baseline sustains bandwidth with fewer resident warps by supplying work
more effectively. A lower register count or shared-memory elimination is
not, by itself, a performance objective.

### Down: the producer waits on global data before storing it to shared

- Both implementations already use 128-bit global loads. Merely asking for
  vectorized loads would not address the observed path.
- LF long-scoreboard stalls are **79.65%**, versus 55.97%; issue-active is
  **3.34%**, versus 11.07%. DRAM volume is again equal.
- The hottest sampled LF instructions are `STS.128` consuming registers from
  preceding `LDG.E.128`: 5160 and 1348 long-scoreboard samples. A stall on
  this shared-store instruction is waiting for the **global-load result**;
  it must not be mislabeled a shared-memory bank conflict.
- LF launches 32 CTAs x 64 threads; baseline 64 x 32. LF active-warps
  occupancy is actually slightly higher, 4.17% versus 3.69%, yet bandwidth
  is far lower. CTA distribution and the dependency pipeline both need
  consideration; CTA count alone does not prove the entire 4x cause.
- LF has 1.48x the instructions and substantial runtime address/loop work.
  The measurements localize the bottleneck but do not apportion exact speedup
  between a distribution change and a producer-pipeline change without A/Bs.

### Gate/up: expensive staging/control plus weaker work distribution

- Equal matrix instruction count and equal DRAM bytes; **4.28x total warp
  instructions** in LF.
- LF issues 184,704 `PRMT`, 147,456 scalar `LDS`, 178,368 `SEL`, and extensive
  integer/address/branch work. Its hot global-dependency PC is again a
  `STS.128` consuming a preceding load.
- Barrier contribution: **12.30% versus 0.20%**. Grid/block: **96 x 64 versus
  384 x 32**; active warps: **11.09% versus 18.91%**.
- LF's long-scoreboard percentage (57.09%) is lower than the baseline's
  74.35%, but it is slower: extra control/staging changes the denominator.
  Stall percentages must be interpreted with instructions, elapsed time,
  work distribution and achieved bandwidth, not ranked in isolation.

### Output and QKV: the same instruction overhead has different importance

Output has **4.40x** the instructions despite equal MMA count, the same
64 x 32 launch size, similar global sectors and nearly identical DRAM bytes.
Shared scalar loads, permutations and control/address generation remain
expensive. QKV has 6.41x the instructions but only about 1.16x cold-replay
time: its memory throughput is already much closer to the baseline. This
is another reason not to optimize solely by instruction counts.

## 4. Long input: correct the earlier attention priority

Stable long-input C8 decode attention is **3.842 / 3.604 / 3.460 ms** for
LF/vLLM/SGLang. At C16 it is **7.678 / 6.878 / 6.824 ms**. Attention grows
with context and concurrency in **all three** engines. A high percentage
of LF's own time did not establish a similarly large comparative gap.

Long 1528/32 C8, whole measured burst:

| Group, kernel ms | LF | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| QKV | 160.676 | 77.692 | 84.606 |
| Output | 90.810 | 41.315 | 44.642 |
| Gate/up including activation | 189.347 | 124.644 | 132.400 |
| Down | 139.025 | 62.189 | 66.563 |
| Head | 79.626 | 28.241 | 29.751 |
| Attention | 302.888 | 188.663 | 174.491 |
| Other | 73.685 | 31.890 | 62.043 |

LF spends 447.146 ms in pure-prefill plans, 195.918 ms in mixed plans and
392.995 ms in pure-decode plans. There are 12 prefill/mixed plans plus 31
decode plans, versus vLLM's seven piecewise/eager and 31 full-graph forwards.
Those counts are **not** equivalent logical request progress units: padding,
chunk granularity, head suppression and mixed decode work differ.

The whole-window LF-vLLM attention difference is 114.225 ms, much larger
than the approximately 0.238 ms steady-decode per-step attention difference.
Prefill/mixed attention and projection work therefore remain relevant, but
it is incorrect to attribute this to decode attention alone. Exact
prefill-only instruction/byte normalization requires per-forward query and
context metadata; this capture does not contain enough baseline CPU metadata
to assert that decomposition. It remains an explicit unresolved item.

## What this changes about the next optimization

1. Start with **head operand supply**, **down producer pipeline**, and
   **gate/up staging/control and work distribution**. They are measured
   comparative bottlenecks, not generic feature-gap guesses.
2. Lowering must preserve a useful amount of independent memory work ahead
   of consumption, while specializing static address/ownership maps. Do not
   replace this with an occupancy-only or instruction-count-only heuristic.
3. The functional semantic layer can stay pure. The issue is the generated
   dependency graph, memory schedule and launch mapping; a count of compiler
   passes is not a diagnosis. Hardware-specific load/MMA choices stay in the
   CUDA lowering boundary.
4. Do not promise that fixing one counter fixes the total gap. Require an
   isolated mechanism A/B and the same end-to-end vectors after a change.
   No new optimizer is admitted by this report.

The earlier conjecture that a baseline kernel's `K128` name proved a
different accumulation order is withdrawn. The captured calls specify
`CUBLAS_OP_T/N`, BF16 operands/output, `CUBLAS_COMPUTE_32F`, default math,
default tensor-op algorithm, and an 8,519,680-byte workspace. Actual SASS,
not template names, is used for the instruction comparisons above.

No production deployment, soak rerun, new runtime dependency, or performance
claim beyond these measured vectors is made.
