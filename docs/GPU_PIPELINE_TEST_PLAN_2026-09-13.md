# GPU pipeline test plan — 2026-09-13

## Objective and scope

Investigate whether LunaFlux can complete an identical request set in at most
half the time of the measured vLLM baseline. This is an experimental target,
not a promised speedup. Record losing cases and unavailable measurements.
Keep production kernels unchanged during the initial diagnosis round.

Starting implementation: `f9df97626b709c1518f3c79fdf97f81b694112dd`;
report commit: `f4e9b516b84d8bc0d2afafa6b5a00c81c187feb2`.
Use the existing compiled runtime for this implementation, not unrelated
uncommitted model-family work. Prior results and experiment identities are in
[SELECTED_KERNEL_GAP_2026-09-12.md](SELECTED_KERNEL_GAP_2026-09-12.md).

Hardware is the RTX 5060 Ti, UUID
`GPU-50c44f23-00cd-8871-b4c7-0c5a62d3e7f6`, PCI `00000000:17:00.0`.
Keep GPU experiments serial, on this device, with no production deployment.
Use Qwen3-0.6B BF16, identical token IDs, greedy decoding, forced requested
output lengths, and no prefix reuse for the main comparison. Installed
baselines are vLLM 0.24.0 and SGLang 0.5.2; record actual executable/library
selection rather than assuming a sibling checkout is the measured version.

## Targets and accounting

The previous unprofiled 1528-input/32-output/C8 run measured 339.82 output
tokens/s for LunaFlux and 441.89 for vLLM. Half-time at fixed work requires
883.79 tokens/s, or about 2.60 times current LunaFlux throughput. TTFT is a
separate target: 304.96 ms currently, 225.83 ms for vLLM, 112.92 ms target.
Use raw trial durations for subsequent time ratios, not the reciprocal of a
mean throughput. Do not conflate finite-burst throughput with saturated load.

The saved latest trace has 715.027 ms summed kernel time, 714.881 ms kernel
busy union, 753.184 ms first-to-last kernel envelope, and 24.701 ms
between-step gaps. The envelope minus busy union is 38.303 ms (5.09%). Removing
all these visible gaps alone offers only 1.054x. This excludes pre-first-kernel
and post-last-kernel host work and does not measure internal pipeline bubbles.

Sampled warp-stall percentages are not additive wall-time fractions. Distinguish
issued/eligible/active warps, memory dependency latency, resource limits,
instruction overhead, useful arithmetic, padding, and GPU timeline gaps.
NCU replay time is diagnostic; CUDA-event time and ordinary serving time are
separate measurements. Never multiply unrelated microbenchmark speedups.

## Ordered tests

### T0 — Calibrated hardware capacity and feasibility budget

Measure streaming read and copy bandwidth with a working set larger than L2;
measure a small cache-resident working set separately. Count copy reads and
writes consistently. Calibrate BF16/F32-accumulate GEMM at large regular shapes
and the actual projection shapes, with deterministic nonzero data and output
checks. Compilation, allocation, initialization and warm-up are outside timing.
Capture clocks, temperature, power and device properties around experiments.

For each selected operation, record useful FLOPs, minimum operand/output
bytes, actual DRAM/L2/shared traffic and measured time. Compute an optimistic
calibrated budget using the maximum of compute and traffic costs, not their
sum when overlap is possible. Measured calibration throughput is not a proven
hardware maximum: a budget derived from it is an engineering estimate, not a
mathematical impossibility proof. Include cache residency and repeated reuse.

Deliverable: capacity observations, per-shape achieved rates, and explicit
missing information before judging the half-vLLM target feasible.

### T1 — Selected GEMM issue and dependency bottlenecks

Replay QKV, output, gate/up and down using the actual current artifacts and
the baseline libraries that ran. Start with M=1528, then small rows and tail
shapes. Pair layout, transpose, dtype, useful work, cache policy and clocks.
Baseline standalone gate/up timings exclude activation; aggregate comparisons
must add it. Do not substitute primary kernels for runtime-selected buckets.

Collect launch dimensions, registers, shared bytes, resident blocks/warps,
eligible warps/cycle, issued warps/cycle, skipped issue slots, Tensor activity,
DRAM/L2/shared throughput and bytes, executed instructions and source-PC
dependency stalls. Separate empty issue slots from cycles with useful work
issued by another warp. Quantify address/control/layout instructions against
identical HMMA work. Missing metrics stay missing, never zero.

Deliverable: one row per engine, exact function, shape and cache mode, with
time, utilization, resource constraint and the hottest dependency chain.

### T2 — Prefill and continuation attention

Distinguish initial empty-cache prefill, continuation with existing KV,
mixed prefill/decode, pure decode and the final small grid. Use the actual
page tables, query lengths, causal positions and launch selection. Sweep
query chunks and past-KV length independently; include page and tile tails.

Measure query/key/score map instructions, causal masked work, QK/softmax/PV
costs, shared-result traffic, spills, warp eligibility, memory dependencies,
and Tensor activity. Compare baseline attention on equivalent semantic work;
different fusion/call boundaries are not automatically equal per-call work.

Deliverable: determine whether remaining cost is mask/index work, shared
transport, softmax arithmetic, inadequate overlap, occupancy or grid tails.
Do not infer steady-state continuation behavior from empty-cache prefill.

### T3 — Controlled pipeline and layout experiments

After T1/T2, isolate one transformation at a time in diagnostic artifacts:
interior/boundary specialization; operand-segment partial evaluation;
accumulator-result ownership; direct versus staged result placement;
copy issue distance; one/two/three/four/six stages where resources permit.
Record prologue, steady-state and drain effects using reduction-length sweeps.
Jointly inspect occupancy, registers, shared capacity and spills.

Existing negative controls must remain visible: simple gate/up lookahead
improved long rows by only about 1% but regressed small rows 14–15%; unconditional
guard removal regressed an eight-row output case. More stages and fewer
barriers are hypotheses, not acceptance criteria. Preserve the numeric fold
unless a separately named tolerance contract is explicitly tested.

Deliverable: paired correctness and timing by shape, identifying a causal win
or a rejected hypothesis. No diagnostic artifact becomes the default here.

### T4 — Cross-operation traffic, batching and graph timeline

Account for complete projection-to-normalization/activation/KV-write chains:
intermediate writes, reloads, launch count, critical path and fusion-induced
resource costs. Existing partial fusion and graph replay are the starting
point, not missing features. Separate independent host work from unavoidable
autoregressive dependencies before proposing overlap.

For C1/2/4/8/16 record actual rows, prefill/decode token counts, padding,
bucket selection, graph replay/fallback, metadata transfers, CPU submission,
GPU issue start, completion and the next step. Include ragged arrivals and
shrinking batches. Attribute first-token delay separately from decode tail.

Deliverable: reconcile family sums, busy union, GPU envelope and request
window; state how much removable external gap actually remains.

### T5 — Ordinary end-to-end confirmation

Keep the existing (59,256), (128,128), (512,64), (1528,32) by C1/2/4/8/16
matrix as the continuity anchor. Then vary input and output independently:
input 32/128/512/1528 and capacity-supported 4096/8192; output 32/128/256.
Screen representative cells before a larger Cartesian campaign. Explicitly
record capacity rejections instead of silently lowering context or concurrency.

Use at least five independent measured repetitions for confirmation, warm-up
outside timing, alternating/counterbalanced engine order, raw durations and
uncertainty. Report TTFT, output-token spacing, total duration, throughput,
errors and exact output lengths. Add sustained/open-loop arrivals and latency
percentiles only with sufficient samples; three trials do not establish p99.
Compare intermediate/logit tolerances and deterministic tokens under the
declared numerical policy, including batch variation and boundary cases.

Quantization, prefix reuse and speculative decoding are separate tracks with
their own quality/workload contracts and equivalent baseline opportunities.
They cannot silently satisfy the identical-BF16/no-reuse target.

## Functional compiler ownership of eventual fixes

Use immutable indexed maps, partial evaluation, common-subexpression sharing,
loop-invariant placement, explicit value ownership and effect/lifetime-aware
pipeline scheduling. Hardware-neutral semantics and scheduling requirements
belong above the CUDA lowering; lane widths and instructions stay below it.
No Qwen-name dispatch, token-path profiling, new runtime JIT, or extra security
work is part of this performance campaign.

## Execution ledger

| Test | Status | Result |
| --- | --- | --- |
| T0 | Measured | Bandwidth and shape GEMM calibration; see GPU_PIPELINE_MEASUREMENTS_2026-09-13.md |
| T1 | Partial | M1528 three-engine issue/eligibility counters; small/tail coverage remains |
| T2 | Partial | 18 query/history timing cases and four NCU cases; baseline-equivalent continuation remains |
| T3 | Planned | Run only after identifying the dependency to isolate |
| T4 | Partial | Long C8 sums/busy/envelope reconciled; full batch/metadata coverage remains |
| T5 | Partial | New-toolchain LF/control anchor complete; fresh three-engine extensions and C2 diagnosis remain |

Every entry records command/input identity, raw result location, conclusion,
and next action. Reuse existing tested harnesses; new orchestration is MoonBit
`.mbtx`. Save results in new directories and download them. Keep raw large
profiles outside Git; commit the plan, reusable diagnostics and concise results.
Run focused diagnostic self-tests and formatting; backend changes, if later
made, additionally receive correctness/sanitizer checks for their affected
ownership boundary. No unrelated 24-hour soak or release campaign is required.
