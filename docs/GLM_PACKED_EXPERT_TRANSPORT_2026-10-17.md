# Packed expert operand transport

This experiment separates two physical changes: expert-contiguous quantized
rows with the existing consumer schedule, then the same rows with independent
producer and consumer owners. At most these two schedules are evaluated.
The retained indexed 32-column schedule is the control. Wider columns and
changed numerical folds are not bundled into this experiment.

`grouped_matrix_ir` owns the stable job-to-padded-row materialization and its
bounded payload/scale regions. `physical_tile_ir` owns a finite publication/
retirement ring with disjoint producer and consumer owners. Numerical IR stays
unchanged. CUDA lowering chooses thread roles and synchronization instructions;
the model, routing, and scheduler contain no NVIDIA-specific decisions.

Quantization precedes byte-preserving row/scale packing. Gate/up reads the
packed input; its original-job-order BF16 publication is unchanged. Product
quantization then precedes packing for down. The two packed planes reuse one
startup allocation only after the earlier consumer retires. Global scale is
borrowed from the original quantizer. Invalid/padded rows are zero-filled.
No request-path allocation, host readback, comparison, JIT or hashing is added.

The CUDA experiment uses separate copy and MMA thread groups and two shared
slots. A producer waits for prior readers before overwriting a slot, completes
its asynchronous copies before publication, and proceeds to the other slot.
Consumers wait for publication, preserve ascending K arithmetic, and release
the slot after their last read. Terminal draining closes both slot epochs
before scratch reuse between projections or work tiles. This is not TMA and
must not be described as reproducing CUTLASS's implementation.

Measure the entire grouping/quantization/packing/gate-up/down/combine chain,
not matrix-only timing. Require exact retained outputs and quantized operands,
independent padded-row mapping tests, tails, sanitizer/leak checks, actual
generated-source selection, counters and unprofiled full-model timing. Compare
vLLM outside the compiler; record differing KV/timing policies and output counts.
No favorable microbenchmark alone promotes the default.

Both original GLM containers may be stopped temporarily and must be restored
unchanged afterward. Benchmark owners remain below 100 GiB per Spark, with
16 GiB independently observed host reserve and no task swap. The original
restored service already falls below that reserve; do not confuse restoration
of its unchanged configuration with a compliant benchmark. Preserve NFS and
all unrelated dirty source/toolchain files.

## Implementation and gates

Both transports are explicit, nondefault startup choices, propagated through
GLM recurrent/sparse stage construction, exported symbols, launch metadata,
workspace budgets and frame binding. `LUNA_GLM_EXPERT_TRANSPORT=packed|pipeline`
selects the same policy in the offline preparer and full-model runner. The
ordinary indexed policy is unchanged. The two extra launches materialize
quantized input and product rows; shared experts and arithmetic stay unchanged.

The first named-barrier implementation is rejected: wrapped K-tail racecheck
reported shared-memory hazards despite bitwise outputs. Its earlier timings
are not a validated winner. Phase-counted publication/retirement barriers
replace that handshake. The wrapped 704/576, three-chunk probe v2 passes
memcheck/full leak, racecheck and synccheck at R3/capacity under 2 GiB/no swap.
The final full-width producer-consumer-v3 probe passes all four gates at R1
(memcheck capacity; race/sync trimmed) under 8 GiB/no swap. No check is disabled.
Racechecker CPU workers and outstanding launches are bounded to one; this
reduces checker bookkeeping, not intra-kernel synchronization coverage.
The failed full-width 8 GiB and wrapped 2 GiB checker OOMs remain retained.

Correctness now clears candidate scratch and poisons outputs before execution,
compares retained output/quantized workspaces directly, and independently
checks both packed payload/scale planes against the stable job mapping. No
hashing is involved. CPU validation: 44 compiler/source/frame tests and 27 GLM
integration tests pass. Targeted info has 1084 pre-existing migration warnings,
zero errors; warning-denied check passes with the unchanged migration warning
exclusions. Unqualified check still fails on those existing migration warnings.

Raw roots: `/tmp/lunaflux-glm-fp4-performance-producer-consumer-20261017-v{1,2,3}`,
`/tmp/lunaflux-glm-fp4-performance-pipeline-wrap-20261017-v{1,2}` and
`/tmp/lunaflux-glm-fp4-performance-packed-20261017-v1`.
Full-model and final counter results are pending; no default promotion.

## Corrected pipeline component result

The final v3 source uses phase-counted barriers with explicit invalidation
before shared storage reuse. All eight R1/R3/R22/R64 × capacity/trimmed trials
pass bitwise chain/workspace comparison and the independent packed-row oracle.
At R22/capacity the median complete chain changes 6.796→7.040 ms (+3.58%);
R64/capacity 13.182→13.577 ms (+3.00%). The earlier unsafe named-barrier timing
is explicitly excluded. Packed-only also regresses; its R3 full-bank capacity
memcheck/leak/race/sync gates all pass with bounded checker bookkeeping.

Selected v3 Nsight replay (same synthetic 288 distinct banks and actual
generated native kernels; not a serving trace):

| Metric | Indexed gate/up | Pipeline gate/up | Indexed down | Pipeline down |
|---|---:|---:|---:|---:|
| GPU replay ms | 4.180 | 4.516 | 2.571 | 2.145 |
| Warp instructions, millions | 96.53 | 282.30 | 49.42 | 134.78 |
| L2 read sectors, millions | 36.49 | 58.44 | 18.23 | 29.22 |
| Barrier stall % | 19.23 | 2.63 | 12.09 | 8.96 |
| Long-scoreboard stall % | 76.84 | 59.39 | 60.08 | 70.03 |
| Active-warps occupancy % | 45.64 | 47.48 | 80.86 | 63.75 |
| Registers/thread | 71 | 54 | 48 | 62 |
| Static shared bytes/block | 13,824 | 27,712 | 6,912 | 13,856 |

This is not a win from lower barrier counters. The generated phase waits
execute retry loops and copy owners still issue many `cp.async` instructions;
they are not the installed CUTLASS bulk-TMA producer. Packed padding also
changes zero-fill copies into reads of materialized zero rows. The higher
instruction/read counts and doubled shared storage are measured costs; their
individual contributions to elapsed time have not been independently isolated.
Down's replay benefit is insufficient to win the unprofiled complete chain.
Raw counters: v3 `matrix.ncu-rep` and `matrix-raw.stdout`.

The retained installed FlashInfer CUTLASS R22 component capture is
`/tmp/lunaflux-glm-cutlass-component-r22-20261017-v3`: gate/up 3.247 ms,
down 1.745 ms, 79.03M/64.37M instructions. It uses 128×128×256 tiles, 48
persistent CTAs, one producer and two consumer warp groups. Native uses 16-row
groups, 512 CTA workers and one consumer group. The component shares logical
shapes/banks but has different physical padding. It is not proof of the exact
kernel invocation in vLLM serving, and cannot account for the whole-model gap.

Implementation `e461b17f` is pushed. Its clean commit-pinned ARM native build
passes with 786.5 MiB reported peak and zero swap. The full-model export logs
explicitly select `expert_operand_transport=pipeline`; unprofiled full-model
validation remains below. No experiment has been promoted to the default.

## Full-model native run

Clean pinned runtime/AOT:
`/tmp/lunaflux-glm-expert-tp-pipeline-20261017-v1`.
Matched run:
`/tmp/lunaflux-model-bench-glm-expert-pipeline-20261017-v1`.
Both export and execution select the explicit pipeline mode; recurrent and
sparse routed expert families contain the new packing symbols and phase waits.
Both TP ranks close CUDA successfully. Planned bytes are 99,441,446,188 and
100,710,859,052, below 100 GiB each; cgroup caps remain 100 GiB/no swap.
Observed host minimum is 21.8604 GiB. Startup and one warmup are excluded.

| Prompt input tokens | Output tokens | Retained completion ms | Pipeline completion ms | Pipeline TTFT ms |
|---:|---:|---:|---:|---:|
| 22 | 4 | 577 | 604 | 327 |
| 23 | 3 | 507 | 513 | 320 |
| 22 | 8 | 935 | 932 | 310 |
| 23 | 8 | 949 | 959 | 333 |
| 24 | 8 | 954 | 963 | 337 |
| 36 | 3 | 585 | 601 | 408 |

All six native output ID arrays are unchanged. Completion totals 4.572 s vs
retained 4.507 s (+1.44% in these cohorts); TTFT totals 2.035 s vs 1.992 s.
This single full-model cohort establishes no reliable win. The independent
alternating-order component trials also regress. Leave the indexed default
unchanged rather than introducing a default slowdown. These short C1 prompts
do not establish long-context or concurrent-serving performance or quality.

Full native suite: **5165/5165 pass**, with unchanged migration exclusions
`-20-25-29-35-79-92`. Warning-denied native check and targeted formatting pass.
No unrelated dirty migration/TLS files were committed. The script regression
tests pass (probe 1/1, benchmark runner 3/3). Fresh vLLM results and restoration
are recorded next when complete.
