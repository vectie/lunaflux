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
