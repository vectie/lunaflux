# Connected execution selection

Status: implementation and physical experiment in progress.

The prior equal-information ablation did not prove that the continuation
frontier is faster than the old selector. Both chose the same kernel. Keep that
result: a new abstraction does not make a minimum operation smarter.

## Implemented boundary

Compose executable regions before local-cost pruning. Every connection carries
the complete live-boundary contract (layout, numerical law, ownership/effects).
Reject incompatible connections; do not invent a conversion kernel. Observe
the entire serial cover and pass its cost and resource maxima to the existing
frontier. Bind the returned implementation vector atomically, offline. This
adds no IR layer and no request-path tuning or allocation.

Persistent tensor/KV memory remains the memory planner's responsibility. The
serial scratch maximum cannot model concurrent regions or replace that plan.
Unknown local-memory requirements are explicit. The generic selector cannot
admit them against a finite local ceiling. This serving comparison leaves that
dimension unconstrained for both old and new selectors: private/vendor kernel
residency is established by existing artifact/runtime admission, not invented
as zero by the export pass. The known 32 MiB projection workspace is retained.

## Fixed experiment budget

Four existing executable plans: Q64KV128/Q64KV64 attention crossed with generated
or vendor-backed output/down projection. Their materialized BF16 interface is
identical; this experiment does not claim a new fused or nonmaterialized kernel.
Same frozen worker, model, numerical contract, decode, inputs and output limits.

Calibration: two fresh starts per plan in counterbalanced order, with one warmup
and three measured samples for each of 512/64 C8, 8192/64 C8, 32512/64 C1. Freeze
one plan per declared workload from calibration only. Independent validation:
counterbalanced selected/control runs, no selection on validation samples.
At most these four plans, one calibration and one validation pass; no
winner-only reporting. Preserve the current KV64/vendor control.

Compare automatic joint binding with the current manually selected best plan.
Also execute the unmodified old selector with identical complete-plan timings;
it should find the same minimum. This experiment does not measure isolated
producer/consumer costs and therefore does not establish regret for a local
component-cost strategy. Do not attribute an equal-information speedup to new
compiler layers.

Decode policy is held fixed for the exercised C1/C8 buckets. Preserve each
attention artifact's original measurement scope and matched baseline records;
exclude an alternate where today's control deliberately selects baseline.
The unused C16 policies are not made equivalent. The control bundle must match
the current best bundle byte-for-byte before timing. Selection must export the
same bytes as its corresponding manually prepared alternative, including its
digest-checked route table. Held-out runs use the automatically exported bundle.

Spark .179 only, one GPU workload at a time, >=32 GiB available-memory reserve,
64 GiB serving cgroup ceiling, no swap, 900-second service deadline. No deployment.
