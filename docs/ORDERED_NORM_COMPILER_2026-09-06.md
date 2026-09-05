# Ordered norm communication and rounded-value forwarding

The next bounded optimization preserves the residual/RMSNorm expression and
its existing numerical contract. LunaTile now plans the communication levels
of a descending pairwise reduction from supplied workgroup/subgroup geometry.
It does not replace the tree by a subgroup-first reduction or permit floating
point reassociation. CUDA lowering exchanges the final levels through shuffle
instructions while preserving their operand pairs and addition order.

A separate pure liveness/budget decision retains at most 16 rounded producer
values per lane. The production residual/RMSNorm lowering forwards those BF16-
rounded values through registers to normalization, avoiding a read of the just-
written residual output. Larger rows retain the materialized route. Residual
output remains available for downstream consumers, and no rounded boundary is
removed. The existing six-pointer ABI, 128-thread launch, dynamic-shared-memory
claim, runtime ownership and model/scheduler semantics are unchanged.

This is a reusable compiler scheduling/dataflow building block with its first
consumer in residual/RMSNorm, not a claim that all pointwise families now pass
through a complete semantic compiler pipeline. The backend still owns its
128-thread/32-lane geometry and register budget. No model name enters the plan.

Validation in progress: portable topology/order/budget tests and focused CUDA
source regressions pass. Physical correctness, race/leak checks and current-
source Qwen A/B results must be recorded before a speedup is claimed. The
diagnostic profiler runtime is disposable and never a production artifact.
