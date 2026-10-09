# GLM hybrid execution plan

This package owns the semantic execution vocabulary that is specific to GLM's
hybrid decoder profiles: DSA index computation and cross-layer reuse, Kimi
Delta Attention recurrent state, dense/MoE layer schedules, and the optional
vision-conditioning frontend.

It deliberately reuses `model/advanced_decoder_execution_plan` for shared MoE,
mHC, and MTP contracts. A `GlmHybridExecutionPlan` is immutable and carries a
canonical model identity, but it grants no weight-file, device, kernel, KV,
recurrent-state allocation, or request-execution authority. Those joins must
fail during startup until a backend provides exact admitted implementations.
