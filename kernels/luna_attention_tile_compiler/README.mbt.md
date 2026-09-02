# Functional attention tile compiler

This package is the pure compiler core between model-independent attention
problems and backend lowering. Its typed stages are `Request -> Selected ->
Semantic -> Scheduled`; no pass can consume a value from the wrong phase.

Autotune records and portable device capabilities are explicit immutable
inputs. The passes perform no I/O, mutate no global compiler state, and return
deterministic values. CUDA, HIP, Metal, and CPU instruction choices begin only
after `ScheduledAttentionTileCompilation`.
