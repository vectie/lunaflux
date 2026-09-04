# Functional attention tile compiler

This package is the pure compiler core between model-independent attention
problems and backend lowering. Its typed stages are `Request -> Selected ->
Semantic -> Optimized -> Scheduled`; no pass can consume a value from the
wrong phase. Optimization is a bounded pure pass fold that returns immutable
equational regions and a deterministic trace.

Autotune records and portable device capabilities are explicit immutable
inputs. The passes perform no I/O, mutate no global compiler state, and return
deterministic values. CUDA, HIP, Metal, and CPU instruction choices begin only
after `ScheduledAttentionTileCompilation`.

Storage planning is expressed as pure lifetime analysis. In particular, the
online-softmax rewrite overlaps dead key/probability regions and keeps fold
state local to its owner before any backend selects a concrete address space.
