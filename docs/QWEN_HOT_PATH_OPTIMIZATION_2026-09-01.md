# Qwen hot-path optimization status (2026-09-01)

This note tracks performance work by whether it reaches the Qwen execution
path. It distinguishes backend-neutral planning, CUDA lowering, compiled
artifact binding, and runtime dispatch. A candidate or binder is not itself a
claim that production dispatch uses the kernel.

## Completed in this series

- Reusable residual/RMSNorm sidecars now cover every same-shape Qwen decoder
  layer. Assembly adds the sidecar before the kernel root becomes read-only.
- `luna_fusion_plan` recognizes the real Qwen sequence
  `QKV -> QK RMSNorm -> positioned RoPE -> paged attention/KV state` without
  hardware vocabulary.
- The CUDA backend lowers that semantic plan to deterministic qualification
  and canary-free production sources. Both ABIs can bind deterministic offline
  compiler output.
- `luna_projection_strategy` owns backend-neutral decode GEMV versus prefill
  matrix-tile selection, row buckets, and deterministic offline autotune
  record selection. CUDA-specific subgroup and matrix-tile capabilities are
  adapted only inside `luna_cuda_projection_aot`.
- `luna_attention_strategy` owns independent prefill and decode plans. The CUDA
  backend emits distinct phase entry points and recipes rather than only one
  mixed source. The current CUDA ABI advertises one KV split; the generic plan
  supports split/merge selection but no recipe claims an unimplemented split.
- `luna_execution_graph_strategy` maps phase, batch rows, query tokens, and
  context length to a power-of-two graph slot. `device_step` selects and seals
  the slot into the staged execution capability before descriptor publication,
  without a runtime bucket scan or heap allocation.

## Remaining before the whole series is production-complete

1. Extend the fused runtime sidecar from residual-only payloads to a bounded
   multi-family bundle and replace each admitted Qwen ingress span with the
   compiled `QKV+QKNorm+RoPE+KV-write` production entry.
2. Give the CUDA decode-attention ABI bounded split-K workspace, implement
   partial online-softmax state and deterministic merge, then physically tune
   split counts by context bucket.
3. Materialize one captured native ordered executor per admitted graph slot and
   dispatch the staged slot to that owner. The current staging selection still
   launches the existing single maximum-shape executor.
4. Feed physical projection measurements into signed/offline autotune records;
   the generic selector exists, but CUDA release production still uses its
   deterministic default capability plan.
5. Run current-source Qwen physical correctness and performance campaigns on
   the target GPU after these runtime joins. No benchmark claim is made by this
   document.

## Architectural boundary

Model builders, the scheduler, and generic strategy packages contain no CUDA,
SM, warp, or WMMA names. CUDA source, subgroup width, matrix tile capability,
and graph-resource ownership remain in backend packages. This keeps LunaFlux
tile-native and backend-pluggable without sacrificing device-specific lowering.
