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
- Reusable fused-runtime bundle v2 is now executable for Qwen. Startup expands
  one compiled ingress/read-only module pair over every shape-identical
  `QKV -> QK RMSNorm -> RoPE -> Attention` layer, loads each module once, and
  atomically replaces four graph steps with two. The same bundle can retain the
  already-working residual/RMSNorm fusion. Worker plans keep only typed spans;
  Llama and Mistral reject this Qwen-only bundle.
- The Qwen ingress CUDA lowering now launches one block per token and projected
  head instead of serializing all Q/K/V heads in one block.
- `luna_sampling_strategy` now owns a bounded, backend-neutral stochastic
  sampling plan and fixed workspace ABI. The CUDA backend lowers temperature,
  explicit top-k, top-p-prefix sampling, deterministic counter RNG, and
  non-finite rejection to an offline AOT candidate. Existing production greedy
  sampling is unchanged.

## Remaining before the whole series is production-complete

1. Compile and package the new Qwen ingress/read-only/residual bundle for the
   target GPU, then run end-to-end numerical and benchmark qualification. The
   runtime join is complete, but a source-level join is not a performance claim.
2. Extend the production manifest with bounded decode split-K workspace and two
   ordered launches. Partial online-softmax plus deterministic merge exists as
   a non-bindable CUDA candidate; split count still needs physical tuning by
   context bucket.
3. Materialize captured native ordered executors for useful non-maximum graph
   buckets. The current sparse O(1) dispatcher owns one exact maximum-envelope
   executor per prefill/decode/mixed phase and uses eager fallback for other
   buckets.
4. Feed physical projection measurements into signed/offline autotune records;
   the generic selector exists, but CUDA release production still uses its
   deterministic default capability plan.
5. Add parameter, workspace, and result operands to the sampling manifest and
   executor before binding stochastic v2. It remains `manifest_bindable=false`;
   pure full-vocabulary top-p continues to use the canonical host route.
6. Quantized Qwen routing and speculative draft/verify execution remain later
   work. No benchmark claim is made by this document.

## Architectural boundary

Model builders, the scheduler, and generic strategy packages contain no CUDA,
SM, warp, or WMMA names. CUDA source, subgroup width, matrix tile capability,
and graph-resource ownership remain in backend packages. This keeps LunaFlux
tile-native and backend-pluggable without sacrificing device-specific lowering.
