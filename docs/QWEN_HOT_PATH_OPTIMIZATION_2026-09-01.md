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
- Reusable fused-runtime bundle v3 is executable for Qwen. It deliberately
  preserves the shape-aware QKV projection and replaces only
  `QK RMSNorm -> RoPE -> KV write` with one postprocess launch followed by the
  read-only attention launch. Startup expands the three-launch sequence over
  every shape-identical Qwen layer and loads each compiled module once. The
  same bundle can retain the already-working residual/RMSNorm fusion. Worker
  plans keep only typed spans; Llama and Mistral reject this Qwen-only bundle.
- The Qwen postprocess CUDA lowering launches one block per token and projected
  head. Q/K heads perform RMSNorm and positioned RoPE, K/V heads update the
  paged cache, and the projection weights are never read by this kernel.
- Decode split-K paged attention now has a production ordered partial/merge
  executor path. Its workspace is validated and allocated once at startup;
  token-step dispatch only selects a prebuilt owner. Prefill and mixed phases
  retain their independent baseline paths.
- `luna_sampling_strategy` now owns a bounded, backend-neutral stochastic
  sampling plan and fixed workspace ABI. The CUDA backend lowers temperature,
  explicit top-k, top-p-prefix sampling, deterministic counter RNG, and
  non-finite rejection to an offline AOT candidate. Existing production greedy
  sampling is unchanged.

## Remaining before the whole series is production-complete

1. Compile and package the new Qwen postprocess/read-only/residual bundle for the
   target GPU, then run end-to-end numerical and benchmark qualification. The
   runtime join is complete, but a source-level join is not a performance claim.
2. Propagate the optional decode split-K binding through worker/bootstrap
   assembly and physically tune split count by context bucket. The device-step
   executor and deterministic merge path are already production-bindable, but
   an absent binding still selects the baseline executor.
3. Materialize captured native ordered executors for useful non-maximum graph
   buckets. The current sparse O(1) dispatcher owns one exact maximum-envelope
   executor per prefill/decode/mixed phase and uses eager fallback for other
   buckets.
4. Feed physical projection measurements into offline autotune records. The
   generic selector and CUDA runtime-dispatch adapter exist, but release
   production still uses its deterministic default capability plan until those
   records are supplied at startup.
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
