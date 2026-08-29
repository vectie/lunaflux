# FP8 v2 compile-evidence candidate emitter

This executable emits two exact generated LunaFlux CUDA candidates for one
closed target: the v2 QKV simple projection and the v2 compound gated MLP. The
source and recipe are derived from one authenticated synthetic dense-Llama FP8
v2 plan, exact paged profile, exact raw operands, and an externally measured
compiler identity. Callers cannot supply CUDA source.

The companion compile harness compiles both candidates for `sm_89`, `sm_90`,
and `sm_120` with exact CUDA 13.1.115 `nvcc`, then seals source, recipe,
compiler, target, and ELF CUBIN digests. This is compile-only evidence. It opens
no CUDA device, launches no kernel, performs no numerical comparison, grants no
runtime authority, and makes no readiness or physical execution claim.
