# Compact expert compute, 9 October 2026

Feature work resumes with TLS-provider and broad warning-migration work paused.
This implementation connects packed checkpoint storage to actual expert
computation; it is not another admission/evidence abstraction.

The functional path is:

1. The family adapter supplies expert IDs, gate/up/down shapes and SwiGLU clamp.
2. ExpertMlpPrecision immutably plans bank offsets, rank mapping and workspaces.
3. PackedBufferLayout is shared by bounded startup upload and precision lowering.
4. CUDA lowers three stages: gate/up activation, weighted down, F32 combination.
5. The generic integration prepares reusable launches for OrderedKernelExecutor.

Weights stay compact. There is no whole-bank BF16 expansion or token-path JIT.
Projection work is computed once per selected expert, rather than recomputing
the entire gate/up dot inside each down-output loop. Device resources remain
caller-owned and use the existing deterministic-release API.

## Tested

- 20 targeted native tests passed across precision IR, materialization, expert
  source generation and GLM upload planning.
- Native compile of the executable binder passed.
- Actual three-stage CUDA execution passed on Spark .178, GB10 UUID
  GPU-9c3d3cf0-439a-5da2-67e9-20414255879f, CUDA 13.0.88, sm_121.
- NVFP4, MXFP4 and BF16 fixtures use 32 hidden/intermediate dimensions, two local
  experts with global IDs [3,1], and routing including an unowned expert.
- Live-row cases 2, 1 and 0 check numerical output and untouched inactive rows.
  CPU reference comparison uses absolute tolerance 0.002.
- Compute Sanitizer memcheck: zero errors and zero leaked allocations.
  Racecheck: zero hazards. Synccheck: zero errors.
- GPU fixture data is below 32 KiB; no full model was loaded and trial workloads
  were untouched. Raw remote outputs are in /tmp/lunaflux-packed-expert.CTnuyg.

moon info completed with zero errors. Warning-denied checks still encounter
pre-existing toolchain deprecation warnings; their broad migration is not part
of this feature change.

## Remaining real features

This is a correctness compute path, not a performance or full-model claim.
Pairwise F32 reductions and explicit BF16 stage rounding are not bitwise-equivalent
to the old serial-dot implementation. Rank-local output is F32: distributed
execution must reduce it and apply the model's final rounding.

Still required: tensor-core compact expert GEMM, rank transport/reduction,
GLM whole-decoder state/attention integration, DeepSeek dynamic-FP8 activations
and complete compressed-attention execution, and MiniMax whole-request component
residency/execution. No three-model serving success is claimed.
