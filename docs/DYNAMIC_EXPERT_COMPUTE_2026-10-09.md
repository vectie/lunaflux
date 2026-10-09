# Dynamic activation expert execution

This feature round prioritizes executable model computation, not TLS or new
deployment contracts.

## Implemented chain

Model adapter → immutable precision/expert IR → physical operand regions →
CUDA terminal lowering → prepared ordered launches.

The five stages are BF16 input quantization, gate/up plus SwiGLU and routing,
product quantization, down, and local FP32 expert combination. Each producer
computes a row/block amax once and encodes both payload and UE8M0 scale in one
CTA. There is no full FP32 activation scratch allocation, token-path JIT,
model-name branch, or checkpoint scan in execution.

DeepSeek supplies 1x128 activation blocks and a 1e-4 amax floor, F32 SiLU/product
arithmetic, and routing before the down-input BF16 cast. This is numerically
different from the old staged-BF16, score-after-down path. Both remain explicit
alternatives in shared IR.

The installed DSpark shard header was read without executing model code.
Its first routed expert has I8-packed w1/w3 [2048,2048], w2 [4096,1024],
and F8_E8M0 scales [2048,128] / [4096,64]. The compact loader now handles
these raw-byte tags without expanding weights to BF16.

## Verification

- Focused native tests: 54/54; ordinary native compile, format and interfaces pass.
- Warning-denied check: blocked by existing unrelated deprecated derived methods.
- Hardware: .178 NVIDIA GB10,
  UUID GPU-9c3d3cf0-439a-5da2-67e9-20414255879f, CUDA 13.0.88, sm_121.
- Fixture: hidden/intermediate 128, maximum two rows, two selected experts,
  noncontiguous local IDs, an unowned expert, zero routing weight, and zero input.
- Six input/live-row cases: exact input quantization bytes/scales; final maximum
  observed absolute error 0 against an independent blockwise CPU oracle.
- Declared comparison tolerance: absolute 0.001 + relative 0.02. This does not
  promise bitwise equality with the upstream tensor-core accumulation schedule.
- GPU memcheck/leak, racecheck and synccheck: zero errors/hazards/leaks.
- Existing NVFP4, MXFP4 and BF16 three-stage probes: passed.
- After tests: no compute process, RAM 4.2 GiB used and 117 GiB available.

Generated sources and logs remain under /tmp/lunaflux-dynamic-expert.3j8Jtq
on .178. The first generated concatenation failed CUDA compile due to a missing
newline; version 2 fixes this at the source generator and is the tested artifact.
No production service was changed.

## Still needed

This is a correctness-first SIMT expert implementation, not high-performance
tensor-core grouped GEMM or an end-to-end benchmark. Shared experts, global
rank reduction, full decoder cache/mHC execution, DSpark iteration, and complete
GLM/DeepSeek/MiniMax serving remain separate real implementation work.
