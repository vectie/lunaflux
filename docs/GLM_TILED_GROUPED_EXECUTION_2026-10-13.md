# Tiled projections and grouped experts

## Measured target

The retained two-rank GLM trace attributes 40.40% of GPU kernel time to KDA
projections and 30.39% to routed experts. The selected BF16 projection performs
one scalar reduction per output; its R16 control is 22.40x slower than a vendor
GEMM on identical dyadic operands. This control is not a vLLM kernel capture.

The local vLLM implementation batches projection work and groups selected token
rows by expert before tiled matrix multiplication. We will implement those
general execution strategies, not import reference comparisons into the compiler.

## Architecture and finite experiment

1. Precision IR explicitly distinguishes existing scalar accumulation from
   BF16 matrix/F32 accumulation. Matrix reassociation is not a bitwise rewrite.
2. Pure physical IR owns bounded matrix tiles and padded expert grouping.
   Routing IDs, weights, activation rounding, clamp, and expert-ID combine order
   remain semantic inputs; grouping only changes placement.
3. Private CUDA lowering owns MMA, shared storage and synchronization. Packed
   weights are decoded in operand tiles, never into a full dequantized bank.
4. Source-produced launch metadata and startup-owned workspace reach the actual
   executor. No token-step allocation or reference-dependent selection is added.

Budget: two coherent changes (projections and grouped experts), at most two
tuning revisions each. Validate tails, idle rows, repeated/nonlocal experts,
stage numerics, deterministic replay and sanitizer checks before matched model
measurements. Preserve ordered defaults until measured checks justify otherwise.

Probe memory is bounded to 2 GiB, no swap. Full model runs remain at 96 GiB per
Spark with a 16 GiB host reserve and one workload per GPU. No payload hashing.
Record actual selected code, hardware counters, model outputs and failed trials;
neither vLLM agreement nor a finite tensor establishes correctness by itself.

## Executable first implementation

`DenseDotLaw::MatrixFloat32` reaches the KDA projection owner and both recurrent
and sparse-layer MoE owners. Ordinary commands remain scalar. The explicit
`-tiled-grouped` CLI suffix is shared by AOT export and execution; it does not
select the previous non-winning chunk recurrence. `MatrixTilePlan` describes
16x64x32 tiles; the backend uses four BF16 matrix warps. Expert grouping is a
separate bounded launch, gathering selected jobs by expert and scattering
activation/down values back to original slots. Combine order remains ascending
expert ID. Metadata is startup-owned, bounded by jobs rather than expert count;
no full dequantized weights or hot-path heap allocations are introduced.

Initial generated-source probe: `/tmp/lunaflux-glm-tiled-grouped-20261013-v2`.
The v1 harness compile failure is retained. BF16 and NVFP4 fixtures exercise
0/1/7/16/17/18 live rows against capacity 17, noncontiguous local experts,
repeated experts, absent/invalid experts, padding, untouched inactive cells,
stage bitwise comparisons, independent dyadic scalar oracle and deterministic
replay. All pass. Six memcheck/full-leak/racecheck/synccheck campaigns pass with
zero errors/hazards/leaks, 2 GiB caps and zero process swap. This synthetic
arithmetic result does not imply bitwise model equivalence.

Projection [16,4096] x [8192,4096]^T: five-trial median 0.52833 ms, library
control 0.30944 ms. Retained scalar median 6.72473 ms: about 12.7x improvement.
One-row median 0.40309 ms (retained scalar 0.62890 ms), library 0.27974 ms.
Measurements use exact dyadic data and compare BF16 outputs; they are not
vLLM full-model timing. Full-model and counter results follow separately.

The counter-directed revision adds 128-bit dense tile transfers, with scalar
transfers retained for unaligned pointers and K tails. Final pre-model source
probe `/tmp/lunaflux-glm-tiled-grouped-20261013-v4` measures 0.38183 ms at R16
(17.6x below the retained scalar median). The preceding aligned-only v3 was
0.36574 ms; v4 preserves the complete two-byte-aligned operand contract rather
than assuming every view is 16-byte aligned. R17/N65/K33 and K40, one-element
pointer offsets, invalid/idle rows and all expert fixtures pass; all nine
memory/full-leak/race/sync campaigns pass with no errors or leaks.

107 affected native tests and two benchmark-runner tests pass with the existing
migration warning exclusions. The broader packed-execution suite reports nine
old fixture failures (108/117 pass); the exact same failures were reproduced
from untouched `d49c7595` in `/tmp/lunaflux-tiled-baseline-check.UpvIewHx`.
New grouped-owner binding checks pass, including allocation-free replay and
deterministic close. Unexcluded full check/test stop on existing toolchain
migration warnings; no new exclusions or unrelated TLS edits were introduced.
