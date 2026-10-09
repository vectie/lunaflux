# Block FP8 UE8M0 projection CUDA source

This family-neutral package emits an inert correctness CUDA source for a
row-major E4M3FN projection with 128-by-128 UE8M0 parameter scales and dynamic
per-row, per-128-input activation scaling. The activation maximum has the
official `1e-4` floor, the scale is rounded upward to E8M0, each block dot is
ordered in F32, block scales are applied before the ordered cross-block sum,
and the output has one BF16-RNE boundary.

The five pointer operands are counts, BF16 activation input, raw E4M3 weight
codes, raw UE8M0 scale codes, and BF16 output. The source grants no compile,
materialization, upload, launch, or physical execution authority.
