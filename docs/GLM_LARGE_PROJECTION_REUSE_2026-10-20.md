# Large projection ownership experiment

Scope: selected KDA/projection kernels and both-rank execution graphs, not an
overall benchmark. Keep 100 GiB per Spark and 16 GiB host reserve, no hashing,
and restore the owned GLM service after the campaign. Comparisons stay outside
the compiler.

The preceding K128 campaign reduced instructions but regressed the square
matrices: lower residency, increased L2 traffic and load waits outweighed fewer
barriers. Do not repeat that blanket staging change.

The retained installed vLLM trace selects BF16 WMMA 32x32 output tiles, including
K128 single-stage kernels. Its installed KDA source merges six input projections;
LunaFlux already merges those same independent regions. Missing input fusion is
therefore not the hypothesis. The current native 16x64 tile loads each weight
tile again for each 16-token row group. A 32x32 tile halves that weight repetition
but doubles left-operand repetition. This is a testable tradeoff, not a proven
speedup. Both use four independent 16x16 accumulators and the same ascending K16
fold, with unchanged F32/BF16 publication and the existing single-token route.

Budget: two alternatives only, 32x32x32 and 32x32x64, on the existing eleven-shape
fixture, then selected counters and sanitizers for any winner. Resolve ownership
in immutable matrix IR; lowering owns warp coordinates and shared scratch.
An executable winner must reach launch metadata and fused scratch planning,
then be retraced on both ranks before any serving claim. The retained vLLM trace
is October 19, not a fresh memory-qualified reference run.
