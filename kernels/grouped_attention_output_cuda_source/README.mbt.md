# Grouped low-rank attention output CUDA lowering

The immutable precision plan defines adjacent-pair inverse suffix rotary,
group-local BF16 low-rank output and dynamically quantized block-FP8 output.
Output-A's E4M3/UE8M0 parameters are converted individually to BF16 before
multiplication, avoiding an expanded persistent matrix while preserving its
numeric boundary. Output-B reuses the existing block-128 projection lowering.
This ordered-F32 correctness implementation is not a tuned performance kernel.
