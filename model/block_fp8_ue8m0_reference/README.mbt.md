# Block FP8 UE8M0 scalar reference

This family-neutral bounded oracle defines finite E4M3FN decoding, NVIDIA
UE8M0FNU decoding, upward power-of-two activation-scale selection, finite
E4M3FN rounding, and ordered 128-wide block-dot accumulation. It models values
already crossing a BF16 activation boundary; it is not a checkpoint loader or
an execution implementation.
