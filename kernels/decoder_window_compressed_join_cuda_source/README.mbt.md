# Deterministic window/compressed index join CUDA source

This family-neutral package renders an inert I32 correctness kernel for the
official window-first, compressed-second index boundary. It preserves prefill
causal padding, decode ring order, compressed `-1` padding, and literal
concatenation without sorting or deduplication. Deterministic compressed-index
production, K/V preparation and ownership, attention, compilation, and launch
authority remain outside the package.
