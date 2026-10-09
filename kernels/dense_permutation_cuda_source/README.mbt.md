# Dense permutation CUDA source

Lowers the generic dense permutation plan to a bounds-checked, out-of-place
32-bit-cell copy. This preserves every F32 bit pattern, including signed zero
and NaN payloads. Constant index terms are unrolled; identity becomes a direct
copy. Non-overlap and exact launch/storage binding are caller obligations.

This is source generation, not a physically qualified/tuned kernel. It is an
initial boundary-layout implementation; device profiling may motivate tiled
transpose lowering. H3 should pack video once before denoising, retain packed
state through the loop, and unpack video/audio once for VAE input—not run a
layout conversion at every diffusion step.
