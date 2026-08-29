# Positioned paged KV-write AOT candidate

This package emits a deterministic, qualification-only CUDA source candidate
for writing already-positioned Key and Value activation matrices into the
canonical paged KV layout. It deliberately does not implement attention and
is not a mode of the monolithic paged-attention kernel.

The candidate binds one exact model plan, attention operation, positioned
full-QKV activation, layout, profile, target, compiler policy, raw-pointer ABI, and the exact
source/recipe identity of the existing monolithic write-and-attend fallback.
Candidate, source export, and compiled binding remain non-manifest-bindable and
carry no physical or promotion authority.

Runtime descriptor validation remains the bounds authority. Kernel guards are
defense in depth; a future physical campaign must prove every valid live token
causes its exact Key and Value cache mutations while all non-target sentinels
remain untouched, so an accidental guard return cannot count as success.
