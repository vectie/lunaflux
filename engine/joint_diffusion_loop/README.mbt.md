# Joint diffusion iteration driver

Preparation also owns immutable boundary permutations: video raw channel-major
latents to patch rows, and stream-major audio rows to VAE `[S,C,T]` storage.
Inverse video mapping is derived rather than separately handwritten. Keep
denoising state packed throughout the loop; these are boundary copies, not
per-step work. `dense_permutation_cuda_source` can lower the generic maps, but
device buffer binding/launch remains external to this driver.

This package owns synchronous joint-denoising progress, not model loading or
device resources. Preparation copies and validates immutable paired step
scalars once. `advance` dispatches one step to borrowed prepared storage and
commits progress only after completion. Cancellation prevents new work; any
backend error permanently fails the loop, including partial latent writes.

The backend must compute both predictions from the old joint state, then apply
both modality updates. It retains latent, prediction and workspace buffers
across calls and completes work before returning. Caller owns deterministic
backend resource release on success, cancellation and failure. This interface
is synchronous and single-owner, not a cross-thread cancellation primitive.

`PreparedPlan::new` consumes a family-neutral joint model plan and resolved
request, prepares both shifted-flow grids, and derives checked video/audio
latent element counts under an explicit combined ceiling. `start` shares those
immutable records with a fresh progress owner; it does not rebuild schedules.
The counts are not a complete device-memory estimate: dtype, prediction
layout, conditioning, transformer workspace and VAE arenas remain separate.

No model-family, CUDA or reference package is imported by production code.
Tests use H3's reference grids and tiny numerical fixtures. They do not execute
H3 weights or prove GPU performance. Full media inference remains incomplete.
