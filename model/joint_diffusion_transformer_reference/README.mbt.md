# Tiny joint transformer correctness reference

This family-neutral package executes one bounded scalar joint audio/video
transformer step. It projects ordered text, keyframe, or reference
conditioning into the canonical packed layout, applies multi-head joint
self-attention, separate video/audio ReLU MLPs and output projections, splits
latent predictions, and performs one shifted Rectified Flow Euler update.
The update uses H3's clean-state velocity convention: `x0 = x + sigma*v`,
then `next_sigma/sigma*x + (1-next_sigma/sigma)*x0`. This is an independent
Double-precision algebraic reference, not the ordered-F32 production numerical
oracle. Tests check the equivalent positive Euler delta for video and audio
at every step, including terminal zero sigma.

Latents come from the plan-bound counter-based reference RNG. Every shape,
allocation, arithmetic result, and schedule index is bounded or checked for
finiteness. The implementation is an executable correctness oracle for tiny
fixtures only; it is not a production transformer, VAE, device, or kernel
fallback.
