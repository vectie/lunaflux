# Joint diffusion plan

This package owns LunaFlux's architecture-neutral startup contract for one
joint audio/video diffusion pipeline. It is deliberately separate from the
decoder-only `model/plan` graph and from token scheduling and KV ownership.

A plan fixes:

- supported conditioning workflows and canonical stage order;
- weight-bearing component roles;
- independent video/audio scheduler identities inside one denoising call;
- bounded canvas, duration, sigma-step, latent, and output geometry;
- exact AudioVAE latent-trunk and initial decoder dimensions needed to split
  decoder-input projection from the remaining decoder body;
- distinct latent-input and output-head precision plus exact video-patch and
  audio-row layouts;
- conditioning-count ceilings; and
- a deterministic digest bound to verified model content.

The plan is execution planning, not physical execution authority. Device
materialization, kernel admission, worker ownership, and request/result
protocols must bind this immutable contract in their own packages before a
model can run.
