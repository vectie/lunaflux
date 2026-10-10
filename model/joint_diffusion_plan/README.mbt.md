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

Audio output length is an explicit family-neutral law: `FrameDuration` retains
the nominal video-duration sample count, while `LatentStride(samples)` preserves
all decoded samples from the rounded latent extent. The stride must agree with
the declared sample/latent rates. `audio_samples()` remains nominal duration;
`audio_output_samples()` is the exact per-channel publication length. Result
metadata, output budgets and decoder joins use the latter. This is immutable
startup planning, not a per-step check or a new device branch.

```mbt check
///|
test "explicit rounded audio output retains the complete decoder extent" {
  let shapes = @joint_diffusion_plan.JointDiffusionShapeContract::new(
    frames_per_second=24,
    min_duration_seconds=5,
    max_duration_seconds=15,
    frames_per_chunk=17,
    latents_per_chunk=5,
    canvas_multiple=32,
    max_canvas_pixels=1032192L,
    min_aspect_numerator=1,
    min_aspect_denominator=4,
    max_aspect_numerator=4,
    max_aspect_denominator=1,
    video_latent_channels=24,
    video_spatial_compression=16,
    audio_latent_channels=32,
    audio_latents_per_second=40,
    audio_sample_rate=32000,
    audio_output_channels=2,
    audio_output_length=LatentStride(800),
    audio_vae_latent_dimension=2048,
    audio_vae_decoder_dimension=1024,
    min_inference_steps=2,
    max_inference_steps=1000,
  )
  let shape = shapes.resolve_request(
    height=32,
    width=32,
    requested_frames=120,
    inference_steps=5,
  )
  assert_eq(shape.audio_samples(), 165333L)
  assert_eq(shape.audio_output_samples(), 165600L)
}
```
