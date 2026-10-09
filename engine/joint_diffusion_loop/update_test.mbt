///|
test "velocity convention moves toward clean state and retains F32 precision" {
  let step : @joint_diffusion_loop.JointStep = {
    video_timestep: 0.0,
    audio_timestep: 0.0,
    video_sigma: 1.0,
    audio_sigma: 1.0,
    video_next_sigma: 0.0,
    audio_next_sigma: 0.0,
  }
  assert_eq(step.update_video(1.0, 0.001), (1.001 : Float))
  assert_eq(step.update_audio(2.0, 3.0), (5.0 : Float))
  assert_eq(step.video_coefficients().ratio, (0.0 : Float))
  assert_eq(step.audio_coefficients().complement, (1.0 : Float))
}

///|
test "RF update preserves separate multiply add rounding instead of FMA" {
  let step : @joint_diffusion_loop.JointStep = {
    video_timestep: 0.25,
    audio_timestep: 0.25,
    video_sigma: 0.75,
    audio_sigma: 0.75,
    video_next_sigma: 0.0,
    audio_next_sigma: 0.0,
  }
  // Exact product is 8388608.25, rounded to 8388608 before adding state.
  // FMA would yield 0.25 and violate the staged F32 pipeline.
  assert_eq(step.update_video(-8388608.0, 11184811.0), (0.0 : Float))
}
