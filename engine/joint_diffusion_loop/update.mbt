///|
/// Ordered F32 rectified-flow velocity-to-clean-state and eta-zero update.
/// Preserve each intermediate rounding; replacing this with x + delta*v (or
/// fused multiply-add) is not the same floating-point contract.
fn coefficients(
  timestep : Double,
  sigma : Double,
  next_sigma : Double,
) -> FlowCoefficients {
  let time = Float::from_double(timestep)
  let sigma_t : Float = 1.0 - time
  let ratio : Float = Float::from_double(next_sigma) / Float::from_double(sigma)
  let complement : Float = 1.0 - ratio
  { sigma_t, ratio, complement, }
}

///|
/// Prepare these once per step for the three-scalar device operand.
pub(all) struct FlowCoefficients {
  sigma_t : Float
  ratio : Float
  complement : Float
}

///|
pub fn FlowCoefficients::update(
  self : Self,
  state : Float,
  velocity : Float,
) -> Float {
  let scaled : Float = self.sigma_t * velocity
  let denoised : Float = state + scaled
  let retained : Float = self.ratio * state
  let predicted : Float = self.complement * denoised
  retained + predicted
}

///|
pub fn JointStep::video_coefficients(self : Self) -> FlowCoefficients {
  coefficients(self.video_timestep, self.video_sigma, self.video_next_sigma)
}

///|
pub fn JointStep::audio_coefficients(self : Self) -> FlowCoefficients {
  coefficients(self.audio_timestep, self.audio_sigma, self.audio_next_sigma)
}

///|
pub fn JointStep::update_video(
  self : Self,
  state : Float,
  velocity : Float,
) -> Float {
  self.video_coefficients().update(state, velocity)
}

///|
pub fn JointStep::update_audio(
  self : Self,
  state : Float,
  velocity : Float,
) -> Float {
  self.audio_coefficients().update(state, velocity)
}
