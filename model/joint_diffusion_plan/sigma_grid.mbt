///|
/// Pure startup scalar grid. No device instructions or mutable runtime state.
/// CPU linspace uses one rounded multiply-add per point; subsequent shift
/// operations are separately rounded F32, matching the original tensor law.
pub fn JointDiffusionScheduler::sigma_at(
  self : Self,
  grid : JointDiffusionSigmaGrid,
  requested_points : Int,
  index : Int,
) -> Double raise JointDiffusionPlanError {
  guard requested_points >= 2 && index >= 0 else { raise InvalidScheduler }
  match grid {
    TrainingFloorFloat64 => {
      guard index <= requested_points else { raise InvalidScheduler }
      if index == requested_points {
        return 0.0
      }
      let fraction = index.to_double() / (requested_points - 1).to_double()
      let base = 1.0 +
        (1.0 / self.training_timestep_count.to_double() - 1.0) * fraction
      let shift = self.shift()
      shift * base / (1.0 + (shift - 1.0) * base)
    }
    InclusiveZeroFloat32 => {
      guard index < requested_points else { raise InvalidScheduler }
      let step : Float = -1.0 / Float::from_int(requested_points - 1)
      // Promote the product for one final F32 rounding, equivalent to FMA
      // for these bounded exact-integer indices without a device ABI call.
      let base : Float = if index < requested_points / 2 {
        Float::from_double(1.0 + step.to_double() * index.to_double())
      } else {
        Float::from_double(
          0.0 - step.to_double() * (requested_points - index - 1).to_double(),
        )
      }
      let shift = Float::from_double(self.shift())
      let numerator = shift * base
      let shifted_minus_one = (shift - 1.0) * base
      let denominator : Float = 1.0 + shifted_minus_one
      (numerator / denominator).to_double()
    }
  }
}
