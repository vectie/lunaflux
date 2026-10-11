///|
test "inclusive zero grid preserves original CPU F32 rounding and cardinality" {
  let scheduler = @joint_diffusion_plan.JointDiffusionScheduler::new(
    latent=VideoLatent,
    kind=RectifiedFlowEuler,
    shift_numerator=12,
    shift_denominator=1,
  )
  // Actual installed original outputs, including the non-exact linspace point.
  assert_eq(
    Float::from_double(scheduler.sigma_at(InclusiveZeroFloat32, 10, 3)).reinterpret_as_uint(),
    1064682126U,
  )
  assert_eq(scheduler.sigma_at(InclusiveZeroFloat32, 2, 0), 1.0)
  assert_eq(scheduler.sigma_at(InclusiveZeroFloat32, 2, 1), 0.0)
  assert_eq(scheduler.sigma_at(TrainingFloorFloat64, 2, 2), 0.0)
  assert_true(scheduler.sigma_at(TrainingFloorFloat64, 2, 1) > 0.0)
  for (count, index) in [(1, 0), (2, -1), (2, 2)] {
    try scheduler.sigma_at(InclusiveZeroFloat32, count, index) catch {
      @joint_diffusion_plan.JointDiffusionPlanError::InvalidScheduler => ()
      _ => fail("wrong sigma domain failure")
    } noraise {
      _ => fail("invalid sigma index accepted")
    }
  }
}
