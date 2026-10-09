///|
fn admission_test_limits() -> JointDiffusionArtifactAdmissionLimits {
  try! JointDiffusionArtifactAdmissionLimits::new(
    maximum_requirements=34,
    maximum_modules=4,
    maximum_entry_points=35,
    maximum_module_bytes=1024L,
    maximum_total_module_bytes=4096L,
    maximum_symbol_bytes=64,
    maximum_operands_per_launch=8,
  )
}

///|
fn admission_test_operands(
  roles : ArrayView[JointDiffusionLaunchOperandRole],
) -> Array[JointDiffusionLaunchOperand] {
  roles.map(role => {
    try! JointDiffusionLaunchOperand::new(role~, byte_count=64L, alignment=16L)
  })
}
