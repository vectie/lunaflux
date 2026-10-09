///|
fn artifact_test_qualification() -> GlmHybridDeclaredQualificationDigest {
  try! GlmHybridDeclaredQualificationDigest::from_sha256(
    "1111111111111111111111111111111111111111111111111111111111111111",
  )
}

///|
fn artifact_test_requirements(
  fill? : Char = 'a',
) -> @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements {
  try! @glm_hybrid_kernel_requirements.derive_requirements(
    artifact_test_plan(fill~),
    limits=try! @glm_hybrid_kernel_requirements.GlmHybridKernelRequirementLimits::new(
      maximum_layers=8,
      maximum_vision_layers=8,
      maximum_requirements=64,
    ),
  )
}

///|
fn artifact_test_target(minor? : Int = 0) -> @catalog.DeviceTarget {
  try! @catalog.DeviceTarget::new(
    compute_major=9,
    compute_minor=minor,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
}

///|
fn artifact_test_launch() -> @launch_contract.AotLaunchDimensions {
  try! @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=32,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  )
}

///|
fn artifact_test_limits() -> GlmHybridArtifactAdmissionLimits {
  try! GlmHybridArtifactAdmissionLimits::new(
    maximum_claims=64,
    maximum_module_bytes=1024,
    maximum_total_module_bytes=65536L,
    maximum_canonical_bytes=65536,
  )
}
