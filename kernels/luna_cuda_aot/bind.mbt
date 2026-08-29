///|
/// Joins deterministic compiler output to the exact specialization that names
/// its module. This is the only type that publishes production artifact inputs.
pub fn bind_luna_cuda_aot_specialization(
  lowering : LunaCudaAotLowering,
  artifact : LunaCudaAotArtifact,
  specialization : @luna_specialization.LunaSpecializationRecord,
) -> LunaCudaAotSpecializedArtifact raise LunaCudaAotError {
  if artifact.target != lowering.target ||
    artifact.source_digest != lowering.source_digest ||
    artifact.recipe_digest != lowering.recipe_digest ||
    artifact.entry_point_id != lowering.entry_point_id ||
    artifact.function_symbol != lowering.function_symbol ||
    artifact.operand_byte_count != lowering.operand_byte_count ||
    artifact.dimensions != lowering.dimensions {
    raise Mismatch(Artifact)
  }
  if specialization.target() != lowering.target ||
    specialization.profile_shape() != lowering.shape ||
    specialization.program_digest() != lowering.program_digest ||
    specialization.cuda_aot_digest() != lowering.planning_digest ||
    specialization.compiler_policy() != lowering.compiler ||
    specialization.entry_point_id().as_int() != lowering.entry_point_id.as_int() {
    raise Mismatch(Specialization)
  }
  guard specialization.operation_shape() is ElementwiseShape(width~) else {
    raise Mismatch(Specialization)
  }
  if specialization.operation_kind() != ResidualAdd ||
    Int64::from_int(width) != lowering.width ||
    specialization.quantization_policy() != LunaUnquantizedBf16V1 {
    raise Mismatch(Specialization)
  }
  if specialization.module_digest() != artifact.module_digest {
    raise Mismatch(Artifact)
  }
  {
    artifact,
    specialization_digest: specialization.digest(),
    operation_id: specialization.operation_id(),
  }
}

///|
pub fn LunaCudaAotSpecializedArtifact::specialization_digest(
  self : LunaCudaAotSpecializedArtifact,
) -> @luna_specialization.LunaSpecializationDigest {
  self.specialization_digest
}

///|
pub fn LunaCudaAotSpecializedArtifact::operation_id(
  self : LunaCudaAotSpecializedArtifact,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn LunaCudaAotSpecializedArtifact::module_digest(
  self : LunaCudaAotSpecializedArtifact,
) -> @catalog.AotArtifactDigest {
  self.artifact.module_digest
}

///|
pub fn LunaCudaAotSpecializedArtifact::module_relative_path(
  self : LunaCudaAotSpecializedArtifact,
) -> String {
  self.artifact.module_relative_path()
}

///|
pub fn LunaCudaAotSpecializedArtifact::operand_byte_count(
  self : LunaCudaAotSpecializedArtifact,
) -> Int64 {
  self.artifact.operand_byte_count
}

///|
pub fn LunaCudaAotSpecializedArtifact::launch_dimensions(
  self : LunaCudaAotSpecializedArtifact,
) -> @launch_contract.AotLaunchDimensions {
  self.artifact.dimensions
}

///|
pub fn LunaCudaAotSpecializedArtifact::module_input(
  self : LunaCudaAotSpecializedArtifact,
) -> @artifact.KernelModuleInput {
  @artifact.KernelModuleInput::from_bytes(
    digest=self.artifact.module_digest,
    module_bytes=self.artifact.module_bytes,
  )
}

///|
/// Produces the exact artifact-admission identity and generated CUDA symbol.
/// The family ID remains typed catalog metadata; the module, entry-point ID,
/// and symbol come from specialization-bound compiler output.
pub fn LunaCudaAotSpecializedArtifact::entry_point_input(
  self : LunaCudaAotSpecializedArtifact,
  family_id : @catalog.AotKernelFamilyId,
) -> @artifact.KernelEntryPointInput {
  @artifact.KernelEntryPointInput::new(
    entry_point=@catalog.AotKernelEntryPoint::new(
      family=@catalog.AotKernelFamily::new(
        module_digest=self.artifact.module_digest,
        family_id~,
      ),
      entry_point_id=self.artifact.entry_point_id,
    ),
    function_symbol=self.artifact.function_symbol,
  )
}
