///|
pub struct LunaCudaAotLimits {
  priv max_source_bytes : Int
  priv max_recipe_bytes : Int
  priv max_module_bytes : Int64
  priv max_evidence_bytes : Int
} derive(Debug, Eq)

///|
pub fn LunaCudaAotLimits::new(
  max_source_bytes~ : Int,
  max_recipe_bytes~ : Int,
  max_module_bytes~ : Int64,
  max_evidence_bytes~ : Int,
) -> LunaCudaAotLimits raise LunaCudaAotError {
  if max_source_bytes <= 0 ||
    max_recipe_bytes <= 0 ||
    max_module_bytes <= 0L ||
    max_evidence_bytes <= 0 {
    raise Invalid(Limits)
  }
  { max_source_bytes, max_recipe_bytes, max_module_bytes, max_evidence_bytes }
}

///|
pub struct LunaCudaAotLowering {
  priv target : @catalog.DeviceTarget
  priv shape : @device_profile.ExactExecutionShape
  priv entry_point_id : @catalog.AotKernelEntryPointId
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv width : Int64
  priv element_count : Int64
  priv operand_byte_count : Int64
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv program_digest : @luna_tile_ir.LunaTileProgramDigest
  priv planning_digest : @luna_tile_ir.LunaTileCudaAotDigest
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : LunaCudaSourceDigest
  priv recipe : Bytes
  priv recipe_digest : LunaCudaRecipeDigest
} derive(Debug, Eq)

///|
pub fn LunaCudaAotLowering::target(
  self : LunaCudaAotLowering,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn LunaCudaAotLowering::execution_shape(
  self : LunaCudaAotLowering,
) -> @device_profile.ExactExecutionShape {
  self.shape
}

///|
pub fn LunaCudaAotLowering::entry_point_id(
  self : LunaCudaAotLowering,
) -> @catalog.AotKernelEntryPointId {
  self.entry_point_id
}

///|
pub fn LunaCudaAotLowering::compiler_policy(
  self : LunaCudaAotLowering,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn LunaCudaAotLowering::width(self : LunaCudaAotLowering) -> Int64 {
  self.width
}

///|
pub fn LunaCudaAotLowering::element_count(self : LunaCudaAotLowering) -> Int64 {
  self.element_count
}

///|
pub fn LunaCudaAotLowering::operand_byte_count(
  self : LunaCudaAotLowering,
) -> Int64 {
  self.operand_byte_count
}

///|
pub fn LunaCudaAotLowering::launch_dimensions(
  self : LunaCudaAotLowering,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn LunaCudaAotLowering::program_digest(
  self : LunaCudaAotLowering,
) -> @luna_tile_ir.LunaTileProgramDigest {
  self.program_digest
}

///|
pub fn LunaCudaAotLowering::planning_digest(
  self : LunaCudaAotLowering,
) -> @luna_tile_ir.LunaTileCudaAotDigest {
  self.planning_digest
}

///|
pub fn LunaCudaAotLowering::function_symbol(
  self : LunaCudaAotLowering,
) -> String {
  self.function_symbol
}

///|
pub fn LunaCudaAotLowering::source_bytes(self : LunaCudaAotLowering) -> Bytes {
  self.source
}

///|
pub fn LunaCudaAotLowering::source_digest(
  self : LunaCudaAotLowering,
) -> LunaCudaSourceDigest {
  self.source_digest
}

///|
pub fn LunaCudaAotLowering::recipe_bytes(self : LunaCudaAotLowering) -> Bytes {
  self.recipe
}

///|
pub fn LunaCudaAotLowering::recipe_digest(
  self : LunaCudaAotLowering,
) -> LunaCudaRecipeDigest {
  self.recipe_digest
}

///|
pub struct LunaCudaOfflineCompileReceipt {
  priv source_digest : LunaCudaSourceDigest
  priv recipe_digest : LunaCudaRecipeDigest
  priv toolchain_digest_sha256 : String
  priv driver_identity_sha256 : String
  priv first_artifact_digest : LunaCudaArtifactDigest
  priv second_artifact_digest : LunaCudaArtifactDigest
} derive(Debug, Eq)

///|
/// Constructs a receipt emitted by an isolated offline compiler driver.
///
/// This value binds reported identities; `admit_luna_cuda_aot_output` still
/// recomputes both module identities and compares every byte before admitting
/// deterministic output.
pub fn LunaCudaOfflineCompileReceipt::from_offline_builder(
  source_digest~ : LunaCudaSourceDigest,
  recipe_digest~ : LunaCudaRecipeDigest,
  toolchain_digest_sha256~ : String,
  driver_identity_sha256~ : String,
  first_artifact_digest~ : LunaCudaArtifactDigest,
  second_artifact_digest~ : LunaCudaArtifactDigest,
) -> LunaCudaOfflineCompileReceipt raise LunaCudaAotError {
  if !luna_cuda_aot_is_sha256(toolchain_digest_sha256) ||
    !luna_cuda_aot_is_sha256(driver_identity_sha256) {
    raise Invalid(Toolchain)
  }
  {
    source_digest,
    recipe_digest,
    toolchain_digest_sha256,
    driver_identity_sha256,
    first_artifact_digest,
    second_artifact_digest,
  }
}

///|
pub struct LunaCudaAotArtifact {
  priv target : @catalog.DeviceTarget
  priv entry_point_id : @catalog.AotKernelEntryPointId
  priv function_symbol : String
  priv operand_byte_count : Int64
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv module_bytes : Bytes
  priv module_digest : @catalog.AotArtifactDigest
  priv source_digest : LunaCudaSourceDigest
  priv recipe_digest : LunaCudaRecipeDigest
  priv driver_identity_sha256 : String
  priv evidence : Bytes
  priv evidence_digest : LunaCudaEvidenceDigest
} derive(Debug, Eq)

///|
/// Offline artifact joined to the exact admitted specialization that may name
/// it in a catalog and execution manifest.
pub struct LunaCudaAotSpecializedArtifact {
  priv artifact : LunaCudaAotArtifact
  priv specialization_digest : @luna_specialization.LunaSpecializationDigest
  priv operation_id : @plan.OperationId
} derive(Debug, Eq)
