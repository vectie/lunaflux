///|
pub fn DeepSeekV4TokenHashLaunchBinding::status(
  _self : Self,
) -> DeepSeekV4TokenHashLaunchBindingStatus {
  InertCompleteUnqualified
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::source_identity(
  self : Self,
) -> @spec.ModelIdentity {
  self.source_identity
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::requirements_digest(
  self : Self,
) -> @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirementsDigest {
  self.requirements_digest
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::layer_index(self : Self) -> Int {
  self.layer_index
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::requirement_ordinal(
  self : Self,
) -> Int {
  self.requirement_ordinal
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::maximum_tokens(self : Self) -> Int {
  self.maximum_tokens
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::experts_per_token(self : Self) -> Int {
  self.experts_per_token
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::routed_expert_count(
  self : Self,
) -> Int {
  self.routed_expert_count
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::operand_plan_digest(
  self : Self,
) -> DeepSeekV4TokenHashOperandPlanDigest {
  self.operand_plan_digest
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::sidecar_layout_digest(
  self : Self,
) -> @segmented_device_materialize.SegmentedDeviceLayoutDigest {
  self.sidecar_layout_digest
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::artifact_binding_digest(
  self : Self,
) -> DeepSeekV4TokenHashArtifactBindingDigest {
  self.artifact_binding_digest
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::compile_binding_digest(
  self : Self,
) -> DeepSeekV4TokenHashCompileEvidenceBindingDigest {
  self.compile_binding_digest
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::compile_evidence_digest(
  self : Self,
) -> @cuda_offline_compile_evidence.CudaOfflineCompileEvidenceDigest {
  self.compile_evidence_digest
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::operand_regions(
  self : Self,
) -> @advanced_decoder_operand_region_plan.AdvancedDecoderOperandRegionPlan {
  self.operand_regions
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::source_digest_sha256(
  self : Self,
) -> String {
  self.source_digest_sha256
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::recipe_digest_sha256(
  self : Self,
) -> String {
  self.recipe_digest_sha256
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::compiler_policy(
  self : Self,
) -> @luna_specialization.LunaCompilerPolicy {
  self.compiler
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::target(
  self : Self,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::entry_point(
  self : Self,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::function_symbol(self : Self) -> String {
  self.function_symbol
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::module_digest(
  self : Self,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::dimensions(
  self : Self,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::digest(
  self : Self,
) -> DeepSeekV4TokenHashLaunchBindingDigest {
  self.digest
}

///|
pub fn DeepSeekV4TokenHashLaunchBindingDigest::as_hex(self : Self) -> String {
  let DeepSeekV4TokenHashLaunchBindingDigest(hex) = self
  hex
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::require_loader_authority(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashLaunchBindingError {
  raise LoaderAuthorityUnavailable
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::require_launch_authority(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashLaunchBindingError {
  raise LaunchAuthorityUnavailable
}

///|
pub fn DeepSeekV4TokenHashLaunchBinding::require_physical_qualification(
  _self : Self,
) -> Unit raise DeepSeekV4TokenHashLaunchBindingError {
  raise PhysicalQualificationUnavailable
}
