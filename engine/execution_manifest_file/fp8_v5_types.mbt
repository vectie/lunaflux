///|
priv struct ManifestV5NumericClaim {
  weight_artifact_digest : @numeric_weight_file.NumericWeightArtifactDigest
  kernel_manifest_digest : String
  runtime_recipe_v2_digest : String
  reusable_recipe_v3_digest : String
  workspace_storage_digest : String
}

///|
priv struct ManifestV5Claims {
  profile_id : @launch_contract.KernelProfileId
  modules : Array[ManifestModuleClaim]
  operations : Array[ManifestOperationClaim]
  numeric : ManifestV5NumericClaim
  release : ManifestV5ReleaseClaim
}

///|
priv struct ManifestV5ReleaseOperationClaim {
  operation : @plan.OperationId
  module_locator : String
  source_digest : String
  recipe_digest : String
}

///|
priv struct ManifestV5ReleaseClaim {
  manifest_locator : String
  operations : Array[ManifestV5ReleaseOperationClaim]
}

///|
/// Inert reconstruction of the exact FP8 PagedV4 startup and reusable recipe.
/// It owns immutable artifact bytes only; no external approval, device,
/// module, allocation, executor, worker, or readiness authority is present.
pub struct Fp8PagedExecutionAdmissionV5 {
  priv model_plan : @plan.ModelPlan
  priv weight_inspection : @numeric_weight_file.NumericWeightFileInspection
  priv kv_layout : @device_layout.DeviceKvLayout
  priv device_step_limits : @device_step.DeviceStepLimits
  priv static_plan : @device_plan.StaticDevicePlan
  priv memory_plan : @device_memory.DeviceMemoryPlan
  priv profile : @launch_contract.PagedKernelProfile
  priv contracts : @launch_contract.PagedV4LaunchContractSet
  priv artifacts : @artifact.KernelArtifactBundle
  priv numeric_artifacts : @numeric_capability_manifest.AdmittedFp8KernelArtifacts
  priv startup : @fp8_startup_admission.Fp8StartupAdmissionV2
  priv exact_recipe : @fp8_runtime_recipe.Fp8RuntimeRecipeV2
  priv reusable_recipe : @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3
  priv release_authority : @fp8_release_authority.Fp8ReusablePagedEnvelopeAuthorityV3
}

///|
pub fn Fp8PagedExecutionAdmissionV5::model_plan(self : Self) -> @plan.ModelPlan {
  self.model_plan
}

///|
pub fn Fp8PagedExecutionAdmissionV5::weight_inspection(
  self : Self,
) -> @numeric_weight_file.NumericWeightFileInspection {
  self.weight_inspection
}

///|
pub fn Fp8PagedExecutionAdmissionV5::kv_layout(
  self : Self,
) -> @device_layout.DeviceKvLayout {
  self.kv_layout
}

///|
pub fn Fp8PagedExecutionAdmissionV5::static_plan(
  self : Self,
) -> @device_plan.StaticDevicePlan {
  self.static_plan
}

///|
pub fn Fp8PagedExecutionAdmissionV5::memory_plan(
  self : Self,
) -> @device_memory.DeviceMemoryPlan {
  self.memory_plan
}

///|
pub fn Fp8PagedExecutionAdmissionV5::profile(
  self : Self,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn Fp8PagedExecutionAdmissionV5::device_step_limits(
  self : Self,
) -> @device_step.DeviceStepLimits {
  self.device_step_limits
}

///|
pub fn Fp8PagedExecutionAdmissionV5::contracts(
  self : Self,
) -> @launch_contract.PagedV4LaunchContractSet {
  self.contracts
}

///|
pub fn Fp8PagedExecutionAdmissionV5::artifacts(
  self : Self,
) -> @artifact.KernelArtifactBundle {
  self.artifacts
}

///|
pub fn Fp8PagedExecutionAdmissionV5::numeric_artifacts(
  self : Self,
) -> @numeric_capability_manifest.AdmittedFp8KernelArtifacts {
  self.numeric_artifacts
}

///|
pub fn Fp8PagedExecutionAdmissionV5::startup(
  self : Self,
) -> @fp8_startup_admission.Fp8StartupAdmissionV2 {
  self.startup
}

///|
pub fn Fp8PagedExecutionAdmissionV5::reusable_recipe(
  self : Self,
) -> @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3 {
  self.reusable_recipe
}

///|
pub fn Fp8PagedExecutionAdmissionV5::exact_recipe(
  self : Self,
) -> @fp8_runtime_recipe.Fp8RuntimeRecipeV2 {
  self.exact_recipe
}

///|
pub fn Fp8PagedExecutionAdmissionV5::release_authority(
  self : Self,
) -> @fp8_release_authority.Fp8ReusablePagedEnvelopeAuthorityV3 {
  self.release_authority
}
