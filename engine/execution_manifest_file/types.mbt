///|
/// Independent lowercase SHA-256 identity of the exact manifest bytes.
pub struct ExecutionManifestDigest(String) derive(Debug, Eq)

///|
pub fn ExecutionManifestDigest::from_sha256(
  value : String,
) -> ExecutionManifestDigest raise ExecutionManifestFileError {
  if !is_lower_sha256(value) {
    raise InvalidExpectedDigest
  }
  ExecutionManifestDigest(value)
}

///|
pub fn ExecutionManifestDigest::as_hex(
  self : ExecutionManifestDigest,
) -> String {
  self.0
}

///|
priv struct ManifestModuleClaim {
  digest : @catalog.AotArtifactDigest
  path : String
  entries : Array[ManifestEntryClaim]
}

///|
priv struct ManifestEntryClaim {
  family_id : @catalog.AotKernelFamilyId
  entry_point_id : @catalog.AotKernelEntryPointId
  symbol : String
}

///|
priv struct ManifestOperationClaim {
  module_index : Int
  family_id : @catalog.AotKernelFamilyId
  entry_point_id : @catalog.AotKernelEntryPointId
  workspace_bytes : Int64
  workspace_alignment : Int64
  dimensions : @launch_contract.AotLaunchDimensions
}

///|
priv struct ManifestClaims {
  profile_id : @launch_contract.KernelProfileId
  modules : Array[ManifestModuleClaim]
  operations : Array[ManifestOperationClaim]
}

///|
/// Inert, fully admitted inputs for one paged device-worker execution plan.
///
/// The aggregate contains no file/root alias, mutable module storage, device
/// address, allocation, loaded native module, stream, or lifecycle authority.
/// Its immutable admitted artifact bundle is intentionally available to
/// device-worker startup.
pub struct PagedExecutionAdmission {
  priv model_plan : @plan.ModelPlan
  priv weight_layout : @device_materialize.DeviceWeightLayout
  priv layout : @device_layout.DeviceKvLayout
  priv device_step_limits : @device_step.DeviceStepLimits
  priv static_plan : @device_plan.StaticDevicePlan
  priv memory_plan : @device_memory.DeviceMemoryPlan
  priv profile_id : @launch_contract.KernelProfileId
  priv contracts : @launch_contract.PagedKvLaunchContractSet
  priv artifacts : @artifact.KernelArtifactBundle
  priv blueprint : @device_step.PagedKvExecutionBlueprint
}

///|
pub fn PagedExecutionAdmission::model_plan(
  self : PagedExecutionAdmission,
) -> @plan.ModelPlan {
  self.model_plan
}

///|
/// Returns the exact immutable device-weight layout supplied to `load_paged`.
pub fn PagedExecutionAdmission::weight_layout(
  self : PagedExecutionAdmission,
) -> @device_materialize.DeviceWeightLayout {
  self.weight_layout
}

///|
pub fn PagedExecutionAdmission::layout(
  self : PagedExecutionAdmission,
) -> @device_layout.DeviceKvLayout {
  self.layout
}

///|
pub fn PagedExecutionAdmission::device_step_limits(
  self : PagedExecutionAdmission,
) -> @device_step.DeviceStepLimits {
  self.device_step_limits
}

///|
pub fn PagedExecutionAdmission::static_plan(
  self : PagedExecutionAdmission,
) -> @device_plan.StaticDevicePlan {
  self.static_plan
}

///|
pub fn PagedExecutionAdmission::memory_plan(
  self : PagedExecutionAdmission,
) -> @device_memory.DeviceMemoryPlan {
  self.memory_plan
}

///|
pub fn PagedExecutionAdmission::profile_id(
  self : PagedExecutionAdmission,
) -> @launch_contract.KernelProfileId {
  self.profile_id
}

///|
pub fn PagedExecutionAdmission::contracts(
  self : PagedExecutionAdmission,
) -> @launch_contract.PagedKvLaunchContractSet {
  self.contracts
}

///|
pub fn PagedExecutionAdmission::artifacts(
  self : PagedExecutionAdmission,
) -> @artifact.KernelArtifactBundle {
  self.artifacts
}

///|
pub fn PagedExecutionAdmission::blueprint(
  self : PagedExecutionAdmission,
) -> @device_step.PagedKvExecutionBlueprint {
  self.blueprint
}

///|
fn is_lower_sha256(value : StringView) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}
