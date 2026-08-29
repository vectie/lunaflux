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
  luna_authorization : ManifestLunaAuthorizationClaim?
}

///|
priv struct ManifestV4NumericClaim {
  i8_admission_digest : String
  weight_artifact_digest : @numeric_weight_file.NumericWeightArtifactDigest
}

///|
priv struct ManifestV4Claims {
  profile_id : @launch_contract.KernelProfileId
  modules : Array[ManifestModuleClaim]
  operations : Array[ManifestOperationClaim]
  numeric : ManifestV4NumericClaim
}

///|
priv struct ManifestLunaAuthorizationClaim {
  operation_id : Int
  specialization_locator : String
  specialization_digest : String
  capability_locator : String
  capability_digest : String
  key_identity : String
  signature_digest : String
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
  priv luna_authorization : @luna_artifact_admission.LunaAotKernelAdmission?
  priv blueprint : @device_step.PagedKvExecutionBlueprint
}

///|
/// Exact inert schema-v4/catalog-v4 production-I8 execution admission.
///
/// This aggregate owns immutable artifact bytes and planning evidence only.
/// It has no authenticated loaded-weight authority and therefore cannot
/// construct an executor, bootstrap a ready worker, or open a device.
pub struct I8PagedExecutionAdmissionV4 {
  priv model_plan : @plan.ModelPlan
  priv device_numeric : @device.DeviceNumericCapability
  priv production_policy : @i8_admission.I8ProductionPolicyAdmission
  priv weight_layout : @device_materialize.DeviceWeightLayout
  priv layout : @device_layout.DeviceKvLayout
  priv device_step_limits : @device_step.DeviceStepLimits
  priv static_plan : @device_plan.StaticDevicePlan
  priv memory_plan : @device_memory.DeviceMemoryPlan
  priv profile_id : @launch_contract.KernelProfileId
  priv contracts : @launch_contract.PagedV4LaunchContractSet
  priv artifacts : @artifact.KernelArtifactBundle
  priv blueprint : @device_step.I8PagedExecutionBlueprint
  priv bootstrap : @device_step.I8DeviceWorkerBootstrapManifestV3
  priv weight_artifact_digest : @numeric_weight_file.NumericWeightArtifactDigest
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
/// Returns the sole opaque Phase-5 authorization, when schema v3 required it.
pub fn PagedExecutionAdmission::luna_authorization(
  self : PagedExecutionAdmission,
) -> @luna_artifact_admission.LunaAotKernelAdmission? {
  self.luna_authorization
}

///|
pub fn PagedExecutionAdmission::blueprint(
  self : PagedExecutionAdmission,
) -> @device_step.PagedKvExecutionBlueprint {
  self.blueprint
}

///|
pub fn I8PagedExecutionAdmissionV4::model_plan(
  self : I8PagedExecutionAdmissionV4,
) -> @plan.ModelPlan {
  self.model_plan
}

///|
pub fn I8PagedExecutionAdmissionV4::device_numeric_capability(
  self : I8PagedExecutionAdmissionV4,
) -> @device.DeviceNumericCapability {
  self.device_numeric
}

///|
pub fn I8PagedExecutionAdmissionV4::production_policy(
  self : I8PagedExecutionAdmissionV4,
) -> @i8_admission.I8ProductionPolicyAdmission {
  self.production_policy
}

///|
pub fn I8PagedExecutionAdmissionV4::weight_layout(
  self : I8PagedExecutionAdmissionV4,
) -> @device_materialize.DeviceWeightLayout {
  self.weight_layout
}

///|
pub fn I8PagedExecutionAdmissionV4::layout(
  self : I8PagedExecutionAdmissionV4,
) -> @device_layout.DeviceKvLayout {
  self.layout
}

///|
pub fn I8PagedExecutionAdmissionV4::device_step_limits(
  self : I8PagedExecutionAdmissionV4,
) -> @device_step.DeviceStepLimits {
  self.device_step_limits
}

///|
pub fn I8PagedExecutionAdmissionV4::static_plan(
  self : I8PagedExecutionAdmissionV4,
) -> @device_plan.StaticDevicePlan {
  self.static_plan
}

///|
pub fn I8PagedExecutionAdmissionV4::memory_plan(
  self : I8PagedExecutionAdmissionV4,
) -> @device_memory.DeviceMemoryPlan {
  self.memory_plan
}

///|
pub fn I8PagedExecutionAdmissionV4::profile_id(
  self : I8PagedExecutionAdmissionV4,
) -> @launch_contract.KernelProfileId {
  self.profile_id
}

///|
pub fn I8PagedExecutionAdmissionV4::contracts(
  self : I8PagedExecutionAdmissionV4,
) -> @launch_contract.PagedV4LaunchContractSet {
  self.contracts
}

///|
pub fn I8PagedExecutionAdmissionV4::artifacts(
  self : I8PagedExecutionAdmissionV4,
) -> @artifact.KernelArtifactBundle {
  self.artifacts
}

///|
pub fn I8PagedExecutionAdmissionV4::blueprint(
  self : I8PagedExecutionAdmissionV4,
) -> @device_step.I8PagedExecutionBlueprint {
  self.blueprint
}

///|
pub fn I8PagedExecutionAdmissionV4::bootstrap_manifest(
  self : I8PagedExecutionAdmissionV4,
) -> @device_step.I8DeviceWorkerBootstrapManifestV3 {
  self.bootstrap
}

///|
pub fn I8PagedExecutionAdmissionV4::weight_artifact_digest(
  self : I8PagedExecutionAdmissionV4,
) -> @numeric_weight_file.NumericWeightArtifactDigest {
  self.weight_artifact_digest
}

///|
/// Performs the final four-way loaded-weight join while preserving an inert,
/// non-executable result. The returned owner has explicit close authority but
/// exposes no allocation or device-executor constructor.
pub fn I8PagedExecutionAdmissionV4::join_loaded_weights(
  self : I8PagedExecutionAdmissionV4,
  weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority,
) -> @device_step.I8LoadedExecutionBlueprint raise ExecutionManifestFileError {
  @device_step.join_i8_loaded_execution_blueprint_v1(
    self.blueprint,
    self.bootstrap,
    weights,
    expected_weight_artifact_digest=self.weight_artifact_digest,
  ) catch {
    _ => raise AdmissionRejected(Blueprint)
  }
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
