///|
pub fn Fp8LaunchAbiVersion::v2() -> Fp8LaunchAbiVersion {
  Fp8LaunchAbiVersion(2)
}

///|
pub(all) enum Fp8StagedDynamicScaleKind {
  SimpleExternalFiniteE4M3F32V2
  GatedExternalAndPostSiluFiniteE4M3F32V2
} derive(Debug, Eq)

///|
/// Exact admitted raw-launch authority retained by ABI v2.
///
/// V2 deliberately accepts only catalog-v4 paged contracts because they are
/// the first launch-contract schema that authenticates numeric execution and
/// interleaved weight/scale operands.
pub(all) enum Fp8V2LaunchContractSourceVersion {
  PagedNumericV4
} derive(Debug, Eq)

///|
/// One package-derived 4-byte cell inside the single caller-owned Workspace.
/// Callers cannot construct or reorder stage bindings.
pub struct Fp8ActivationScaleStageRegion {
  priv stage : @numeric_contract.ActivationScaleStage
  priv ordinal : Int
  priv region : Fp8LaunchStorageRegion
} derive(Debug, Eq)

///|
pub fn Fp8ActivationScaleStageRegion::stage(
  self : Fp8ActivationScaleStageRegion,
) -> @numeric_contract.ActivationScaleStage {
  self.stage
}

///|
pub fn Fp8ActivationScaleStageRegion::ordinal(
  self : Fp8ActivationScaleStageRegion,
) -> Int {
  self.ordinal
}

///|
pub fn Fp8ActivationScaleStageRegion::region(
  self : Fp8ActivationScaleStageRegion,
) -> Fp8LaunchStorageRegion {
  self.region
}

///|
/// Version-2 whole-Workspace FP8 ABI. One authenticated caller region owns
/// every ordered activation-scale stage; typed subregions are derived here.
pub struct Fp8StagedDynamicLaunchAbi {
  priv version : Fp8LaunchAbiVersion
  priv scope : Fp8LaunchScope
  priv plan_identity : @spec.ModelIdentity
  priv numeric_schema : @numeric_contract.ModelNumericSchemaDigest
  priv manifest_digest : @numeric_capability_manifest.Fp8KernelCapabilityManifestDigest
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.KernelProfileId
  priv operation : @plan.OperationId
  priv operation_kind : @plan.OperationKind
  priv execution : @numeric_contract.OperationExecutionDigest
  priv module_digest : @catalog.AotArtifactDigest
  priv entry_point : @catalog.AotKernelEntryPoint
  priv launch_source_version : Fp8V2LaunchContractSourceVersion
  priv launch_dimensions : @launch_contract.AotLaunchDimensions
  priv raw_operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv shape : Fp8ExactExecutionShape
  priv policy : @numeric_contract.ActivationScalePolicy
  priv scale_kind : Fp8StagedDynamicScaleKind
  priv weight_scales : ReadOnlyArray[
    @numeric_capability_manifest.Fp8KernelWeightBinding,
  ]
  priv scale_tensor_regions : ReadOnlyArray[Fp8ScaleTensorStorageRegion]
  priv workspace : Fp8LaunchStorageRegion
  priv stages : ReadOnlyArray[Fp8ActivationScaleStageRegion]
  priv digest : Fp8LaunchAbiDigest
} derive(Debug, Eq)

///|
pub fn Fp8StagedDynamicLaunchAbi::version(
  self : Fp8StagedDynamicLaunchAbi,
) -> Fp8LaunchAbiVersion {
  self.version
}

///|
pub fn Fp8StagedDynamicLaunchAbi::scope(
  self : Fp8StagedDynamicLaunchAbi,
) -> Fp8LaunchScope {
  self.scope
}

///|
pub fn Fp8StagedDynamicLaunchAbi::plan_identity(
  self : Fp8StagedDynamicLaunchAbi,
) -> @spec.ModelIdentity {
  self.plan_identity
}

///|
pub fn Fp8StagedDynamicLaunchAbi::numeric_schema_digest(
  self : Fp8StagedDynamicLaunchAbi,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.numeric_schema
}

///|
pub fn Fp8StagedDynamicLaunchAbi::manifest_digest(
  self : Fp8StagedDynamicLaunchAbi,
) -> @numeric_capability_manifest.Fp8KernelCapabilityManifestDigest {
  self.manifest_digest
}

///|
pub fn Fp8StagedDynamicLaunchAbi::target(
  self : Fp8StagedDynamicLaunchAbi,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn Fp8StagedDynamicLaunchAbi::profile(
  self : Fp8StagedDynamicLaunchAbi,
) -> @launch_contract.KernelProfileId {
  self.profile
}

///|
pub fn Fp8StagedDynamicLaunchAbi::operation(
  self : Fp8StagedDynamicLaunchAbi,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn Fp8StagedDynamicLaunchAbi::operation_kind(
  self : Fp8StagedDynamicLaunchAbi,
) -> @plan.OperationKind {
  self.operation_kind
}

///|
pub fn Fp8StagedDynamicLaunchAbi::execution_digest(
  self : Fp8StagedDynamicLaunchAbi,
) -> @numeric_contract.OperationExecutionDigest {
  self.execution
}

///|
pub fn Fp8StagedDynamicLaunchAbi::module_digest(
  self : Fp8StagedDynamicLaunchAbi,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn Fp8StagedDynamicLaunchAbi::entry_point(
  self : Fp8StagedDynamicLaunchAbi,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn Fp8StagedDynamicLaunchAbi::launch_source_version(
  self : Fp8StagedDynamicLaunchAbi,
) -> Fp8V2LaunchContractSourceVersion {
  self.launch_source_version
}

///|
pub fn Fp8StagedDynamicLaunchAbi::launch_dimensions(
  self : Fp8StagedDynamicLaunchAbi,
) -> @launch_contract.AotLaunchDimensions {
  self.launch_dimensions
}

///|
pub fn Fp8StagedDynamicLaunchAbi::raw_operands(
  self : Fp8StagedDynamicLaunchAbi,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.raw_operands
}

///|
pub fn Fp8StagedDynamicLaunchAbi::exact_shape(
  self : Fp8StagedDynamicLaunchAbi,
) -> Fp8ExactExecutionShape {
  self.shape
}

///|
pub fn Fp8StagedDynamicLaunchAbi::activation_scale_policy(
  self : Fp8StagedDynamicLaunchAbi,
) -> @numeric_contract.ActivationScalePolicy {
  self.policy
}

///|
pub fn Fp8StagedDynamicLaunchAbi::scale_kind(
  self : Fp8StagedDynamicLaunchAbi,
) -> Fp8StagedDynamicScaleKind {
  self.scale_kind
}

///|
pub fn Fp8StagedDynamicLaunchAbi::weight_scales(
  self : Fp8StagedDynamicLaunchAbi,
) -> ReadOnlyArray[@numeric_capability_manifest.Fp8KernelWeightBinding] {
  self.weight_scales
}

///|
pub fn Fp8StagedDynamicLaunchAbi::scale_tensor_regions(
  self : Fp8StagedDynamicLaunchAbi,
) -> ReadOnlyArray[Fp8ScaleTensorStorageRegion] {
  self.scale_tensor_regions
}

///|
pub fn Fp8StagedDynamicLaunchAbi::workspace(
  self : Fp8StagedDynamicLaunchAbi,
) -> Fp8LaunchStorageRegion {
  self.workspace
}

///|
pub fn Fp8StagedDynamicLaunchAbi::stages(
  self : Fp8StagedDynamicLaunchAbi,
) -> ReadOnlyArray[Fp8ActivationScaleStageRegion] {
  self.stages
}

///|
pub fn Fp8StagedDynamicLaunchAbi::digest(
  self : Fp8StagedDynamicLaunchAbi,
) -> Fp8LaunchAbiDigest {
  self.digest
}
