///|
pub struct Fp8LaunchAbiVersion(Int) derive(Debug, Eq)

///|
pub fn Fp8LaunchAbiVersion::v1() -> Fp8LaunchAbiVersion {
  Fp8LaunchAbiVersion(1)
}

///|
pub fn Fp8LaunchAbiVersion::as_int(self : Fp8LaunchAbiVersion) -> Int {
  self.0
}

///|
pub(all) enum Fp8DynamicScaleKind {
  FiniteE4M3DynamicF32V1
} derive(Debug, Eq)

///|
pub struct Fp8LaunchAbiDigest(String) derive(Debug, Eq, Hash)

///|
pub fn Fp8LaunchAbiDigest::as_sha256(self : Fp8LaunchAbiDigest) -> String {
  self.0
}

///|
pub(all) enum Fp8LaunchContractSource {
  Stateless(@launch_contract.LaunchContractSet)
  Paged(@launch_contract.PagedKvLaunchContractSet)
  PagedV4(@launch_contract.PagedV4LaunchContractSet)
}

///|
/// Exact model scale-tensor identity joined to its runtime storage region.
/// Tensor and semantic-storage identity come from the admitted manifest;
/// callers provide only the opaque allocation-space region.
pub struct Fp8ScaleTensorStorageRegion {
  priv tensor : @plan.ScaleTensorRef
  priv storage_digest : @numeric_contract.TensorStorageDigest
  priv region : Fp8LaunchStorageRegion
} derive(Debug, Eq)

///|
pub fn Fp8ScaleTensorStorageRegion::tensor(
  self : Fp8ScaleTensorStorageRegion,
) -> @plan.ScaleTensorRef {
  self.tensor
}

///|
pub fn Fp8ScaleTensorStorageRegion::storage_digest(
  self : Fp8ScaleTensorStorageRegion,
) -> @numeric_contract.TensorStorageDigest {
  self.storage_digest
}

///|
pub fn Fp8ScaleTensorStorageRegion::region(
  self : Fp8ScaleTensorStorageRegion,
) -> Fp8LaunchStorageRegion {
  self.region
}

///|
priv struct SelectedLaunchContract {
  entry_point : @catalog.AotKernelEntryPoint
  workspace_bytes : Int64
  workspace_alignment : Int64
}

///|
pub struct Fp8DynamicLaunchAbi {
  priv version : Fp8LaunchAbiVersion
  priv scope : Fp8LaunchScope
  priv plan_identity : @spec.ModelIdentity
  priv numeric_schema : @numeric_contract.ModelNumericSchemaDigest
  priv manifest_digest : @numeric_capability_manifest.Fp8KernelCapabilityManifestDigest
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.KernelProfileId
  priv operation : @plan.OperationId
  priv execution : @numeric_contract.OperationExecutionDigest
  priv module_digest : @catalog.AotArtifactDigest
  priv entry_point : @catalog.AotKernelEntryPoint
  priv shape : Fp8ExactExecutionShape
  priv scale_kind : Fp8DynamicScaleKind
  priv scale_granularity : @numeric_contract.ScaleGranularity
  priv weight_scales : ReadOnlyArray[
    @numeric_capability_manifest.Fp8KernelWeightBinding,
  ]
  priv scale_tensor_regions : ReadOnlyArray[Fp8ScaleTensorStorageRegion]
  priv activation_scale_workspace : Fp8LaunchStorageRegion
  priv kernel_workspace : Fp8LaunchStorageRegion?
  priv workspace_bytes : Int64
  priv workspace_alignment : Int64
  priv digest : Fp8LaunchAbiDigest
} derive(Debug, Eq)

///|
pub fn Fp8DynamicLaunchAbi::version(
  self : Fp8DynamicLaunchAbi,
) -> Fp8LaunchAbiVersion {
  self.version
}

///|
pub fn Fp8DynamicLaunchAbi::scope(self : Fp8DynamicLaunchAbi) -> Fp8LaunchScope {
  self.scope
}

///|
pub fn Fp8DynamicLaunchAbi::plan_identity(
  self : Fp8DynamicLaunchAbi,
) -> @spec.ModelIdentity {
  self.plan_identity
}

///|
pub fn Fp8DynamicLaunchAbi::numeric_schema_digest(
  self : Fp8DynamicLaunchAbi,
) -> @numeric_contract.ModelNumericSchemaDigest {
  self.numeric_schema
}

///|
pub fn Fp8DynamicLaunchAbi::manifest_digest(
  self : Fp8DynamicLaunchAbi,
) -> @numeric_capability_manifest.Fp8KernelCapabilityManifestDigest {
  self.manifest_digest
}

///|
pub fn Fp8DynamicLaunchAbi::target(
  self : Fp8DynamicLaunchAbi,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn Fp8DynamicLaunchAbi::profile(
  self : Fp8DynamicLaunchAbi,
) -> @launch_contract.KernelProfileId {
  self.profile
}

///|
pub fn Fp8DynamicLaunchAbi::operation(
  self : Fp8DynamicLaunchAbi,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn Fp8DynamicLaunchAbi::execution_digest(
  self : Fp8DynamicLaunchAbi,
) -> @numeric_contract.OperationExecutionDigest {
  self.execution
}

///|
pub fn Fp8DynamicLaunchAbi::module_digest(
  self : Fp8DynamicLaunchAbi,
) -> @catalog.AotArtifactDigest {
  self.module_digest
}

///|
pub fn Fp8DynamicLaunchAbi::entry_point(
  self : Fp8DynamicLaunchAbi,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn Fp8DynamicLaunchAbi::exact_shape(
  self : Fp8DynamicLaunchAbi,
) -> Fp8ExactExecutionShape {
  self.shape
}

///|
pub fn Fp8DynamicLaunchAbi::scale_kind(
  self : Fp8DynamicLaunchAbi,
) -> Fp8DynamicScaleKind {
  self.scale_kind
}

///|
pub fn Fp8DynamicLaunchAbi::scale_granularity(
  self : Fp8DynamicLaunchAbi,
) -> @numeric_contract.ScaleGranularity {
  self.scale_granularity
}

///|
pub fn Fp8DynamicLaunchAbi::weight_scales(
  self : Fp8DynamicLaunchAbi,
) -> ReadOnlyArray[@numeric_capability_manifest.Fp8KernelWeightBinding] {
  self.weight_scales
}

///|
pub fn Fp8DynamicLaunchAbi::scale_tensor_regions(
  self : Fp8DynamicLaunchAbi,
) -> ReadOnlyArray[Fp8ScaleTensorStorageRegion] {
  self.scale_tensor_regions
}

///|
pub fn Fp8DynamicLaunchAbi::activation_scale_workspace(
  self : Fp8DynamicLaunchAbi,
) -> Fp8LaunchStorageRegion {
  self.activation_scale_workspace
}

///|
pub fn Fp8DynamicLaunchAbi::kernel_workspace(
  self : Fp8DynamicLaunchAbi,
) -> Fp8LaunchStorageRegion? {
  self.kernel_workspace
}

///|
pub fn Fp8DynamicLaunchAbi::workspace_bytes(
  self : Fp8DynamicLaunchAbi,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn Fp8DynamicLaunchAbi::workspace_alignment(
  self : Fp8DynamicLaunchAbi,
) -> Int64 {
  self.workspace_alignment
}

///|
pub fn Fp8DynamicLaunchAbi::digest(
  self : Fp8DynamicLaunchAbi,
) -> Fp8LaunchAbiDigest {
  self.digest
}
