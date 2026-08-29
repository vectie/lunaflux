///|
pub struct Fp8KernelCapabilityManifestVersion(Int) derive(Debug, Eq)

///|
pub fn Fp8KernelCapabilityManifestVersion::v1() -> Fp8KernelCapabilityManifestVersion {
  Fp8KernelCapabilityManifestVersion(1)
}

///|
pub fn Fp8KernelCapabilityManifestVersion::as_int(
  self : Fp8KernelCapabilityManifestVersion,
) -> Int {
  self.0
}

///|
pub(all) enum Fp8KernelExecutionScope {
  StatelessFullContextV1
  PagedFullGraphV1
} derive(Debug, Eq)

///|
pub struct Fp8KernelCapabilityManifestDigest(String) derive(Debug, Eq, Hash)

///|
pub fn Fp8KernelCapabilityManifestDigest::as_hex(
  self : Fp8KernelCapabilityManifestDigest,
) -> String {
  self.0
}

///|
/// Deployment claim for one exact profile and FP8 operation launch.
///
/// The claim owns no module bytes, symbol text, device, or execution resource.
pub struct Fp8KernelManifestClaim {
  priv profile : @launch_contract.KernelProfileId
  priv operation : @plan.OperationId
  priv execution : @numeric_contract.OperationExecutionDigest
  priv entry_point : @catalog.AotKernelEntryPoint
} derive(Debug, Eq)

///|
pub fn Fp8KernelManifestClaim::new(
  profile~ : @launch_contract.KernelProfileId,
  operation~ : @plan.OperationId,
  execution~ : @numeric_contract.OperationExecutionDigest,
  entry_point~ : @catalog.AotKernelEntryPoint,
) -> Fp8KernelManifestClaim {
  { profile, operation, execution, entry_point }
}

///|
/// Exact model-schema identity for one FP8 weight and its scalar F32 scale.
pub struct Fp8KernelWeightBinding {
  priv tensor : @plan.TensorRef
  priv storage : @numeric_contract.TensorStorageDigest
  priv scale : @plan.ScaleTensorRef
  priv scale_storage : @numeric_contract.TensorStorageDigest
} derive(Debug, Eq)

///|
/// Immutable admitted view of one exact FP8 launch capability.
pub struct Fp8KernelManifestEntry {
  priv profile : @launch_contract.KernelProfileId
  priv operation : @plan.OperationId
  priv execution : @numeric_contract.OperationExecutionDigest
  priv entry_point : @catalog.AotKernelEntryPoint
  priv module_digest : @catalog.AotArtifactDigest
  priv artifact_module_index : Int
  priv weights : ReadOnlyArray[Fp8KernelWeightBinding]
} derive(Debug, Eq)

///|
/// Canonical numeric kernel capability for one model, target, and launch set.
pub struct Fp8KernelCapabilityManifest {
  priv version : Fp8KernelCapabilityManifestVersion
  priv scope : Fp8KernelExecutionScope
  priv model_identity : @spec.ModelIdentity
  priv numeric_schema : @numeric_contract.ModelNumericSchemaDigest
  priv device_numeric : @device.DeviceNumericCapabilityDigest
  priv target : @catalog.DeviceTarget
  priv catalog_version : @catalog.CatalogVersion
  priv entries : ReadOnlyArray[Fp8KernelManifestEntry]
  priv canonical : Bytes
  priv digest : Fp8KernelCapabilityManifestDigest
} derive(Debug, Eq)

///|
/// Inert startup admission retaining exactly one previously admitted artifact
/// bundle plus immutable per-operation views. It loads and launches nothing.
pub struct AdmittedFp8KernelArtifacts {
  priv manifest : Fp8KernelCapabilityManifest
  priv artifacts : @artifact.KernelArtifactBundle
}

///|
pub struct Fp8KernelManifestLimits {
  priv max_entries : Int
  priv max_canonical_bytes : Int
} derive(Debug, Eq)

///|
pub fn Fp8KernelManifestLimits::new(
  max_entries~ : Int,
  max_canonical_bytes~ : Int,
) -> Fp8KernelManifestLimits raise Fp8KernelManifestError {
  if max_entries <= 0 ||
    max_entries > 65536 ||
    max_canonical_bytes <= 0 ||
    max_canonical_bytes > 64 * 1024 * 1024 {
    raise InvalidFp8KernelManifest(Limit)
  }
  { max_entries, max_canonical_bytes }
}

///|
pub(all) enum Fp8KernelManifestRule {
  Limit
  ExecutionScope
  ModelIdentity
  DeviceCapability
  DeviceTarget
  CatalogVersion
  BindingOrder
  ExecutionContract
  MissingFp8Operation
  VendorOrBf16Fallback
  ClaimCount
  ClaimOrder
  ClaimExecution
  ClaimEntryPoint
  ArtifactBundle
  CanonicalBytes
} derive(Debug, Eq)

///|
pub(all) suberror Fp8KernelManifestError {
  InvalidFp8KernelManifest(Fp8KernelManifestRule)
} derive(Debug, Eq)

///|
priv struct ExpectedFp8Operation {
  operation : @plan.OperationId
  execution : @numeric_contract.OperationExecutionDigest
  family : @catalog.AotKernelFamily
  weights : ReadOnlyArray[Fp8KernelWeightBinding]
}
