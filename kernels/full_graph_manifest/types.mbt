///|
/// Bounds for deterministic schema-v2 execution-manifest production.
///
/// Artifact path and symbol bounds are delegated to the same typed limits used
/// by runtime artifact-file admission.
pub struct FullGraphManifestLimits {
  priv max_manifest_bytes : Int64
  priv artifact_file : @artifact_file.KernelArtifactFileLimits
}

///|
pub fn FullGraphManifestLimits::new(
  max_manifest_bytes~ : Int64,
  artifact_file~ : @artifact_file.KernelArtifactFileLimits,
) -> FullGraphManifestLimits raise FullGraphManifestError {
  if max_manifest_bytes <= 0L ||
    max_manifest_bytes > artifact_file.max_manifest_bytes() {
    raise Invalid(Limits)
  }
  { max_manifest_bytes, artifact_file }
}

///|
pub fn FullGraphManifestLimits::max_manifest_bytes(
  self : FullGraphManifestLimits,
) -> Int64 {
  self.max_manifest_bytes
}

///|
pub fn FullGraphManifestLimits::artifact_file(
  self : FullGraphManifestLimits,
) -> @artifact_file.KernelArtifactFileLimits {
  self.artifact_file
}

///|
/// Exact offline artifact metadata for one semantic model-plan operation.
///
/// Launch dimensions and ordered operands come from the artifact producer.
/// They are checked against the independently admitted FullGraph contract
/// before any manifest bytes are emitted.
pub struct FullGraphAotOperationArtifact {
  priv operation_id : @plan.OperationId
  priv entry_point : @catalog.AotKernelEntryPoint
  priv module_relative_path : String
  priv function_symbol : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv workspace_bytes : Int64
  priv workspace_alignment : Int64
}

///|
pub fn FullGraphAotOperationArtifact::new(
  operation_id~ : @plan.OperationId,
  entry_point~ : @catalog.AotKernelEntryPoint,
  module_relative_path~ : String,
  function_symbol~ : String,
  dimensions~ : @launch_contract.AotLaunchDimensions,
  operands~ : ArrayView[@launch_contract.AotOperand],
  workspace_bytes~ : Int64,
  workspace_alignment~ : Int64,
) -> FullGraphAotOperationArtifact raise FullGraphManifestError {
  if operands.is_empty() || workspace_bytes < 0L || workspace_alignment <= 0L {
    raise Invalid(OperandContract)
  }
  if (workspace_alignment & (workspace_alignment - 1L)) != 0L {
    raise Invalid(WorkspaceContract)
  }
  {
    operation_id,
    entry_point,
    module_relative_path,
    function_symbol,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    workspace_bytes,
    workspace_alignment,
  }
}

///|
pub fn FullGraphAotOperationArtifact::operation_id(
  self : FullGraphAotOperationArtifact,
) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn FullGraphAotOperationArtifact::entry_point(
  self : FullGraphAotOperationArtifact,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn FullGraphAotOperationArtifact::module_relative_path(
  self : FullGraphAotOperationArtifact,
) -> String {
  self.module_relative_path
}

///|
pub fn FullGraphAotOperationArtifact::function_symbol(
  self : FullGraphAotOperationArtifact,
) -> String {
  self.function_symbol
}

///|
pub fn FullGraphAotOperationArtifact::launch_dimensions(
  self : FullGraphAotOperationArtifact,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn FullGraphAotOperationArtifact::operands(
  self : FullGraphAotOperationArtifact,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn FullGraphAotOperationArtifact::workspace_bytes(
  self : FullGraphAotOperationArtifact,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn FullGraphAotOperationArtifact::workspace_alignment(
  self : FullGraphAotOperationArtifact,
) -> Int64 {
  self.workspace_alignment
}

///|
/// Canonical schema-v2 bytes plus their independent SHA-256 identity.
pub struct FullGraphExecutionManifestInput {
  priv bytes : Bytes
  priv digest_sha256 : String
  priv module_count : Int
  priv entry_point_count : Int
  priv operation_count : Int
}

///|
pub fn FullGraphExecutionManifestInput::bytes(
  self : FullGraphExecutionManifestInput,
) -> Bytes {
  self.bytes
}

///|
pub fn FullGraphExecutionManifestInput::digest_sha256(
  self : FullGraphExecutionManifestInput,
) -> String {
  self.digest_sha256
}

///|
pub fn FullGraphExecutionManifestInput::module_count(
  self : FullGraphExecutionManifestInput,
) -> Int {
  self.module_count
}

///|
pub fn FullGraphExecutionManifestInput::entry_point_count(
  self : FullGraphExecutionManifestInput,
) -> Int {
  self.entry_point_count
}

///|
pub fn FullGraphExecutionManifestInput::operation_count(
  self : FullGraphExecutionManifestInput,
) -> Int {
  self.operation_count
}

///|
priv struct FullGraphManifestEntry {
  family_id : @catalog.AotKernelFamilyId
  entry_point_id : @catalog.AotKernelEntryPointId
  function_symbol : String
}

///|
priv struct FullGraphManifestModule {
  digest : @catalog.AotArtifactDigest
  path : String
  entries : Array[FullGraphManifestEntry]
}
