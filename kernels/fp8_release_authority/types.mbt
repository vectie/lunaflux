///|
pub(all) enum Fp8ReleaseAuthorityRule {
  Limit
  TotalModuleBytes
  ManifestFile
  ExternalApproval
  Recipe
  ModelIdentity
  Target
  OperationOrder
  EntryPoint
  Symbol
  LaunchDimensions
  RawOperands
  Workspace
  ModuleFile
  ModuleDigest
  ArtifactBytes
} derive(Debug, Eq)

///|
pub(all) suberror Fp8ReleaseAuthorityError {
  InvalidFp8ReleaseAuthority(Fp8ReleaseAuthorityRule)
} derive(Debug, Eq)

///|
pub struct Fp8ReleaseAuthorityLimits {
  priv max_manifest_bytes : Int64
  priv max_module_bytes : Int64
  priv max_total_module_bytes : Int64
  priv max_operations : Int
}

///|
pub fn Fp8ReleaseAuthorityLimits::new(
  max_manifest_bytes~ : Int64,
  max_module_bytes~ : Int64,
  max_total_module_bytes~ : Int64,
  max_operations~ : Int,
) -> Fp8ReleaseAuthorityLimits raise Fp8ReleaseAuthorityError {
  if max_manifest_bytes <= 0L ||
    max_manifest_bytes > 16L * 1024L * 1024L ||
    max_module_bytes <= 0L ||
    max_module_bytes > 1024L * 1024L * 1024L ||
    max_total_module_bytes <= 0L ||
    max_total_module_bytes > 2147483647L ||
    max_module_bytes > max_total_module_bytes ||
    max_operations <= 0 ||
    max_operations > 65536 {
    raise InvalidFp8ReleaseAuthority(Limit)
  }
  {
    max_manifest_bytes,
    max_module_bytes,
    max_total_module_bytes,
    max_operations,
  }
}

///|
pub struct Fp8ReleaseOperationAuthorityV2 {
  priv operation : @plan.OperationId
  priv entry_point : @catalog.AotKernelEntryPoint
  priv function_symbol : String
  priv module_bytes : Bytes
  priv source_digest_sha256 : String
  priv recipe_digest_sha256 : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv workspace_bytes : Int64
  priv workspace_alignment : Int64
}

///|
pub struct Fp8ReleaseAuthorityV2 {
  priv recipe : @fp8_runtime_recipe.Fp8RuntimeRecipeV2
  priv approval : @luna_capability_manifest.LunaAdmittedFp8ReleaseManifestV2
  priv operations : ReadOnlyArray[Fp8ReleaseOperationAuthorityV2]
  priv digest : Fp8ReleaseAuthorityDigest
}

///|
pub struct Fp8ReleaseAuthorityDigest(String) derive(Debug, Eq, Hash)

///|
pub fn Fp8ReleaseAuthorityDigest::as_hex(
  self : Fp8ReleaseAuthorityDigest,
) -> String {
  self.0
}

///|
pub fn Fp8ReleaseAuthorityV2::recipe(
  self : Fp8ReleaseAuthorityV2,
) -> @fp8_runtime_recipe.Fp8RuntimeRecipeV2 {
  self.recipe
}

///|
pub fn Fp8ReleaseAuthorityV2::recipe_digest(
  self : Fp8ReleaseAuthorityV2,
) -> @fp8_runtime_recipe.Fp8RuntimeRecipeV2Digest {
  self.recipe.digest()
}

///|
pub fn Fp8ReleaseAuthorityV2::target(
  self : Fp8ReleaseAuthorityV2,
) -> @catalog.DeviceTarget {
  self.recipe.startup().kernel_artifacts().manifest().target()
}

///|
pub fn Fp8ReleaseAuthorityV2::operations(
  self : Fp8ReleaseAuthorityV2,
) -> ReadOnlyArray[Fp8ReleaseOperationAuthorityV2] {
  self.operations
}

///|
pub fn Fp8ReleaseAuthorityV2::digest(
  self : Fp8ReleaseAuthorityV2,
) -> Fp8ReleaseAuthorityDigest {
  self.digest
}

///|
pub fn Fp8ReleaseAuthorityV2::approved_source_identity(
  self : Fp8ReleaseAuthorityV2,
) -> @luna_capability_manifest.LunaApprovedImmutableSourceIdentity {
  self.approval.approved_source_identity()
}

///|
pub fn Fp8ReleaseOperationAuthorityV2::operation(
  self : Fp8ReleaseOperationAuthorityV2,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn Fp8ReleaseOperationAuthorityV2::entry_point(
  self : Fp8ReleaseOperationAuthorityV2,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn Fp8ReleaseOperationAuthorityV2::module_digest(
  self : Fp8ReleaseOperationAuthorityV2,
) -> @catalog.AotArtifactDigest {
  self.entry_point.family().module_digest()
}

///|
pub fn Fp8ReleaseOperationAuthorityV2::function_symbol(
  self : Fp8ReleaseOperationAuthorityV2,
) -> String {
  self.function_symbol
}

///|
pub fn Fp8ReleaseOperationAuthorityV2::module_bytes(
  self : Fp8ReleaseOperationAuthorityV2,
) -> Bytes {
  self.module_bytes
}

///|
pub fn Fp8ReleaseOperationAuthorityV2::source_digest_sha256(
  self : Fp8ReleaseOperationAuthorityV2,
) -> String {
  self.source_digest_sha256
}

///|
pub fn Fp8ReleaseOperationAuthorityV2::recipe_digest_sha256(
  self : Fp8ReleaseOperationAuthorityV2,
) -> String {
  self.recipe_digest_sha256
}

///|
pub fn Fp8ReleaseOperationAuthorityV2::dimensions(
  self : Fp8ReleaseOperationAuthorityV2,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn Fp8ReleaseOperationAuthorityV2::operands(
  self : Fp8ReleaseOperationAuthorityV2,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn Fp8ReleaseOperationAuthorityV2::workspace_bytes(
  self : Fp8ReleaseOperationAuthorityV2,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn Fp8ReleaseOperationAuthorityV2::workspace_alignment(
  self : Fp8ReleaseOperationAuthorityV2,
) -> Int64 {
  self.workspace_alignment
}
