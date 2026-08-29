///|
pub struct Fp8ReusablePagedReleaseOperationV3 {
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
/// Opaque externally approved reusable paged release authority. It binds a
/// profile envelope; concrete worker frames still require device-step tokens.
pub struct Fp8ReusablePagedEnvelopeAuthorityV3 {
  priv recipe : @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3
  priv approval : @luna_capability_manifest.LunaAdmittedFp8ReusablePagedReleaseManifestV3
  priv operations : ReadOnlyArray[Fp8ReusablePagedReleaseOperationV3]
  priv digest : Fp8ReleaseAuthorityDigest
}

///|
pub fn Fp8ReusablePagedEnvelopeAuthorityV3::recipe(
  self : Fp8ReusablePagedEnvelopeAuthorityV3,
) -> @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3 {
  self.recipe
}

///|
pub fn Fp8ReusablePagedEnvelopeAuthorityV3::recipe_digest(
  self : Fp8ReusablePagedEnvelopeAuthorityV3,
) -> @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3Digest {
  self.recipe.digest()
}

///|
pub fn Fp8ReusablePagedEnvelopeAuthorityV3::target(
  self : Fp8ReusablePagedEnvelopeAuthorityV3,
) -> @catalog.DeviceTarget {
  self.recipe.startup().kernel_artifacts().manifest().target()
}

///|
pub fn Fp8ReusablePagedEnvelopeAuthorityV3::operations(
  self : Fp8ReusablePagedEnvelopeAuthorityV3,
) -> ReadOnlyArray[Fp8ReusablePagedReleaseOperationV3] {
  self.operations
}

///|
pub fn Fp8ReusablePagedEnvelopeAuthorityV3::digest(
  self : Fp8ReusablePagedEnvelopeAuthorityV3,
) -> Fp8ReleaseAuthorityDigest {
  self.digest
}

///|
pub fn Fp8ReusablePagedEnvelopeAuthorityV3::approved_source_identity(
  self : Fp8ReusablePagedEnvelopeAuthorityV3,
) -> @luna_capability_manifest.LunaApprovedImmutableSourceIdentity {
  self.approval.approved_source_identity()
}

///|
pub fn Fp8ReusablePagedEnvelopeAuthorityV3::manifest_digest(
  self : Fp8ReusablePagedEnvelopeAuthorityV3,
) -> @luna_capability_manifest.LunaCapabilityManifestDigest {
  self.approval.manifest().digest()
}

///|
pub fn Fp8ReusablePagedEnvelopeAuthorityV3::approved_key_identity(
  self : Fp8ReusablePagedEnvelopeAuthorityV3,
) -> @luna_capability_manifest.LunaSignatureKeyIdentityDigest {
  self.approval.approved_key_identity()
}

///|
pub fn Fp8ReusablePagedEnvelopeAuthorityV3::detached_signature_digest(
  self : Fp8ReusablePagedEnvelopeAuthorityV3,
) -> @luna_capability_manifest.LunaDetachedSignatureDigest {
  self.approval.detached_signature_digest()
}

///|
pub fn Fp8ReusablePagedReleaseOperationV3::operation(
  self : Fp8ReusablePagedReleaseOperationV3,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn Fp8ReusablePagedReleaseOperationV3::entry_point(
  self : Fp8ReusablePagedReleaseOperationV3,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn Fp8ReusablePagedReleaseOperationV3::module_digest(
  self : Fp8ReusablePagedReleaseOperationV3,
) -> @catalog.AotArtifactDigest {
  self.entry_point.family().module_digest()
}

///|
pub fn Fp8ReusablePagedReleaseOperationV3::function_symbol(
  self : Fp8ReusablePagedReleaseOperationV3,
) -> String {
  self.function_symbol
}

///|
pub fn Fp8ReusablePagedReleaseOperationV3::module_bytes(
  self : Fp8ReusablePagedReleaseOperationV3,
) -> Bytes {
  self.module_bytes
}

///|
pub fn Fp8ReusablePagedReleaseOperationV3::source_digest_sha256(
  self : Fp8ReusablePagedReleaseOperationV3,
) -> String {
  self.source_digest_sha256
}

///|
pub fn Fp8ReusablePagedReleaseOperationV3::recipe_digest_sha256(
  self : Fp8ReusablePagedReleaseOperationV3,
) -> String {
  self.recipe_digest_sha256
}

///|
pub fn Fp8ReusablePagedReleaseOperationV3::dimensions(
  self : Fp8ReusablePagedReleaseOperationV3,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn Fp8ReusablePagedReleaseOperationV3::operands(
  self : Fp8ReusablePagedReleaseOperationV3,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn Fp8ReusablePagedReleaseOperationV3::workspace_bytes(
  self : Fp8ReusablePagedReleaseOperationV3,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn Fp8ReusablePagedReleaseOperationV3::workspace_alignment(
  self : Fp8ReusablePagedReleaseOperationV3,
) -> Int64 {
  self.workspace_alignment
}
