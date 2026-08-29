///|
/// One externally reviewed FP8 ABI-v2 operation. Construction is inert: only
/// the separately admitted detached-signature approval grants authority.
pub struct LunaFp8ReleaseOperationV2 {
  priv operation : @plan.OperationId
  priv entry_point : @catalog.AotKernelEntryPoint
  priv module_locator : String
  priv function_symbol : String
  priv source_digest_sha256 : String
  priv recipe_digest_sha256 : String
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv operands : ReadOnlyArray[@launch_contract.AotOperand]
  priv workspace_bytes : Int64
  priv workspace_alignment : Int64
} derive(Debug, Eq)

///|
pub fn LunaFp8ReleaseOperationV2::new(
  operation~ : @plan.OperationId,
  entry_point~ : @catalog.AotKernelEntryPoint,
  module_locator~ : String,
  function_symbol~ : String,
  source_digest_sha256~ : String,
  recipe_digest_sha256~ : String,
  dimensions~ : @launch_contract.AotLaunchDimensions,
  operands~ : ArrayView[@launch_contract.AotOperand],
  workspace_bytes~ : Int64,
  workspace_alignment~ : Int64,
) -> LunaFp8ReleaseOperationV2 raise LunaCapabilityManifestError {
  if module_locator !=
    "sha256/" + entry_point.family().module_digest().sha256() + ".cubin" ||
    function_symbol.is_empty() ||
    function_symbol.length() > 255 ||
    !luna_capability_is_sha256(source_digest_sha256) ||
    !luna_capability_is_sha256(recipe_digest_sha256) ||
    operands.is_empty() ||
    workspace_bytes <= 0L ||
    workspace_alignment <= 0L ||
    (workspace_alignment & (workspace_alignment - 1L)) != 0L {
    raise Invalid(Evidence)
  }
  {
    operation,
    entry_point,
    module_locator,
    function_symbol,
    source_digest_sha256,
    recipe_digest_sha256,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    workspace_bytes,
    workspace_alignment,
  }
}

///|
pub struct LunaFp8ReleaseManifestV2 {
  priv identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv runtime_recipe_digest_sha256 : String
  priv operations : ReadOnlyArray[LunaFp8ReleaseOperationV2]
  priv canonical : Bytes
  priv digest : LunaCapabilityManifestDigest
} derive(Debug, Eq)

///|
pub struct LunaAdmittedFp8ReleaseManifestV2 {
  priv manifest : LunaFp8ReleaseManifestV2
  priv approved_source : LunaApprovedImmutableSourceIdentity
  priv key_identity : LunaSignatureKeyIdentityDigest
  priv signature_digest : LunaDetachedSignatureDigest
} derive(Debug, Eq)

///|
pub fn LunaFp8ReleaseManifestV2::model_identity(
  self : LunaFp8ReleaseManifestV2,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn LunaFp8ReleaseManifestV2::target(
  self : LunaFp8ReleaseManifestV2,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn LunaFp8ReleaseManifestV2::runtime_recipe_digest_sha256(
  self : LunaFp8ReleaseManifestV2,
) -> String {
  self.runtime_recipe_digest_sha256
}

///|
pub fn LunaFp8ReleaseManifestV2::operations(
  self : LunaFp8ReleaseManifestV2,
) -> ReadOnlyArray[LunaFp8ReleaseOperationV2] {
  self.operations
}

///|
pub fn LunaFp8ReleaseManifestV2::signing_bytes(
  self : LunaFp8ReleaseManifestV2,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaFp8ReleaseManifestV2::digest(
  self : LunaFp8ReleaseManifestV2,
) -> LunaCapabilityManifestDigest {
  self.digest
}

///|
pub fn LunaAdmittedFp8ReleaseManifestV2::manifest(
  self : LunaAdmittedFp8ReleaseManifestV2,
) -> LunaFp8ReleaseManifestV2 {
  self.manifest
}

///|
pub fn LunaAdmittedFp8ReleaseManifestV2::approved_source_identity(
  self : LunaAdmittedFp8ReleaseManifestV2,
) -> LunaApprovedImmutableSourceIdentity {
  self.approved_source
}

///|
pub fn LunaFp8ReleaseOperationV2::operation(
  self : LunaFp8ReleaseOperationV2,
) -> @plan.OperationId {
  self.operation
}

///|
pub fn LunaFp8ReleaseOperationV2::entry_point(
  self : LunaFp8ReleaseOperationV2,
) -> @catalog.AotKernelEntryPoint {
  self.entry_point
}

///|
pub fn LunaFp8ReleaseOperationV2::module_locator(
  self : LunaFp8ReleaseOperationV2,
) -> String {
  self.module_locator
}

///|
pub fn LunaFp8ReleaseOperationV2::function_symbol(
  self : LunaFp8ReleaseOperationV2,
) -> String {
  self.function_symbol
}

///|
pub fn LunaFp8ReleaseOperationV2::source_digest_sha256(
  self : LunaFp8ReleaseOperationV2,
) -> String {
  self.source_digest_sha256
}

///|
pub fn LunaFp8ReleaseOperationV2::recipe_digest_sha256(
  self : LunaFp8ReleaseOperationV2,
) -> String {
  self.recipe_digest_sha256
}

///|
pub fn LunaFp8ReleaseOperationV2::dimensions(
  self : LunaFp8ReleaseOperationV2,
) -> @launch_contract.AotLaunchDimensions {
  self.dimensions
}

///|
pub fn LunaFp8ReleaseOperationV2::operands(
  self : LunaFp8ReleaseOperationV2,
) -> ReadOnlyArray[@launch_contract.AotOperand] {
  self.operands
}

///|
pub fn LunaFp8ReleaseOperationV2::workspace_bytes(
  self : LunaFp8ReleaseOperationV2,
) -> Int64 {
  self.workspace_bytes
}

///|
pub fn LunaFp8ReleaseOperationV2::workspace_alignment(
  self : LunaFp8ReleaseOperationV2,
) -> Int64 {
  self.workspace_alignment
}
