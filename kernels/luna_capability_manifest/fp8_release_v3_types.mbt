///|
/// Externally reviewed reusable paged FP8 release. Unlike v2, its recipe
/// digest names a profile envelope and never a concrete row shape.
pub struct LunaFp8ReusablePagedReleaseManifestV3 {
  priv identity : @spec.ModelIdentity
  priv target : @catalog.DeviceTarget
  priv reusable_recipe_digest_sha256 : String
  priv profile : @launch_contract.PagedKernelProfile
  priv kv_layout_version : Int
  priv tokens_per_page : Int
  priv operations : ReadOnlyArray[LunaFp8ReleaseOperationV2]
  priv canonical : Bytes
  priv digest : LunaCapabilityManifestDigest
} derive(Debug, Eq)

///|
pub struct LunaAdmittedFp8ReusablePagedReleaseManifestV3 {
  priv manifest : LunaFp8ReusablePagedReleaseManifestV3
  priv approved_source : LunaApprovedImmutableSourceIdentity
  priv key_identity : LunaSignatureKeyIdentityDigest
  priv signature_digest : LunaDetachedSignatureDigest
} derive(Debug, Eq)

///|
pub fn LunaFp8ReusablePagedReleaseManifestV3::model_identity(
  self : LunaFp8ReusablePagedReleaseManifestV3,
) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn LunaFp8ReusablePagedReleaseManifestV3::target(
  self : LunaFp8ReusablePagedReleaseManifestV3,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn LunaFp8ReusablePagedReleaseManifestV3::reusable_recipe_digest_sha256(
  self : LunaFp8ReusablePagedReleaseManifestV3,
) -> String {
  self.reusable_recipe_digest_sha256
}

///|
pub fn LunaFp8ReusablePagedReleaseManifestV3::profile(
  self : LunaFp8ReusablePagedReleaseManifestV3,
) -> @launch_contract.PagedKernelProfile {
  self.profile
}

///|
pub fn LunaFp8ReusablePagedReleaseManifestV3::kv_layout_version(
  self : LunaFp8ReusablePagedReleaseManifestV3,
) -> Int {
  self.kv_layout_version
}

///|
pub fn LunaFp8ReusablePagedReleaseManifestV3::tokens_per_page(
  self : LunaFp8ReusablePagedReleaseManifestV3,
) -> Int {
  self.tokens_per_page
}

///|
pub fn LunaFp8ReusablePagedReleaseManifestV3::operations(
  self : LunaFp8ReusablePagedReleaseManifestV3,
) -> ReadOnlyArray[LunaFp8ReleaseOperationV2] {
  self.operations
}

///|
pub fn LunaFp8ReusablePagedReleaseManifestV3::signing_bytes(
  self : LunaFp8ReusablePagedReleaseManifestV3,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaFp8ReusablePagedReleaseManifestV3::digest(
  self : LunaFp8ReusablePagedReleaseManifestV3,
) -> LunaCapabilityManifestDigest {
  self.digest
}

///|
pub fn LunaAdmittedFp8ReusablePagedReleaseManifestV3::manifest(
  self : LunaAdmittedFp8ReusablePagedReleaseManifestV3,
) -> LunaFp8ReusablePagedReleaseManifestV3 {
  self.manifest
}

///|
pub fn LunaAdmittedFp8ReusablePagedReleaseManifestV3::approved_source_identity(
  self : LunaAdmittedFp8ReusablePagedReleaseManifestV3,
) -> LunaApprovedImmutableSourceIdentity {
  self.approved_source
}

///|
pub fn LunaAdmittedFp8ReusablePagedReleaseManifestV3::approved_key_identity(
  self : LunaAdmittedFp8ReusablePagedReleaseManifestV3,
) -> LunaSignatureKeyIdentityDigest {
  self.key_identity
}

///|
pub fn LunaAdmittedFp8ReusablePagedReleaseManifestV3::detached_signature_digest(
  self : LunaAdmittedFp8ReusablePagedReleaseManifestV3,
) -> LunaDetachedSignatureDigest {
  self.signature_digest
}
