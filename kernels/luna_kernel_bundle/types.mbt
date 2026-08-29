///|
/// Caller-constructible report of two offline compiler observations.
///
/// Construction validates digest syntax only. This is inert assembly input,
/// never external approval, executable authority, promotion, or readiness.
pub struct LunaDeterministicCompileReceipt {
  priv source_digest_sha256 : String
  priv recipe_digest_sha256 : String
  priv toolchain_digest_sha256 : String
  priv driver_identity_sha256 : String
  priv first_artifact_digest_sha256 : String
  priv second_artifact_digest_sha256 : String
} derive(Debug, Eq)

///|
pub fn LunaDeterministicCompileReceipt::from_offline_builder(
  source_digest_sha256~ : String,
  recipe_digest_sha256~ : String,
  toolchain_digest_sha256~ : String,
  driver_identity_sha256~ : String,
  first_artifact_digest_sha256~ : String,
  second_artifact_digest_sha256~ : String,
) -> LunaDeterministicCompileReceipt raise LunaKernelBundleError {
  for
    digest in [
      source_digest_sha256, recipe_digest_sha256, toolchain_digest_sha256, driver_identity_sha256,
      first_artifact_digest_sha256, second_artifact_digest_sha256,
    ] {
    if !kernel_bundle_is_sha256(digest) {
      raise Invalid(Digest)
    }
  }
  {
    source_digest_sha256,
    recipe_digest_sha256,
    toolchain_digest_sha256,
    driver_identity_sha256,
    first_artifact_digest_sha256,
    second_artifact_digest_sha256,
  }
}

///|
/// Bounds for one exact bundle. Manifest limits are delegated to the existing
/// authenticated schema-v2 producer.
pub struct LunaKernelBundleLimits {
  priv max_module_bytes : Int64
  priv max_total_module_bytes : Int64
  priv max_modules : Int
  priv max_operations : Int
  priv manifest : @full_graph_manifest.FullGraphManifestLimits
}

///|
pub fn LunaKernelBundleLimits::new(
  max_module_bytes~ : Int64,
  max_total_module_bytes~ : Int64,
  max_modules~ : Int,
  max_operations~ : Int,
  manifest~ : @full_graph_manifest.FullGraphManifestLimits,
) -> LunaKernelBundleLimits raise LunaKernelBundleError {
  if max_module_bytes <= 0L ||
    max_total_module_bytes < max_module_bytes ||
    max_modules <= 0 ||
    max_operations <= 0 {
    raise Invalid(Limits)
  }
  {
    max_module_bytes,
    max_total_module_bytes,
    max_modules,
    max_operations,
    manifest,
  }
}

///|
/// One content-addressed immutable CUBIN in the release kernel inventory.
pub struct LunaKernelBundleModule {
  priv digest : @catalog.AotArtifactDigest
  priv relative_path : String
  priv module_bytes : Bytes
}

///|
pub fn LunaKernelBundleModule::digest(
  self : LunaKernelBundleModule,
) -> @catalog.AotArtifactDigest {
  self.digest
}

///|
pub fn LunaKernelBundleModule::relative_path(
  self : LunaKernelBundleModule,
) -> String {
  self.relative_path
}

///|
pub fn LunaKernelBundleModule::module_bytes(
  self : LunaKernelBundleModule,
) -> Bytes {
  self.module_bytes
}

///|
/// Inert contract-bound release material from one typed lowering family.
///
/// This value is caller-constructible through a compile receipt and therefore
/// cannot authorize module loading, execution, promotion, or readiness. Those
/// require separately approved immutable release evidence.
pub struct LunaKernelCompiledOperation {
  priv artifact : @full_graph_manifest.FullGraphAotOperationArtifact
  priv module_item : LunaKernelBundleModule
  priv target : @catalog.DeviceTarget
  priv source_digest_sha256 : String
  priv recipe_digest_sha256 : String
  priv toolchain_digest_sha256 : String
  priv driver_identity_sha256 : String
}

///|
pub fn LunaKernelCompiledOperation::operation_id(
  self : LunaKernelCompiledOperation,
) -> @plan.OperationId {
  self.artifact.operation_id()
}

///|
pub fn LunaKernelCompiledOperation::module_digest(
  self : LunaKernelCompiledOperation,
) -> @catalog.AotArtifactDigest {
  self.module_item.digest
}

///|
pub fn LunaKernelCompiledOperation::device_target(
  self : LunaKernelCompiledOperation,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn LunaKernelCompiledOperation::source_digest_sha256(
  self : LunaKernelCompiledOperation,
) -> String {
  self.source_digest_sha256
}

///|
pub fn LunaKernelCompiledOperation::recipe_digest_sha256(
  self : LunaKernelCompiledOperation,
) -> String {
  self.recipe_digest_sha256
}

///|
pub fn LunaKernelCompiledOperation::toolchain_digest_sha256(
  self : LunaKernelCompiledOperation,
) -> String {
  self.toolchain_digest_sha256
}

///|
pub fn LunaKernelCompiledOperation::driver_identity_sha256(
  self : LunaKernelCompiledOperation,
) -> String {
  self.driver_identity_sha256
}

///|
/// Exact manifest bytes and exact immutable module inventory for release
/// assembly. It contains no compiler, filesystem, process, PTX, or JIT input.
pub struct LunaKernelBundleInput {
  priv manifest : @full_graph_manifest.FullGraphExecutionManifestInput
  priv modules : ReadOnlyArray[LunaKernelBundleModule]
  priv operations : ReadOnlyArray[LunaKernelCompiledOperation]
  priv total_module_bytes : Int64
}

///|
pub fn LunaKernelBundleInput::execution_manifest(
  self : LunaKernelBundleInput,
) -> @full_graph_manifest.FullGraphExecutionManifestInput {
  self.manifest
}

///|
pub fn LunaKernelBundleInput::modules(
  self : LunaKernelBundleInput,
) -> ReadOnlyArray[LunaKernelBundleModule] {
  self.modules
}

///|
pub fn LunaKernelBundleInput::operations(
  self : LunaKernelBundleInput,
) -> ReadOnlyArray[LunaKernelCompiledOperation] {
  self.operations
}

///|
pub fn LunaKernelBundleInput::total_module_bytes(
  self : LunaKernelBundleInput,
) -> Int64 {
  self.total_module_bytes
}
