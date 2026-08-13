///|
/// Versioned recipe understood by the isolated device-worker bootstrap builder.
pub(all) enum WorkerBootstrapRecipe {
  DenseLlamaPagedAotV1
} derive(Debug, Eq)

///|
/// SHA-256 identity of the exact canonical bootstrap-source bytes excluding the
/// trailing digest field which contains the same value in raw form.
pub struct WorkerBootstrapSourceDigest(String) derive(Debug, Eq)

///|
/// Two lexically validated absolute filesystem-root locators.
///
/// Construction proves only canonical path syntax. Before filesystem use, a
/// deployment boundary must independently admit these locators as stable,
/// read-only roots. Receiving or decoding this value grants no such authority.
pub struct WorkerBootstrapRoots {
  priv model_root : String
  priv kernel_root : String
} derive(Eq)

///|
/// Digest-pinned model metadata and weight locators below the model root.
pub struct WorkerBootstrapModelSource {
  priv config_locator : String
  priv weights_locator : String
  priv config_digest : String
} derive(Eq)

///|
/// Digest-pinned execution-manifest locator below the kernel root.
pub struct WorkerBootstrapKernelSource {
  priv manifest_locator : String
  priv manifest_digest : String
} derive(Eq)

///|
/// Capacity-independent device target and physical KV-page geometry.
pub struct WorkerBootstrapExecution {
  priv compute_major : Int
  priv compute_minor : Int
  priv supports_bf16 : Bool
  priv supports_cublas_lt : Bool
  priv tokens_per_page : Int
  priv total_page_count : Int
} derive(Debug, Eq)

///|
/// Exact startup ceilings consumed by focused model, artifact, and device-plan
/// readers. Parser depths, transfer chunks, alignments, and profile identity are
/// fixed by `DenseLlamaPagedAotV1` rather than repeated here.
pub struct WorkerBootstrapCeilings {
  priv max_model_config_bytes : Int
  priv max_weight_file_bytes : Int64
  priv max_weight_arena_bytes : Int64
  priv max_activation_arena_bytes : Int64
  priv max_kv_arena_bytes : Int64
  priv max_execution_manifest_bytes : Int64
  priv max_module_bytes : Int64
  priv max_total_module_bytes : Int64
} derive(Debug, Eq)

///|
/// Immutable exact canonical source snapshot retained across worker restarts.
/// It contains locators, digests, and scalar ceilings but no model or module
/// bytes and exposes no mutable storage alias.
pub struct EncodedBootstrapSource {
  priv storage : FixedArray[Byte]
  priv length : Int
  priv digest : WorkerBootstrapSourceDigest
  priv roots : WorkerBootstrapRoots
  priv model : WorkerBootstrapModelSource
  priv kernel : WorkerBootstrapKernelSource
  priv execution : WorkerBootstrapExecution
  priv ceilings : WorkerBootstrapCeilings
}

///|
pub fn WorkerBootstrapSourceDigest::from_sha256(
  value : String,
) -> WorkerBootstrapSourceDigest raise WorkerWireError {
  if !bootstrap_source_is_sha256(value) {
    raise InvalidBootstrapSource(Digest)
  }
  WorkerBootstrapSourceDigest(value)
}

///|
pub fn WorkerBootstrapSourceDigest::as_hex(
  self : WorkerBootstrapSourceDigest,
) -> String {
  self.0
}

///|
pub fn EncodedBootstrapSource::length(self : EncodedBootstrapSource) -> Int {
  self.length
}

///|
pub fn EncodedBootstrapSource::capacity(self : EncodedBootstrapSource) -> Int {
  self.storage.length()
}

///|
pub fn EncodedBootstrapSource::digest(
  self : EncodedBootstrapSource,
) -> WorkerBootstrapSourceDigest {
  self.digest
}

///|
pub fn EncodedBootstrapSource::recipe(
  _self : EncodedBootstrapSource,
) -> WorkerBootstrapRecipe {
  DenseLlamaPagedAotV1
}

///|
pub fn EncodedBootstrapSource::roots(
  self : EncodedBootstrapSource,
) -> WorkerBootstrapRoots {
  self.roots
}

///|
pub fn EncodedBootstrapSource::model(
  self : EncodedBootstrapSource,
) -> WorkerBootstrapModelSource {
  self.model
}

///|
pub fn EncodedBootstrapSource::kernel(
  self : EncodedBootstrapSource,
) -> WorkerBootstrapKernelSource {
  self.kernel
}

///|
pub fn EncodedBootstrapSource::execution(
  self : EncodedBootstrapSource,
) -> WorkerBootstrapExecution {
  self.execution
}

///|
pub fn EncodedBootstrapSource::ceilings(
  self : EncodedBootstrapSource,
) -> WorkerBootstrapCeilings {
  self.ceilings
}

///|
pub fn WorkerBootstrapRoots::model_root(self : WorkerBootstrapRoots) -> String {
  self.model_root
}

///|
pub fn WorkerBootstrapRoots::kernel_root(self : WorkerBootstrapRoots) -> String {
  self.kernel_root
}

///|
pub fn WorkerBootstrapModelSource::config_locator(
  self : WorkerBootstrapModelSource,
) -> String {
  self.config_locator
}

///|
pub fn WorkerBootstrapModelSource::weights_locator(
  self : WorkerBootstrapModelSource,
) -> String {
  self.weights_locator
}

///|
pub fn WorkerBootstrapModelSource::config_digest(
  self : WorkerBootstrapModelSource,
) -> String {
  self.config_digest
}

///|
pub fn WorkerBootstrapKernelSource::manifest_locator(
  self : WorkerBootstrapKernelSource,
) -> String {
  self.manifest_locator
}

///|
pub fn WorkerBootstrapKernelSource::manifest_digest(
  self : WorkerBootstrapKernelSource,
) -> String {
  self.manifest_digest
}

///|
pub fn WorkerBootstrapExecution::compute_major(
  self : WorkerBootstrapExecution,
) -> Int {
  self.compute_major
}

///|
pub fn WorkerBootstrapExecution::compute_minor(
  self : WorkerBootstrapExecution,
) -> Int {
  self.compute_minor
}

///|
pub fn WorkerBootstrapExecution::supports_bf16(
  self : WorkerBootstrapExecution,
) -> Bool {
  self.supports_bf16
}

///|
pub fn WorkerBootstrapExecution::supports_cublas_lt(
  self : WorkerBootstrapExecution,
) -> Bool {
  self.supports_cublas_lt
}

///|
pub fn WorkerBootstrapExecution::tokens_per_page(
  self : WorkerBootstrapExecution,
) -> Int {
  self.tokens_per_page
}

///|
pub fn WorkerBootstrapExecution::total_page_count(
  self : WorkerBootstrapExecution,
) -> Int {
  self.total_page_count
}

///|
pub fn WorkerBootstrapCeilings::max_model_config_bytes(
  self : WorkerBootstrapCeilings,
) -> Int {
  self.max_model_config_bytes
}

///|
pub fn WorkerBootstrapCeilings::max_weight_file_bytes(
  self : WorkerBootstrapCeilings,
) -> Int64 {
  self.max_weight_file_bytes
}

///|
pub fn WorkerBootstrapCeilings::max_weight_arena_bytes(
  self : WorkerBootstrapCeilings,
) -> Int64 {
  self.max_weight_arena_bytes
}

///|
pub fn WorkerBootstrapCeilings::max_activation_arena_bytes(
  self : WorkerBootstrapCeilings,
) -> Int64 {
  self.max_activation_arena_bytes
}

///|
pub fn WorkerBootstrapCeilings::max_kv_arena_bytes(
  self : WorkerBootstrapCeilings,
) -> Int64 {
  self.max_kv_arena_bytes
}

///|
pub fn WorkerBootstrapCeilings::max_execution_manifest_bytes(
  self : WorkerBootstrapCeilings,
) -> Int64 {
  self.max_execution_manifest_bytes
}

///|
pub fn WorkerBootstrapCeilings::max_module_bytes(
  self : WorkerBootstrapCeilings,
) -> Int64 {
  self.max_module_bytes
}

///|
pub fn WorkerBootstrapCeilings::max_total_module_bytes(
  self : WorkerBootstrapCeilings,
) -> Int64 {
  self.max_total_module_bytes
}
