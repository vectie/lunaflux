///|
/// Versioned recipe understood by the isolated device-worker bootstrap builder.
pub(all) enum WorkerBootstrapRecipe {
  DenseLlamaPagedAotV2
  DenseLlamaPagedAotV3
  DenseLlamaTensorParallelPagedAotV5
  DenseLlamaI8PagedAotV6
  DenseMistralBf16PagedAotV7
  DenseLlamaTensorParallelPagedAotApprovedV8
  DenseLlamaFp8ReusablePagedAotApprovedV9
  DenseLlamaBf16PagedAotExtensionsV10
  DenseMistralBf16PagedAotExtensionsV11
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
  priv luna_approval : WorkerBootstrapLunaApproval?
  priv fused_runtime : WorkerBootstrapFusedRuntimeSource?
  priv sampling_runtime : WorkerBootstrapSamplingRuntime
} derive(Eq)

///|
/// Startup-selected sampling placement. The host mode preserves full
/// stochastic support; embedded CUDA greedy is valid only for greedy requests.
pub(all) enum WorkerBootstrapSamplingRuntime {
  HostSamplingRuntime
  EmbeddedCudaGreedyV1
} derive(Debug, Eq)

///|
/// Digest-pinned optional fused-runtime descriptor below the inherited kernel
/// root. This is inert wire evidence: it grants neither filesystem nor module
/// loading authority.
pub struct WorkerBootstrapFusedRuntimeSource {
  priv locator : String
  priv digest : String
} derive(Debug, Eq)

///|
/// Inert scalar copy of an independently verified deployment approval.
/// Construction validates canonical identities but performs no signature check.
pub struct WorkerBootstrapLunaApproval {
  priv manifest_digest : String
  priv key_identity : String
  priv signature_digest : String
  priv approved_source : String
} derive(Debug, Eq)

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
/// fixed by `DenseLlamaPagedAotV2` rather than repeated here.
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
  priv recipe : WorkerBootstrapRecipe
  priv tensor_parallel_policy : TensorParallelBootstrapPolicy?
  priv i8_model_source : WorkerI8BootstrapModelSource?
  priv i8_execution : WorkerI8BootstrapExecution?
  priv mistral_model_source : WorkerMistralBootstrapModelSource?
  priv fp8_model_source : WorkerFp8BootstrapModelSource?
  priv fp8_execution : WorkerFp8BootstrapExecution?
}

///|

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
  self : EncodedBootstrapSource,
) -> WorkerBootstrapRecipe {
  self.recipe
}

///|
/// Returns true only for a canonical tensor-parallel recipe/approval pair.
/// Legacy v5 remains explicitly unapproved; production v8 requires an
/// approval whose immutable-source identity is the exact manifest digest.
pub fn EncodedBootstrapSource::has_valid_tensor_parallel_recipe_binding(
  self : EncodedBootstrapSource,
) -> Bool {
  match (self.recipe, self.kernel.luna_approval) {
    (DenseLlamaTensorParallelPagedAotV5, None) => true
    (DenseLlamaTensorParallelPagedAotApprovedV8, Some(approval)) =>
      approval.manifest_digest == self.kernel.manifest_digest &&
      approval.approved_source == self.kernel.manifest_digest
    _ => false
  }
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
/// Returns typed production-I8 source evidence only for recipe v6.
pub fn EncodedBootstrapSource::i8_model_source(
  self : EncodedBootstrapSource,
) -> WorkerI8BootstrapModelSource? {
  self.i8_model_source
}

///|
/// Returns typed production-I8 device facts only for recipe v6.
pub fn EncodedBootstrapSource::i8_execution(
  self : EncodedBootstrapSource,
) -> WorkerI8BootstrapExecution? {
  self.i8_execution
}

///|
/// Returns exact Mistral source evidence only for recipe v7.
pub fn EncodedBootstrapSource::mistral_model_source(
  self : EncodedBootstrapSource,
) -> WorkerMistralBootstrapModelSource? {
  self.mistral_model_source
}

///|
pub fn EncodedBootstrapSource::fp8_model_source(
  self : EncodedBootstrapSource,
) -> WorkerFp8BootstrapModelSource? {
  self.fp8_model_source
}

///|
pub fn EncodedBootstrapSource::fp8_execution(
  self : EncodedBootstrapSource,
) -> WorkerFp8BootstrapExecution? {
  self.fp8_execution
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
pub fn WorkerBootstrapKernelSource::luna_approval(
  self : WorkerBootstrapKernelSource,
) -> WorkerBootstrapLunaApproval? {
  self.luna_approval
}

///|
pub fn WorkerBootstrapKernelSource::fused_runtime_source(
  self : WorkerBootstrapKernelSource,
) -> WorkerBootstrapFusedRuntimeSource? {
  self.fused_runtime
}

///|
pub fn WorkerBootstrapKernelSource::sampling_runtime(
  self : WorkerBootstrapKernelSource,
) -> WorkerBootstrapSamplingRuntime {
  self.sampling_runtime
}

///|
pub fn WorkerBootstrapFusedRuntimeSource::locator(
  self : WorkerBootstrapFusedRuntimeSource,
) -> String {
  self.locator
}

///|
pub fn WorkerBootstrapFusedRuntimeSource::digest_sha256(
  self : WorkerBootstrapFusedRuntimeSource,
) -> String {
  self.digest
}

///|
pub fn WorkerBootstrapLunaApproval::manifest_digest(
  self : WorkerBootstrapLunaApproval,
) -> String {
  self.manifest_digest
}

///|
pub fn WorkerBootstrapLunaApproval::key_identity(
  self : WorkerBootstrapLunaApproval,
) -> String {
  self.key_identity
}

///|
pub fn WorkerBootstrapLunaApproval::signature_digest(
  self : WorkerBootstrapLunaApproval,
) -> String {
  self.signature_digest
}

///|
pub fn WorkerBootstrapLunaApproval::approved_source(
  self : WorkerBootstrapLunaApproval,
) -> String {
  self.approved_source
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
