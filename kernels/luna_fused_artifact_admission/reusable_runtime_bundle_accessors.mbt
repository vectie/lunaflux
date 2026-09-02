///|
pub fn ApprovedReusableFusedRuntimeBundle::source_digest(
  self : ApprovedReusableFusedRuntimeBundle,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.source_digest
}

///|
pub fn ApprovedReusableFusedRuntimeBundle::model_content_sha256(
  self : ApprovedReusableFusedRuntimeBundle,
) -> String {
  self.model_content.as_hex()
}

///|
pub fn ApprovedReusableFusedRuntimeBundle::model_plan_sha256(
  self : ApprovedReusableFusedRuntimeBundle,
) -> String {
  self.model_plan.as_hex()
}

///|
pub fn ApprovedReusableFusedRuntimeBundle::target(
  self : ApprovedReusableFusedRuntimeBundle,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn ApprovedReusableFusedRuntimeBundle::modules(
  self : ApprovedReusableFusedRuntimeBundle,
) -> ReadOnlyArray[ApprovedReusableFusedRuntimeModule] {
  self.modules
}

///|
pub fn ApprovedReusableFusedRuntimeBundle::module_for(
  self : ApprovedReusableFusedRuntimeBundle,
  abi : ReusableFusedRuntimeKernelAbi,
) -> ApprovedReusableFusedRuntimeModule? {
  for module_ in self.modules {
    if module_.abi == abi {
      return Some(module_)
    }
  }
  None
}

///|
/// Returns only the admitted residual/RMSNorm production entry. Absence is a
/// typed result and grants no authority for another family.
pub fn ApprovedReusableFusedRuntimeBundle::residual_rmsnorm_module(
  self : ApprovedReusableFusedRuntimeBundle,
) -> ApprovedReusableFusedRuntimeModule? {
  self.module_for(ResidualRmsNormProductionV2)
}

///|
/// Returns only the admitted Qwen post-projection fusion entry. The fast QKV
/// projection remains a distinct launch; the separate readonly attention
/// accessor must also succeed before a composite can be authorized.
pub fn ApprovedReusableFusedRuntimeBundle::qwen_qknorm_rope_kv_write_module(
  self : ApprovedReusableFusedRuntimeBundle,
) -> ApprovedReusableFusedRuntimeModule? {
  self.module_for(QwenQkNormPositionedRopePagedKvWriteProductionV2)
}

///|
/// Returns only the admitted Qwen cache-read-only attention entry. It cannot
/// be substituted by the Qwen ingress module or inferred from its presence.
pub fn ApprovedReusableFusedRuntimeBundle::qwen_paged_attention_readonly_module(
  self : ApprovedReusableFusedRuntimeBundle,
) -> ApprovedReusableFusedRuntimeModule? {
  match self.module_for(QwenPagedAttentionReadOnlyDecodeSplitProductionV3) {
    Some(module_) => Some(module_)
    None => self.module_for(QwenPagedAttentionReadOnlyProductionV2)
  }
}

///|
/// Returns the dedicated compiler-generated prefill entry when present.
/// Older V3 bundles remain executable by using their mixed readonly kernel for
/// both phases; a bundle containing both V2 and V3 keeps the V3 entry reserved
/// for decode.
pub fn ApprovedReusableFusedRuntimeBundle::qwen_paged_attention_prefill_module(
  self : ApprovedReusableFusedRuntimeBundle,
) -> ApprovedReusableFusedRuntimeModule? {
  match self.module_for(QwenPagedAttentionReadOnlyProductionV2) {
    Some(module_) => Some(module_)
    None => self.module_for(QwenPagedAttentionReadOnlyDecodeSplitProductionV3)
  }
}

///|
/// Returns the decode entry, falling back to an older mixed V2 module only
/// when the bundle predates phase-specific attention compilation.
pub fn ApprovedReusableFusedRuntimeBundle::qwen_paged_attention_decode_module(
  self : ApprovedReusableFusedRuntimeBundle,
) -> ApprovedReusableFusedRuntimeModule? {
  match self.module_for(QwenPagedAttentionReadOnlyDecodeSplitProductionV3) {
    Some(module_) => Some(module_)
    None => self.module_for(QwenPagedAttentionReadOnlyProductionV2)
  }
}

///|
/// Returns the phase-specialized readonly module only when it also carries
/// the stable decode split-K partial and merge entries.
pub fn ApprovedReusableFusedRuntimeBundle::qwen_paged_attention_decode_split_module(
  self : ApprovedReusableFusedRuntimeBundle,
) -> ApprovedReusableFusedRuntimeModule? {
  self.module_for(QwenPagedAttentionReadOnlyDecodeSplitProductionV3)
}

///|
pub fn ApprovedReusableFusedRuntimeBundle::residual_rmsnorm_operation_indices(
  self : ApprovedReusableFusedRuntimeBundle,
) -> (Int, Int)? {
  match self.residual_rmsnorm_module() {
    Some(module_) =>
      Some((module_.first_operation, module_.first_operation + 1))
    None => None
  }
}

///|
pub fn ApprovedReusableFusedRuntimeBundle::qwen_qknorm_rope_kv_write_operation_indices(
  self : ApprovedReusableFusedRuntimeBundle,
) -> (Int, Int, Int)? {
  match self.qwen_qknorm_rope_kv_write_module() {
    Some(module_) =>
      Some(
        (
          module_.first_operation,
          module_.first_operation + 1,
          module_.first_operation + 2,
        ),
      )
    None => None
  }
}

///|
pub fn ApprovedReusableFusedRuntimeBundle::qwen_paged_attention_readonly_operation_index(
  self : ApprovedReusableFusedRuntimeBundle,
) -> Int? {
  match self.qwen_paged_attention_readonly_module() {
    Some(module_) => Some(module_.first_operation)
    None => None
  }
}

///|
pub fn ApprovedReusableFusedRuntimeModule::kernel_abi(
  self : ApprovedReusableFusedRuntimeModule,
) -> ReusableFusedRuntimeKernelAbi {
  self.abi
}

///|
pub fn ApprovedReusableFusedRuntimeModule::operation_span(
  self : ApprovedReusableFusedRuntimeModule,
) -> (Int, Int) {
  (self.first_operation, self.operation_count)
}

///|
pub fn ApprovedReusableFusedRuntimeModule::first_operation_index(
  self : ApprovedReusableFusedRuntimeModule,
) -> Int {
  self.first_operation
}

///|
pub fn ApprovedReusableFusedRuntimeModule::operation_count(
  self : ApprovedReusableFusedRuntimeModule,
) -> Int {
  self.operation_count
}

///|
pub fn ApprovedReusableFusedRuntimeModule::function_symbol(
  self : ApprovedReusableFusedRuntimeModule,
) -> String {
  self.symbol
}

///|
pub fn ApprovedReusableFusedRuntimeModule::launch_dimensions(
  self : ApprovedReusableFusedRuntimeModule,
) -> @launch_contract.AotLaunchDimensions {
  self.launch
}

///|
pub fn ApprovedReusableFusedRuntimeModule::module_digest(
  self : ApprovedReusableFusedRuntimeModule,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.module_digest
}

///|
pub fn ApprovedReusableFusedRuntimeModule::module_bytes(
  self : ApprovedReusableFusedRuntimeModule,
) -> Bytes {
  self.module_bytes
}
