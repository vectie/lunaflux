///|
/// One startup-approved Qwen projection/postprocess/read-only replacement.
///
/// Every value is bound to one immutable model-plan generation. The two AOT
/// modules are retained here so paged executor preparation can load each
/// module once and reuse its function across all shape-identical layer spans.
/// Prefill and decode remain separate values even for legacy bundles that map
/// both phases to the same mixed kernel.
priv struct ApprovedQwenPartitionedPrefill {
  module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  module_bytes : Bytes
  partial_symbol : String
  partial_launch : @launch_contract.AotLaunchDimensions
  merge_symbol : String
  merge_launch : @launch_contract.AotLaunchDimensions
  workspace_bytes : Int64
  workspace_alignment : Int64
} derive(Eq)

///|
pub struct ApprovedQwenFusedAttentionSpan {
  priv generation : @worker_protocol.ModelPlanGeneration
  priv target : @catalog.DeviceTarget
  priv runtime_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv ingress_includes_projection : Bool
  priv postprocess_module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv postprocess_module_bytes : Bytes
  priv postprocess_symbol : String
  priv postprocess_launch : @launch_contract.AotLaunchDimensions
  priv readonly_module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv readonly_module_bytes : Bytes
  priv readonly_symbol : String
  priv readonly_launch : @launch_contract.AotLaunchDimensions
  priv decode_module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv decode_module_bytes : Bytes
  priv decode_symbol : String
  priv decode_launch : @launch_contract.AotLaunchDimensions
  priv partitioned_prefill : ApprovedQwenPartitionedPrefill?
  priv qkv_index : Int
  priv qk_norm_index : Int
  priv rope_index : Int
  priv attention_index : Int
}

///|
pub fn ApprovedQwenFusedAttentionSpan::model_generation(
  self : ApprovedQwenFusedAttentionSpan,
) -> @worker_protocol.ModelPlanGeneration {
  self.generation
}

///|
pub fn ApprovedQwenFusedAttentionSpan::operation_indices(
  self : ApprovedQwenFusedAttentionSpan,
) -> (Int, Int, Int, Int) {
  (self.qkv_index, self.qk_norm_index, self.rope_index, self.attention_index)
}

///|
fn ApprovedQwenFusedAttentionSpan::target(
  self : ApprovedQwenFusedAttentionSpan,
) -> @catalog.DeviceTarget {
  self.target
}

///|
fn ApprovedQwenFusedAttentionSpan::same_runtime_modules(
  self : ApprovedQwenFusedAttentionSpan,
  other : ApprovedQwenFusedAttentionSpan,
) -> Bool {
  self.runtime_digest == other.runtime_digest &&
  self.ingress_includes_projection == other.ingress_includes_projection &&
  self.postprocess_module_digest == other.postprocess_module_digest &&
  self.postprocess_symbol == other.postprocess_symbol &&
  self.postprocess_launch == other.postprocess_launch &&
  self.readonly_module_digest == other.readonly_module_digest &&
  self.readonly_symbol == other.readonly_symbol &&
  self.readonly_launch == other.readonly_launch &&
  self.decode_module_digest == other.decode_module_digest &&
  self.decode_symbol == other.decode_symbol &&
  self.decode_launch == other.decode_launch &&
  self.partitioned_prefill == other.partitioned_prefill
}

///|
fn ApprovedQwenFusedAttentionSpan::ingress_includes_projection(
  self : ApprovedQwenFusedAttentionSpan,
) -> Bool {
  self.ingress_includes_projection
}

///|
fn ApprovedQwenFusedAttentionSpan::postprocess_module_bytes(
  self : ApprovedQwenFusedAttentionSpan,
) -> Bytes {
  self.postprocess_module_bytes
}

///|
fn ApprovedQwenFusedAttentionSpan::postprocess_function_symbol(
  self : ApprovedQwenFusedAttentionSpan,
) -> String {
  self.postprocess_symbol
}

///|
fn ApprovedQwenFusedAttentionSpan::postprocess_launch(
  self : ApprovedQwenFusedAttentionSpan,
) -> @launch_contract.AotLaunchDimensions {
  self.postprocess_launch
}

///|
fn ApprovedQwenFusedAttentionSpan::readonly_module_bytes(
  self : ApprovedQwenFusedAttentionSpan,
) -> Bytes {
  self.readonly_module_bytes
}

///|
fn ApprovedQwenFusedAttentionSpan::readonly_function_symbol(
  self : ApprovedQwenFusedAttentionSpan,
) -> String {
  self.readonly_symbol
}

///|
fn ApprovedQwenFusedAttentionSpan::readonly_launch(
  self : ApprovedQwenFusedAttentionSpan,
) -> @launch_contract.AotLaunchDimensions {
  self.readonly_launch
}

///|
fn ApprovedQwenFusedAttentionSpan::decode_module_bytes(
  self : ApprovedQwenFusedAttentionSpan,
) -> Bytes {
  self.decode_module_bytes
}

///|
fn ApprovedQwenFusedAttentionSpan::decode_function_symbol(
  self : ApprovedQwenFusedAttentionSpan,
) -> String {
  self.decode_symbol
}

///|
fn ApprovedQwenFusedAttentionSpan::decode_launch(
  self : ApprovedQwenFusedAttentionSpan,
) -> @launch_contract.AotLaunchDimensions {
  self.decode_launch
}

///|
fn ApprovedQwenFusedAttentionSpan::has_phase_specific_attention(
  self : ApprovedQwenFusedAttentionSpan,
) -> Bool {
  self.readonly_module_digest != self.decode_module_digest ||
  self.readonly_symbol != self.decode_symbol ||
  self.readonly_launch != self.decode_launch
}

///|
fn ApprovedQwenFusedAttentionSpan::partitioned_prefill(
  self : ApprovedQwenFusedAttentionSpan,
) -> ApprovedQwenPartitionedPrefill? {
  self.partitioned_prefill
}
