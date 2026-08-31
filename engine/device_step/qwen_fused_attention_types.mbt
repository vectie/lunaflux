///|
/// One startup-approved Qwen attention ingress/read-only replacement.
///
/// Every value is bound to one immutable model-plan generation. The two AOT
/// modules are retained here so paged executor preparation can load each
/// module once and reuse its function across all shape-identical layer spans.
pub struct ApprovedQwenFusedAttentionSpan {
  priv generation : @worker_protocol.ModelPlanGeneration
  priv target : @catalog.DeviceTarget
  priv runtime_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv ingress_module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv ingress_module_bytes : Bytes
  priv ingress_symbol : String
  priv ingress_launch : @launch_contract.AotLaunchDimensions
  priv readonly_module_digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv readonly_module_bytes : Bytes
  priv readonly_symbol : String
  priv readonly_launch : @launch_contract.AotLaunchDimensions
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
  self.ingress_module_digest == other.ingress_module_digest &&
  self.ingress_symbol == other.ingress_symbol &&
  self.ingress_launch == other.ingress_launch &&
  self.readonly_module_digest == other.readonly_module_digest &&
  self.readonly_symbol == other.readonly_symbol &&
  self.readonly_launch == other.readonly_launch
}

///|
fn ApprovedQwenFusedAttentionSpan::ingress_module_bytes(
  self : ApprovedQwenFusedAttentionSpan,
) -> Bytes {
  self.ingress_module_bytes
}

///|
fn ApprovedQwenFusedAttentionSpan::ingress_function_symbol(
  self : ApprovedQwenFusedAttentionSpan,
) -> String {
  self.ingress_symbol
}

///|
fn ApprovedQwenFusedAttentionSpan::ingress_launch(
  self : ApprovedQwenFusedAttentionSpan,
) -> @launch_contract.AotLaunchDimensions {
  self.ingress_launch
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
