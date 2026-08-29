///|
/// Exact executable two-launch replacement for QKV projection, positioned
/// RoPE, KV write, and readonly paged attention. It is produced only after the
/// live device joins the authenticated production V2 artifact pair.
pub struct ApprovedFusedQkvReadOnlyAttentionSpan {
  priv generation : @worker_protocol.ModelPlanGeneration
  priv target : @catalog.DeviceTarget
  priv qkv_module_bytes : Bytes
  priv qkv_symbol : String
  priv qkv_launch : @launch_contract.AotLaunchDimensions
  priv readonly_module_bytes : Bytes
  priv readonly_symbol : String
  priv readonly_launch : @launch_contract.AotLaunchDimensions
  priv qkv_index : Int
  priv rope_index : Int
  priv attention_index : Int
}

///|
pub fn ApprovedFusedQkvReadOnlyAttentionSpan::model_generation(
  self : ApprovedFusedQkvReadOnlyAttentionSpan,
) -> @worker_protocol.ModelPlanGeneration {
  self.generation
}

///|
pub fn ApprovedFusedQkvReadOnlyAttentionSpan::operation_indices(
  self : ApprovedFusedQkvReadOnlyAttentionSpan,
) -> (Int, Int, Int) {
  (self.qkv_index, self.rope_index, self.attention_index)
}

///|
fn ApprovedFusedQkvReadOnlyAttentionSpan::target(
  self : ApprovedFusedQkvReadOnlyAttentionSpan,
) -> @catalog.DeviceTarget {
  self.target
}

///|
fn ApprovedFusedQkvReadOnlyAttentionSpan::qkv_module_bytes(
  self : ApprovedFusedQkvReadOnlyAttentionSpan,
) -> Bytes {
  self.qkv_module_bytes
}

///|
fn ApprovedFusedQkvReadOnlyAttentionSpan::qkv_function_symbol(
  self : ApprovedFusedQkvReadOnlyAttentionSpan,
) -> String {
  self.qkv_symbol
}

///|
fn ApprovedFusedQkvReadOnlyAttentionSpan::qkv_launch(
  self : ApprovedFusedQkvReadOnlyAttentionSpan,
) -> @launch_contract.AotLaunchDimensions {
  self.qkv_launch
}

///|
fn ApprovedFusedQkvReadOnlyAttentionSpan::readonly_module_bytes(
  self : ApprovedFusedQkvReadOnlyAttentionSpan,
) -> Bytes {
  self.readonly_module_bytes
}

///|
fn ApprovedFusedQkvReadOnlyAttentionSpan::readonly_function_symbol(
  self : ApprovedFusedQkvReadOnlyAttentionSpan,
) -> String {
  self.readonly_symbol
}

///|
fn ApprovedFusedQkvReadOnlyAttentionSpan::readonly_launch(
  self : ApprovedFusedQkvReadOnlyAttentionSpan,
) -> @launch_contract.AotLaunchDimensions {
  self.readonly_launch
}
