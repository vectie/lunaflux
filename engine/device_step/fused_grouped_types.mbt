///|
/// Exact approved two-operation replacement for one model-plan generation.
/// It is inert until consumed by paged-executor preparation.
pub struct ApprovedFusedResidualRmsNormSpan {
  priv generation : @worker_protocol.ModelPlanGeneration
  priv manifest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv catalog : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv target : @catalog.DeviceTarget
  priv module_bytes : Bytes
  priv function_symbol : String
  priv launch : @launch_contract.AotLaunchDimensions
  priv abi : @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi
  priv dispatch_canary_per_token : UInt
  priv residual_index : Int
  priv norm_index : Int
}

///|
/// Typed non-dispatchable result for the approved QKV family. The existing
/// paged-attention launch combines cache write and attention, so a read-only
/// attention split is required before exact atomic replacement is possible.
pub struct FusedQkvReadOnlyAttentionSplitRequirement {
  priv generation : @worker_protocol.ModelPlanGeneration
  priv manifest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv qkv_index : Int
  priv rope_index : Int
  priv attention_index : Int
}

///|
pub fn ApprovedFusedResidualRmsNormSpan::model_generation(
  self : ApprovedFusedResidualRmsNormSpan,
) -> @worker_protocol.ModelPlanGeneration {
  self.generation
}

///|
pub fn ApprovedFusedResidualRmsNormSpan::manifest_digest(
  self : ApprovedFusedResidualRmsNormSpan,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.manifest
}

///|
pub fn ApprovedFusedResidualRmsNormSpan::catalog_digest(
  self : ApprovedFusedResidualRmsNormSpan,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.catalog
}

///|
pub fn ApprovedFusedResidualRmsNormSpan::operation_indices(
  self : ApprovedFusedResidualRmsNormSpan,
) -> (Int, Int) {
  (self.residual_index, self.norm_index)
}

///|
fn ApprovedFusedResidualRmsNormSpan::target(
  self : ApprovedFusedResidualRmsNormSpan,
) -> @catalog.DeviceTarget {
  self.target
}

///|
fn ApprovedFusedResidualRmsNormSpan::approved_module_bytes(
  self : ApprovedFusedResidualRmsNormSpan,
) -> Bytes {
  self.module_bytes
}

///|
fn ApprovedFusedResidualRmsNormSpan::approved_function_symbol(
  self : ApprovedFusedResidualRmsNormSpan,
) -> String {
  self.function_symbol
}

///|
fn ApprovedFusedResidualRmsNormSpan::approved_launch(
  self : ApprovedFusedResidualRmsNormSpan,
) -> @launch_contract.AotLaunchDimensions {
  self.launch
}

///|
fn ApprovedFusedResidualRmsNormSpan::dispatch_canary_per_token(
  self : ApprovedFusedResidualRmsNormSpan,
) -> UInt {
  self.dispatch_canary_per_token
}

///|
fn ApprovedFusedResidualRmsNormSpan::kernel_abi(
  self : ApprovedFusedResidualRmsNormSpan,
) -> @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi {
  self.abi
}

///|
fn ApprovedFusedResidualRmsNormSpan::uses_qualification_canary(
  self : ApprovedFusedResidualRmsNormSpan,
) -> Bool {
  self.abi == ResidualRmsNormAbiV1
}

///|
pub fn FusedQkvReadOnlyAttentionSplitRequirement::operation_indices(
  self : FusedQkvReadOnlyAttentionSplitRequirement,
) -> (Int, Int, Int) {
  (self.qkv_index, self.rope_index, self.attention_index)
}

///|
pub fn FusedQkvReadOnlyAttentionSplitRequirement::model_generation(
  self : FusedQkvReadOnlyAttentionSplitRequirement,
) -> @worker_protocol.ModelPlanGeneration {
  self.generation
}

///|
pub fn FusedQkvReadOnlyAttentionSplitRequirement::manifest_digest(
  self : FusedQkvReadOnlyAttentionSplitRequirement,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.manifest
}
