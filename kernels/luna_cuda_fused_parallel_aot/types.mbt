///|
pub(all) enum FusedParallelFamily {
  QkvPositionedRopePagedKvWrite
  ResidualRmsNorm
} derive(Debug, Eq)

///|
pub(all) enum FusedParallelNumericalPolicy {
  QkvRopeKvWriteOrderedF32TranscendentalToleranceV1
  ResidualRmsNormBlock128F32ReductionToleranceV1
} derive(Debug, Eq)

///|
pub fn FusedParallelNumericalPolicy::name(
  self : FusedParallelNumericalPolicy,
) -> String {
  match self {
    QkvRopeKvWriteOrderedF32TranscendentalToleranceV1 =>
      "fused-qkv-positioned-rope-paged-kvwrite-ordered-f32-tolerance-v1"
    ResidualRmsNormBlock128F32ReductionToleranceV1 =>
      "fused-residual-rmsnorm-block128-f32-tree-tolerance-v1"
  }
}

///|
pub fn FusedParallelNumericalPolicy::maximum_absolute_error_ppb(
  self : FusedParallelNumericalPolicy,
) -> Int64 {
  match self {
    QkvRopeKvWriteOrderedF32TranscendentalToleranceV1 => 250000L
    ResidualRmsNormBlock128F32ReductionToleranceV1 => 100000L
  }
}

///|
pub fn FusedParallelNumericalPolicy::maximum_relative_error_ppb(
  self : FusedParallelNumericalPolicy,
) -> Int64 {
  match self {
    QkvRopeKvWriteOrderedF32TranscendentalToleranceV1 => 500000L
    ResidualRmsNormBlock128F32ReductionToleranceV1 => 250000L
  }
}

///|
pub struct FusedParallelAotLimits {
  priv maximum_source_bytes : Int
  priv maximum_recipe_bytes : Int
  priv maximum_promotion_bytes : Int
}

///|
pub fn FusedParallelAotLimits::new(
  maximum_source_bytes~ : Int,
  maximum_recipe_bytes~ : Int,
  maximum_promotion_bytes~ : Int,
) -> FusedParallelAotLimits raise FusedParallelAotError {
  if maximum_source_bytes <= 0 ||
    maximum_source_bytes > 1048576 ||
    maximum_recipe_bytes <= 0 ||
    maximum_recipe_bytes > 1048576 ||
    maximum_promotion_bytes <= 0 ||
    maximum_promotion_bytes > 1048576 {
    raise FusedParallelInvalid(Limits)
  }
  { maximum_source_bytes, maximum_recipe_bytes, maximum_promotion_bytes }
}

///|
priv struct FusedQkvParameters {
  model_content : String
  model_plan : String
  qkv_operation : @plan.OperationId
  rope_operation : @plan.OperationId
  attention_operation : @plan.OperationId
  kv_layer : @plan.KvLayerId
  input_width : Int
  output_width : Int
  query_heads : Int
  kv_heads : Int
  head_dimension : Int
  rope_theta : Double
  tokens_per_page : Int
  total_page_count : Int
  page_stride_bytes : Int64
  component_stride_bytes : Int64
}

///|
priv struct FusedResidualNormParameters {
  model_content : String
  model_plan : String
  residual_operation : @plan.OperationId
  norm_operation : @plan.OperationId
  width : Int
  epsilon : Double
}

///|
pub struct FusedQkvRopeKvWriteCudaCandidate {
  priv qkv_fallback : @luna_cuda_projection_aot.ProjectionCudaLowering
  priv rope_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  priv kv_write_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv parameters : FusedQkvParameters
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv numerical_policy : FusedParallelNumericalPolicy
  priv function_symbol : String
  priv dispatch_canary : UInt
  priv source : Bytes
  priv source_digest : FusedParallelDigest
  priv recipe : Bytes
  priv recipe_digest : FusedParallelDigest
  priv candidate_digest : FusedParallelDigest
}

///|
/// Canary-free production companion derived from one exact qualification
/// candidate. It carries no deployment or dispatch authority.
pub struct FusedQkvRopeKvWriteProductionCudaCandidate {
  priv qualification : FusedQkvRopeKvWriteCudaCandidate
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : FusedParallelDigest
  priv recipe : Bytes
  priv recipe_digest : FusedParallelDigest
  priv candidate_digest : FusedParallelDigest
}

///|
/// CUDA lowering of the backend-neutral Qwen attention-ingress fusion plan.
/// All vendor decisions remain private to this backend package.
pub struct FusedQwenQkvQkNormRopeKvWriteCudaCandidate {
  priv semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan
  priv qkv_fallback : @luna_cuda_projection_aot.ProjectionCudaLowering
  priv qk_norm_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  priv rope_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  priv kv_write_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv layout : @device_layout.DeviceKvLayout
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv numerical_policy : FusedParallelNumericalPolicy
  priv function_symbol : String
  priv dispatch_canary : UInt
  priv source : Bytes
  priv source_digest : FusedParallelDigest
  priv recipe : Bytes
  priv recipe_digest : FusedParallelDigest
  priv candidate_digest : FusedParallelDigest
}

///|
pub struct FusedResidualRmsNormCudaCandidate {
  priv residual_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  priv norm_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv compiler : @luna_specialization.LunaCompilerPolicy
  priv parameters : FusedResidualNormParameters
  priv dimensions : @launch_contract.AotLaunchDimensions
  priv numerical_policy : FusedParallelNumericalPolicy
  priv function_symbol : String
  priv dispatch_canary : UInt
  priv source : Bytes
  priv source_digest : FusedParallelDigest
  priv recipe : Bytes
  priv recipe_digest : FusedParallelDigest
  priv candidate_digest : FusedParallelDigest
}

///|
/// Canary-free production companion for one exact qualified residual/RMSNorm
/// candidate. It remains inert until a deployment-approved catalog binds its
/// deterministic CUBIN; construction is possible only from the exact
/// qualification candidate so baked model, fallback, target, profile, and
/// compiler identities cannot drift.
pub struct FusedResidualRmsNormProductionCudaCandidate {
  priv qualification : FusedResidualRmsNormCudaCandidate
  priv function_symbol : String
  priv source : Bytes
  priv source_digest : FusedParallelDigest
  priv recipe : Bytes
  priv recipe_digest : FusedParallelDigest
  priv candidate_digest : FusedParallelDigest
}

///|
pub struct FusedParallelPromotionBinding {
  priv family : FusedParallelFamily
  priv candidate_digest : FusedParallelDigest
  priv source_digest : FusedParallelDigest
  priv recipe_digest : FusedParallelDigest
  priv evidence : FusedParallelPromotionEvidence
  priv record : Bytes
  priv digest : FusedParallelDigest
}
