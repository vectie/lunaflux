///|
/// Exact raw-pointer ABI carried by a fused candidate recipe.
///
/// This vocabulary describes an offline candidate only. It is not a launch
/// contract, a manifest entry, or executable authority.
pub(all) enum FusedParallelKernelAbi {
  QkvPositionedRopePagedKvWriteAbiV1
  QkvPositionedRopePagedKvWriteProductionAbiV2
  QwenQkvQkNormPositionedRopePagedKvWriteAbiV3
  QwenQkvQkNormPositionedRopePagedKvWriteProductionAbiV4
  ResidualRmsNormAbiV1
  ResidualRmsNormProductionAbiV2
} derive(Debug, Eq)

///|
pub(all) enum FusedParallelArtifactPurpose {
  QualificationDiagnostics
  ProductionFastPath
} derive(Debug, Eq)

///|
pub fn FusedParallelKernelAbi::canonical_name(
  self : FusedParallelKernelAbi,
) -> String {
  match self {
    QkvPositionedRopePagedKvWriteAbiV1 =>
      "step_counts,query_positions,query_row_offsets,sequence_lengths," +
      "page_table_offsets,page_table_page_indices,input,q_weight,k_weight," +
      "v_weight,rotated_qkv,key_cache,value_cache,dispatch_canary"
    QkvPositionedRopePagedKvWriteProductionAbiV2 =>
      "step_counts,query_positions,query_row_offsets,sequence_lengths," +
      "page_table_offsets,page_table_page_indices,input,q_weight,k_weight," +
      "v_weight,rotated_qkv,key_cache,value_cache"
    QwenQkvQkNormPositionedRopePagedKvWriteAbiV3 =>
      "step_counts,query_positions,query_row_offsets,sequence_lengths," +
      "page_table_offsets,page_table_page_indices,input,q_weight,k_weight," +
      "v_weight,q_norm_weight,k_norm_weight,rotated_qkv,key_cache,value_cache," +
      "dispatch_canary"
    QwenQkvQkNormPositionedRopePagedKvWriteProductionAbiV4 =>
      "step_counts,query_positions,query_row_offsets,sequence_lengths," +
      "page_table_offsets,page_table_page_indices,input,q_weight,k_weight," +
      "v_weight,q_norm_weight,k_norm_weight,rotated_qkv,key_cache,value_cache"
    ResidualRmsNormAbiV1 =>
      "step_counts,residual,branch,norm_weight,residual_output," +
      "norm_output,dispatch_canary"
    ResidualRmsNormProductionAbiV2 =>
      "step_counts,residual,branch,norm_weight,residual_output,norm_output"
  }
}

///|
pub struct FusedParallelCompiledArtifactLimits {
  priv maximum_module_bytes : Int64
  priv maximum_receipt_bytes : Int
  priv maximum_binding_bytes : Int
}

///|
pub fn FusedParallelCompiledArtifactLimits::new(
  maximum_module_bytes~ : Int64,
  maximum_receipt_bytes~ : Int,
  maximum_binding_bytes~ : Int,
) -> FusedParallelCompiledArtifactLimits raise FusedParallelAotError {
  if maximum_module_bytes <= 0L ||
    maximum_module_bytes > 1073741824L ||
    maximum_receipt_bytes <= 0 ||
    maximum_receipt_bytes > 1048576 ||
    maximum_binding_bytes <= 0 ||
    maximum_binding_bytes > 1048576 {
    raise FusedParallelInvalid(Limits)
  }
  { maximum_module_bytes, maximum_receipt_bytes, maximum_binding_bytes }
}

///|
/// Deterministic CUBIN bytes bound to one exact fused candidate.
///
/// The value is deliberately inert: it exposes no kernel-module input,
/// runtime manifest entry, deployment approval, or promotion projection.
pub struct FusedParallelCompiledArtifactBinding {
  priv family : FusedParallelFamily
  priv candidate_digest : FusedParallelDigest
  priv source_digest : FusedParallelDigest
  priv recipe_digest : FusedParallelDigest
  priv toolchain_digest_sha256 : String
  priv driver_identity_sha256 : String
  priv target : @catalog.DeviceTarget
  priv launch_dimensions : @launch_contract.AotLaunchDimensions
  priv function_symbol : String
  priv kernel_abi : FusedParallelKernelAbi
  priv purpose : FusedParallelArtifactPurpose
  priv dispatch_canary_per_token : UInt
  priv qualification_candidate : FusedParallelDigest?
  priv module_bytes : Bytes
  priv module_digest : FusedParallelDigest
  priv compile_receipt : Bytes
  priv compile_receipt_digest : FusedParallelDigest
  priv binding_record : Bytes
  priv binding_digest : FusedParallelDigest
}
