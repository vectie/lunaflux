///|
/// Authenticates one complete dense-Llama finite-E4M3 W8A8 v2 numeric file.
/// The v2 plan identity includes the compound gated activation-scale policy;
/// this function cannot reinterpret a v1 inspection as v2 evidence.
pub fn inspect_full_context_file_v2(
  root : @approved_fs.ApprovedRoot,
  metadata : @spec.LlamaModelMetadata,
  policy : @llama.LlamaFp8W8A8V2Policy,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_digest~ : @numeric_weight_file.NumericWeightArtifactDigest,
  limits~ : @numeric_weight_file.NumericWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> @numeric_weight_file.NumericWeightFileInspection raise LlamaFp8WeightInspectionError {
  let plan = @llama.build_finite_e4m3_fp8_w8a8_v2(metadata, policy) catch {
    error => raise InvalidModelPlan(error)
  }
  @numeric_weight_file.inspect_file(
    root,
    plan,
    locator,
    expected_artifact_digest~,
    limits~,
    alignment_bytes~,
    max_arena_bytes~,
  ) catch {
    error => raise InvalidNumericWeightFile(error)
  }
}

///|
/// Authenticates the v2 compound-policy representation against an exact
/// paged-KV plan. The admitted batch envelope is part of the plan identity.
pub fn inspect_paged_file_v2(
  root : @approved_fs.ApprovedRoot,
  metadata : @spec.LlamaModelMetadata,
  policy : @llama.LlamaFp8W8A8V2Policy,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_digest~ : @numeric_weight_file.NumericWeightArtifactDigest,
  limits~ : @numeric_weight_file.NumericWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
  batch? : @llama.LlamaPagedBatchEnvelope = @llama.LlamaPagedBatchEnvelope::single_row(),
) -> @numeric_weight_file.NumericWeightFileInspection raise LlamaFp8WeightInspectionError {
  let plan = @llama.build_paged_finite_e4m3_fp8_w8a8_v2(
    metadata,
    policy,
    batch~,
  ) catch {
    error => raise InvalidModelPlan(error)
  }
  @numeric_weight_file.inspect_file(
    root,
    plan,
    locator,
    expected_artifact_digest~,
    limits~,
    alignment_bytes~,
    max_arena_bytes~,
  ) catch {
    error => raise InvalidNumericWeightFile(error)
  }
}
