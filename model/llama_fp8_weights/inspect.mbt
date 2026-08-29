///|
/// Authenticates one complete full-context dense-Llama finite-E4M3 W8A8 v1
/// numeric weight file against its exact family plan and deployment-pinned
/// artifact digest.
///
/// Validation, canonical ordinal names, exact dtype spellings, tensor shapes,
/// contiguous non-overlapping ranges, payload values, and digest ownership are
/// delegated to the canonical numeric weight-file authority. The returned
/// inspection is inert: it owns no device allocation and proves no executor,
/// kernel, CUDA, or service readiness.
pub fn inspect_full_context_file(
  root : @approved_fs.ApprovedRoot,
  metadata : @spec.LlamaModelMetadata,
  policy : @llama.LlamaFp8W8A8Policy,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_digest~ : @numeric_weight_file.NumericWeightArtifactDigest,
  limits~ : @numeric_weight_file.NumericWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> @numeric_weight_file.NumericWeightFileInspection raise LlamaFp8WeightInspectionError {
  let plan = @llama.build_finite_e4m3_fp8_w8a8_v1(metadata, policy) catch {
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
/// Authenticates the same exact numeric representation against a paged-KV
/// dense-Llama plan whose admitted batch envelope is part of plan identity.
pub fn inspect_paged_file(
  root : @approved_fs.ApprovedRoot,
  metadata : @spec.LlamaModelMetadata,
  policy : @llama.LlamaFp8W8A8Policy,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_digest~ : @numeric_weight_file.NumericWeightArtifactDigest,
  limits~ : @numeric_weight_file.NumericWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
  batch? : @llama.LlamaPagedBatchEnvelope = @llama.LlamaPagedBatchEnvelope::single_row(),
) -> @numeric_weight_file.NumericWeightFileInspection raise LlamaFp8WeightInspectionError {
  let plan = @llama.build_paged_finite_e4m3_fp8_w8a8_v1(
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
