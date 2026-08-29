///|
fn mistral_inspect_plan_file(
  root : @approved_fs.ApprovedRoot,
  plan : @plan.ModelPlan,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_digest : @numeric_weight_file.NumericWeightArtifactDigest,
  limits : @numeric_weight_file.NumericWeightFileLimits,
  alignment_bytes : Int64,
  max_arena_bytes : Int64,
) -> MistralWeightFileInspection raise MistralWeightsError {
  let inspection = @numeric_weight_file.inspect_file(
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
  {
    inspection,
    route_manifest_sha256: mistral_route_manifest_sha256(plan, inspection),
  }
}

///|
/// Authenticates one canonical BF16 numeric-weight artifact against the exact
/// full-context Mistral plan. The result is inert and opens no device.
pub fn inspect_full_context_file(
  root : @approved_fs.ApprovedRoot,
  metadata : @mistral_spec.MistralModelMetadata,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_digest~ : @numeric_weight_file.NumericWeightArtifactDigest,
  limits~ : @numeric_weight_file.NumericWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
) -> MistralWeightFileInspection raise MistralWeightsError {
  let plan = @mistral.build(metadata) catch {
    error => raise InvalidModelPlan(error)
  }
  mistral_inspect_plan_file(
    root, plan, locator, expected_artifact_digest, limits, alignment_bytes, max_arena_bytes,
  )
}

///|
/// Authenticates one canonical BF16 numeric-weight artifact against the exact
/// paged Mistral plan, including the sliding-window-capped plan identity.
pub fn inspect_paged_file(
  root : @approved_fs.ApprovedRoot,
  metadata : @mistral_spec.MistralModelMetadata,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_digest~ : @numeric_weight_file.NumericWeightArtifactDigest,
  limits~ : @numeric_weight_file.NumericWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
  batch? : @mistral.MistralPagedBatchEnvelope = @mistral.MistralPagedBatchEnvelope::single_row(),
) -> MistralWeightFileInspection raise MistralWeightsError {
  let plan = @mistral.build_paged(metadata, batch~) catch {
    error => raise InvalidModelPlan(error)
  }
  mistral_inspect_plan_file(
    root, plan, locator, expected_artifact_digest, limits, alignment_bytes, max_arena_bytes,
  )
}
