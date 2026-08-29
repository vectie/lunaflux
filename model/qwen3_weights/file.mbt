///|
fn inspect_qwen3_plan_file(
  root : @approved_fs.ApprovedRoot,
  plan : @plan.ModelPlan,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_digest : @numeric_weight_file.NumericWeightArtifactDigest,
  limits : @numeric_weight_file.NumericWeightFileLimits,
  alignment_bytes : Int64,
  max_arena_bytes : Int64,
  expected_route_manifest_sha256 : StringView,
) -> Qwen3WeightFileInspection raise Qwen3WeightsError {
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
  let route_manifest_sha256 = qwen3_route_manifest_sha256(plan, inspection)
  if expected_route_manifest_sha256.length() != 64 ||
    !expected_route_manifest_sha256.all(ch => {
      ch.is_ascii_digit() || (ch >= 'a' && ch <= 'f')
    }) ||
    route_manifest_sha256 != expected_route_manifest_sha256.to_owned() {
    raise Mismatch(RouteManifest)
  }
  { inspection, route_manifest_sha256 }
}

///|
/// Authenticates one canonical unique-storage BF16 Qwen3 numeric artifact.
/// The LM head is the embedding region's TensorRef(0) alias and therefore has
/// no second materialized region.
pub fn inspect_paged_file(
  root : @approved_fs.ApprovedRoot,
  metadata : @qwen3_spec.Qwen3ModelMetadata,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_artifact_digest~ : @numeric_weight_file.NumericWeightArtifactDigest,
  limits~ : @numeric_weight_file.NumericWeightFileLimits,
  alignment_bytes~ : Int64,
  max_arena_bytes~ : Int64,
  expected_route_manifest_sha256~ : StringView,
  batch? : @qwen3.Qwen3PagedBatchEnvelope = @qwen3.Qwen3PagedBatchEnvelope::single_row(),
) -> Qwen3WeightFileInspection raise Qwen3WeightsError {
  if !metadata.spec().tied_embeddings() {
    raise Mismatch(TiedOutput)
  }
  let plan = @qwen3.build_paged(metadata, batch~) catch {
    error => raise InvalidModelPlan(error)
  }
  inspect_qwen3_plan_file(
    root, plan, locator, expected_artifact_digest, limits, alignment_bytes, max_arena_bytes,
    expected_route_manifest_sha256,
  )
}

///|
/// Reauthenticates and materializes the exact inspected Qwen3 numeric artifact.
/// The returned generic owner retains deterministic close/cleanup authority;
/// no file, digest, or manifest work remains on the token-step path.
pub fn load_inspected_paged_file(
  context : @device.Context,
  root : @approved_fs.ApprovedRoot,
  inspection : Qwen3WeightFileInspection,
) -> @numeric_weight_file.NumericDeviceWeightPreparation raise Qwen3WeightsError {
  @numeric_weight_file.load_inspected_file(
    context,
    root,
    inspection.generic_inspection(),
  ) catch {
    error => raise InvalidNumericWeightFile(error)
  }
}
