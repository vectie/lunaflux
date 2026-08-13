///|
/// Loads and admits one exact canonical v1 paged execution manifest.
///
/// The caller retains the pinned root. The independently typed digest binds
/// the immutable snapshot before parsing. Model semantics, memory placement,
/// catalog v3 semantics, full-graph operands, and launch profile capacities are
/// derived from already-admitted typed inputs rather than deserialized.
pub fn load_paged(
  approved_root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : ExecutionManifestDigest,
  model_plan : @plan.ModelPlan,
  weight_layout : @device_materialize.DeviceWeightLayout,
  kv_layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  device_step_limits : @device_step.DeviceStepLimits,
  limits : ExecutionManifestFileLimits,
) -> PagedExecutionAdmission raise ExecutionManifestFileError {
  let shape = model_plan.shape_constraints()
  if shape.max_batch_rows() != 1 ||
    device_step_limits.max_rows() != 1 ||
    model_plan.kv_execution() != PagedKeyValue ||
    weight_layout.identity().content() != model_plan.identity().content() ||
    kv_layout.identity() != model_plan.identity() ||
    !kv_layout.is_canonical_for(
      model_plan.identity(),
      model_plan.kv_cache_geometry(),
    ) ||
    kv_layout.geometry() != model_plan.kv_cache_geometry() ||
    kv_layout.stable_layout_version() !=
    model_plan.kv_cache_geometry().layout().stable_version() ||
    device_step_limits.max_sequence_tokens() > shape.max_sequence_tokens() ||
    device_step_limits.max_token_id() >= shape.vocabulary_size() ||
    device_step_limits.max_pages() > kv_layout.total_page_count() ||
    device_step_limits.max_pages().to_int64() >
    device_step_limits.max_rows().to_int64() *
    (
      (
        shape.max_sequence_tokens().to_int64() +
        kv_layout.tokens_per_page().to_int64() -
        1L
      ) /
      kv_layout.tokens_per_page().to_int64()
    ) ||
    device_step_limits.max_tokens().to_int64() >
    device_step_limits.max_rows().to_int64() *
    shape.max_sequence_tokens().to_int64() ||
    model_plan.operations().is_empty() ||
    model_plan.operations().length() > limits.max_operations() {
    raise UnsupportedModelEnvelope
  }
  let bytes = read_manifest_snapshot(
    approved_root, locator, expected_digest, limits,
  )
  let claims = parse_manifest(bytes, model_plan, target, limits)
  let resolved = derive_catalog(claims, model_plan, kv_layout, target)
  let static_plan = @device_plan.build(model_plan, weight_layout, resolved) catch {
    _ => raise AdmissionRejected(StaticPlan)
  }
  let memory_plan = @device_memory.plan(static_plan, limits.device_memory()) catch {
    _ => raise AdmissionRejected(MemoryPlan)
  }
  let contracts = derive_contracts(
    claims,
    model_plan,
    static_plan,
    memory_plan,
    kv_layout,
    resolved,
    device_step_limits,
    limits.launch_contract(),
  )
  if contracts.scope() != FullGraph {
    raise AdmissionRejected(LaunchContracts)
  }
  let artifact_source = derive_artifact_source(claims, limits.artifact_file())
  let artifacts = @artifact_file.load_paged_kv_admitted(
    approved_root,
    artifact_source,
    contracts,
    limits.artifact_file(),
  ) catch {
    _ => raise AdmissionRejected(ArtifactFiles)
  }
  let blueprint = @device_step.admit_paged_graph_execution_blueprint(
    model_plan,
    device_step_limits,
    kv_layout,
    static_plan,
    memory_plan,
    claims.profile_id,
    contracts,
    artifacts,
  ) catch {
    _ => raise AdmissionRejected(Blueprint)
  }
  {
    model_plan,
    layout: kv_layout,
    device_step_limits,
    static_plan,
    memory_plan,
    profile_id: claims.profile_id,
    contracts,
    artifacts,
    blueprint,
  }
}
