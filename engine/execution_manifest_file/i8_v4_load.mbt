///|
fn preflight_i8_v4_envelope(
  model_plan : @plan.ModelPlan,
  weight_layout : @device_materialize.DeviceWeightLayout,
  kv_layout : @device_layout.DeviceKvLayout,
  device_numeric : @device.DeviceNumericCapability,
  target : @catalog.DeviceTarget,
  device_step_limits : @device_step.DeviceStepLimits,
  device_ordinal : Int,
  limits : ExecutionManifestFileLimits,
) -> Unit raise ExecutionManifestFileError {
  let shape = model_plan.shape_constraints()
  let numeric_digest = model_plan.numeric_binding().digest()
  if device_ordinal < 0 ||
    device_ordinal > 1023 ||
    model_plan.kv_execution() != PagedKeyValue ||
    model_plan.operations().is_empty() ||
    model_plan.operations().length() > limits.max_operations() ||
    model_plan.numeric_binding().operations().length() !=
    model_plan.operations().length() ||
    weight_layout.identity() != model_plan.identity() ||
    weight_layout.numeric_schema_digest() != Some(numeric_digest) ||
    weight_layout.length() != model_plan.tensor_count() ||
    kv_layout.identity() != model_plan.identity() ||
    !kv_layout.is_canonical_for(
      model_plan.identity(),
      model_plan.kv_cache_geometry(),
    ) ||
    kv_layout.geometry() != model_plan.kv_cache_geometry() ||
    kv_layout.stable_layout_version() !=
    model_plan.kv_cache_geometry().layout().stable_version() ||
    device_numeric.version() != @device.DeviceNumericCapabilityVersion::v1() ||
    device_numeric.feature() !=
    SymmetricI8WeightOnlyPerOutputChannelF32ScaleF32AccumulateBf16OutputV1 ||
    target.compute_major() != device_numeric.compute_major() ||
    target.compute_minor() != device_numeric.compute_minor() ||
    target.supports_bf16() != device_numeric.supports_bf16() ||
    target.supports_cublas_lt() != device_numeric.supports_cublas_lt() ||
    shape.max_batch_rows() != device_step_limits.max_rows() ||
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
    shape.max_sequence_tokens().to_int64() {
    raise UnsupportedModelEnvelope
  }
}

///|
/// Loads and admits one exact schema-v4/catalog-v4 production-I8 manifest.
///
/// Model-only policy, schema version, catalog version, device target, numeric
/// capability, layout, and resource-envelope checks all precede artifact-file
/// opening. This API has no tensor-parallel variant and cannot consume any
/// legacy catalog/launch/artifact type. Its result remains pre-hardware and
/// cannot construct an executor or ready worker until `join_loaded_weights`
/// consumes the loader's four-way authenticated weight authority.
pub fn load_paged_i8_v4(
  approved_root : @approved_fs.ApprovedRoot,
  locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : ExecutionManifestDigest,
  model_plan : @plan.ModelPlan,
  weight_layout : @device_materialize.DeviceWeightLayout,
  kv_layout : @device_layout.DeviceKvLayout,
  device_numeric : @device.DeviceNumericCapability,
  target : @catalog.DeviceTarget,
  device_step_limits : @device_step.DeviceStepLimits,
  device_ordinal~ : Int,
  limits : ExecutionManifestFileLimits,
  i8_limits : @i8_admission.I8InertKernelCapabilityAdmissionLimits,
  bootstrap_limits : @device_step.DeviceBootstrapLimits,
) -> I8PagedExecutionAdmissionV4 raise ExecutionManifestFileError {
  preflight_i8_v4_envelope(
    model_plan, weight_layout, kv_layout, device_numeric, target, device_step_limits,
    device_ordinal, limits,
  )
  @i8_admission.preflight_production_llama_symmetric_i8_weight_only_v1(
    model_plan,
  ) catch {
    _ => raise AdmissionRejected(I8ProductionPolicy)
  }

  let bytes = read_manifest_snapshot(
    approved_root, locator, expected_digest, limits,
  )
  let claims = parse_manifest_v4(
    bytes, model_plan, device_numeric, target, limits,
  )
  let resolved = derive_catalog_v4(claims, model_plan, kv_layout, target)
  let production_policy = @i8_admission.admit_production_llama_symmetric_i8_weight_only_v1(
    model_plan, device_numeric, resolved, i8_limits,
  ) catch {
    _ => raise AdmissionRejected(I8ProductionPolicy)
  }
  if claims.numeric.i8_admission_digest != production_policy.digest().as_hex() {
    raise IdentityMismatch(I8AdmissionDigest)
  }
  let static_plan = @device_plan.build(model_plan, weight_layout, resolved) catch {
    _ => raise AdmissionRejected(StaticPlan)
  }
  let memory_plan = @device_memory.plan(static_plan, limits.device_memory()) catch {
    _ => raise AdmissionRejected(MemoryPlan)
  }
  let contracts = derive_contracts_v4(
    claims,
    model_plan,
    static_plan,
    memory_plan,
    kv_layout,
    resolved,
    device_step_limits,
    limits.launch_contract(),
  )
  let artifact_source = derive_v4_artifact_source(
    claims,
    limits.artifact_file(),
  )
  let artifacts = @artifact_file.load_paged_v4_admitted(
    approved_root,
    artifact_source,
    contracts,
    limits.artifact_file(),
  ) catch {
    _ => raise AdmissionRejected(ArtifactFiles)
  }
  let blueprint = @device_step.admit_i8_paged_graph_execution_blueprint_v1(
    model_plan,
    device_numeric,
    production_policy,
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
  let bootstrap = @device_step.admit_i8_device_worker_bootstrap_v3(
    blueprint,
    device_ordinal~,
    bootstrap_limits,
  ) catch {
    _ => raise AdmissionRejected(I8BootstrapV3)
  }
  {
    model_plan,
    device_numeric,
    production_policy,
    weight_layout,
    layout: kv_layout,
    device_step_limits,
    static_plan,
    memory_plan,
    profile_id: claims.profile_id,
    contracts,
    artifacts,
    blueprint,
    bootstrap,
    weight_artifact_digest: claims.numeric.weight_artifact_digest,
  }
}
