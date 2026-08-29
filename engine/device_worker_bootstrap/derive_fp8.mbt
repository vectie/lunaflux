///|
priv struct DerivedFp8Bootstrap {
  plan : @plan.ModelPlan
  inspection : @numeric_weight_file.NumericWeightFileInspection
  kv_layout : @device_layout.DeviceKvLayout
  numeric : @device.DeviceNumericCapability
  target : @catalog.DeviceTarget
  step_limits : @device_step.DeviceStepLimits
  manifest_limits : @execution_manifest_file.ExecutionManifestFileLimits
  fp8_limits : @numeric_capability_manifest.Fp8KernelManifestLimits
  startup_limits : @fp8_startup_admission.Fp8StartupAdmissionLimits
  exact_limits : @fp8_runtime_recipe.Fp8RuntimeRecipeV2Limits
  reusable_limits : @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3Limits
  release_limits : @fp8_release_authority.Fp8ReleaseAuthorityLimits
  expected_total_memory_bytes : Int64
}

///|
fn validate_fp8_source_envelope(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> Unit raise DeviceWorkerBootstrapError {
  guard source.fp8_execution() is Some(execution) &&
    source.fp8_model_source() is Some(_) &&
    source.kernel().luna_approval() is Some(_) else {
    raise Invalid(Recipe)
  }
  if source.recipe() != DenseLlamaFp8ReusablePagedAotApprovedV9 ||
    source.digest() != startup.bootstrap_source_digest() ||
    !((execution.compute_major() == 8 && execution.compute_minor() == 9) ||
    (execution.compute_major() == 9 && execution.compute_minor() == 0) ||
    (execution.compute_major() == 12 && execution.compute_minor() == 0)) {
    raise Invalid(Recipe)
  }
}

///|
fn build_fp8_limits(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  operation_count : Int,
  tensor_count : Int,
) -> (
  @device_layout.DeviceKvLimits,
  @device_step.DeviceStepLimits,
  @execution_manifest_file.ExecutionManifestFileLimits,
  @numeric_capability_manifest.Fp8KernelManifestLimits,
  @fp8_startup_admission.Fp8StartupAdmissionLimits,
  @fp8_runtime_recipe.Fp8RuntimeRecipeV2Limits,
  @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3Limits,
  @fp8_release_authority.Fp8ReleaseAuthorityLimits,
  @numeric_weight_file.NumericWeightFileLimits,
) raise DeviceWorkerBootstrapError {
  guard source.fp8_execution() is Some(execution) else { raise Invalid(Recipe) }
  let ceilings = source.ceilings()
  let kv = @device_layout.DeviceKvLimits::new(
    tokens_per_page=execution.tokens_per_page(),
    total_page_count=execution.total_page_count(),
    max_arena_bytes=ceilings.max_kv_arena_bytes(),
  ) catch {
    error => raise KvLayout(error)
  }
  let worker = startup.worker_limits()
  let step = @device_step.DeviceStepLimits::new(
    max_rows=worker.max_plan_rows(),
    max_tokens=worker.max_plan_tokens(),
    max_pages=worker.max_plan_pages(),
    max_sequence_tokens=worker.max_sequence_tokens(),
    max_token_id=worker.max_token_id(),
  ) catch {
    error => raise DeviceStep(error)
  }
  let artifact_limits = @artifact.KernelArtifactLimits::new(
    max_modules=operation_count,
    max_entry_points=operation_count,
    max_module_bytes=ceilings.max_module_bytes(),
    max_total_module_bytes=ceilings.max_total_module_bytes(),
    max_function_symbol_bytes=RECIPE_MAX_SYMBOL_BYTES,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let files = @artifact_file.KernelArtifactFileLimits::new(
    max_manifest_bytes=ceilings.max_execution_manifest_bytes(),
    max_json_depth=RECIPE_JSON_DEPTH,
    max_path_bytes=4096,
    artifact=artifact_limits,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let memory = @device_memory.DeviceMemoryLimits::new(
    activation_alignment=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_activation_arena_bytes(),
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let contracts = @launch_contract.LaunchContractLimits::new(
    max_profiles=1,
    max_contracts=operation_count,
    max_operands_per_contract=RECIPE_MAX_OPERANDS,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let manifest = @execution_manifest_file.ExecutionManifestFileLimits::new(
    max_manifest_bytes=ceilings.max_execution_manifest_bytes(),
    max_json_depth=RECIPE_JSON_DEPTH,
    max_modules=operation_count,
    max_operations=operation_count,
    device_memory=memory,
    launch_contract=contracts,
    artifact_file=files,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let fp8 = @numeric_capability_manifest.Fp8KernelManifestLimits::new(
    max_entries=operation_count,
    max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let startup_limits = @fp8_startup_admission.Fp8StartupAdmissionLimits::new(
    max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let exact = @fp8_runtime_recipe.Fp8RuntimeRecipeV2Limits::new(
    max_launches=operation_count,
    max_workspace_arena_bytes=ceilings.max_activation_arena_bytes(),
    max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let reusable = @fp8_runtime_recipe.Fp8ReusablePagedRecipeV3Limits::new(
    max_launches=operation_count,
    max_workspace_arena_bytes=ceilings.max_activation_arena_bytes(),
    max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let release = @fp8_release_authority.Fp8ReleaseAuthorityLimits::new(
    max_manifest_bytes=ceilings.max_execution_manifest_bytes(),
    max_module_bytes=ceilings.max_module_bytes(),
    max_total_module_bytes=ceilings.max_total_module_bytes(),
    max_operations=operation_count,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let weights = @numeric_weight_file.NumericWeightFileLimits::new(
    max_file_bytes=ceilings.max_weight_file_bytes(),
    max_header_bytes=RECIPE_MAX_WEIGHT_HEADER_BYTES,
    transfer_chunk_bytes=RECIPE_TRANSFER_CHUNK_BYTES,
    max_tensor_count=tensor_count,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  (kv, step, manifest, fp8, startup_limits, exact, reusable, release, weights)
}

///|
fn inspect_fp8_model_inputs(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  model_root : @approved_fs.ApprovedRoot,
) -> DerivedFp8Bootstrap raise DeviceWorkerBootstrapError {
  guard source.fp8_model_source() is Some(model_source) &&
    source.fp8_execution() is Some(execution) else {
    raise Invalid(Recipe)
  }
  let ceilings = source.ceilings()
  let config_locator = @approved_fs.ApprovedRelativeLocator::new(
    model_source.config_locator(),
  ) catch {
    _ => raise Invalid(Locator)
  }
  let config_digest = @config_file.ModelConfigDigest::from_sha256(
    model_source.config_digest(),
  ) catch {
    _ => raise Invalid(Digest)
  }
  let config_limits = @config_file.ModelConfigFileLimits::new(
    maximum_bytes=ceilings
      .max_model_config_bytes()
      .to_int64()
      .min(RECIPE_MAX_CONFIG_BYTES),
    maximum_depth=RECIPE_JSON_DEPTH,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let model = @config_file.load(
    model_root,
    config_locator,
    config_digest,
    startup.identity().content(),
    config_limits,
  ) catch {
    error => raise Config(error)
  }
  let spec = model.spec()
  let worker = startup.worker_limits()
  let inference = startup.inference_limits()
  if model.content_digest() != startup.identity().content() ||
    worker.max_plan_rows() <= 0 ||
    worker.max_prefill_rows() > worker.max_plan_rows() ||
    worker.max_decode_rows() > worker.max_plan_rows() ||
    worker.max_sequence_tokens() > spec.context_length() ||
    worker.max_token_id() >= spec.vocabulary_size() ||
    worker.max_completion_slots() < worker.max_plan_rows() ||
    inference.max_input_tokens() > spec.context_length() ||
    inference.max_new_tokens() > spec.context_length() ||
    inference.max_context_tokens() > worker.max_sequence_tokens() ||
    inference.max_token_id() > worker.max_token_id() {
    raise Invalid(WorkerEnvelope)
  }
  let rows = worker.max_plan_rows().to_int64()
  let context = spec.context_length().to_int64()
  let per_page = execution.tokens_per_page().to_int64()
  let pages_per_sequence = (context + per_page - 1L) / per_page
  if worker.max_plan_tokens().to_int64() > rows * context ||
    worker.max_plan_pages() > execution.total_page_count() ||
    worker.max_plan_pages().to_int64() > rows * pages_per_sequence {
    raise Invalid(PageEnvelope)
  }
  let batch = @llama.LlamaPagedBatchEnvelope::new(worker.max_plan_rows()) catch {
    error => raise Plan(error)
  }
  let plan = @llama.build_paged_finite_e4m3_fp8_w8a8_v2(
    model,
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
    batch~,
  ) catch {
    error => raise Plan(error)
  }
  if plan.identity() != startup.identity() {
    raise Invalid(ModelIdentity)
  }
  let count = plan.operations().length()
  let (
    kv_limits,
    step_limits,
    manifest_limits,
    fp8_limits,
    startup_limits,
    exact_limits,
    reusable_limits,
    release_limits,
    weight_limits,
  ) = build_fp8_limits(startup, source, count, plan.tensor_count())
  let kv_layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    kv_limits,
  ) catch {
    error => raise KvLayout(error)
  }
  let capability = @device.DeviceCapability::from_observed_facts(
    compute_major=execution.compute_major(),
    compute_minor=execution.compute_minor(),
    total_memory_bytes=execution.total_memory_bytes(),
    supports_bf16=execution.supports_bf16(),
    supports_cublas_lt=execution.supports_cublas_lt(),
  ) catch {
    _ => raise Invalid(NumericCapability)
  }
  let numeric = @device.admit_finite_fp8_e4m3_w8a8(capability) catch {
    _ => raise Invalid(NumericCapability)
  }
  let target = @catalog.DeviceTarget::from_capability(capability) catch {
    error => raise DeviceTarget(error)
  }
  let inspection = @llama_fp8_weights.inspect_paged_file_v2(
    model_root,
    model,
    @llama.LlamaFp8W8A8V2Policy::untied_v2(),
    @approved_fs.ApprovedRelativeLocator::new(
      model_source.numeric_weights_locator(),
    ) catch {
      _ => raise Invalid(Locator)
    },
    expected_artifact_digest=@numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
      model_source.numeric_weight_artifact_digest(),
    ) catch {
      _ => raise Invalid(Digest)
    },
    limits=weight_limits,
    alignment_bytes=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_weight_arena_bytes(),
    batch~,
  ) catch {
    _ => raise Invalid(NumericWeightArtifact)
  }
  {
    plan,
    inspection,
    kv_layout,
    numeric,
    target,
    step_limits,
    manifest_limits,
    fp8_limits,
    startup_limits,
    exact_limits,
    reusable_limits,
    release_limits,
    expected_total_memory_bytes: execution.total_memory_bytes(),
  }
}
