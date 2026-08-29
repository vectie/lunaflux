///|
priv struct DerivedI8Bootstrap {
  plan : @plan.ModelPlan
  inspection : @numeric_weight_file.NumericWeightFileInspection
  kv_layout : @device_layout.DeviceKvLayout
  device_numeric : @device.DeviceNumericCapability
  target : @catalog.DeviceTarget
  step_limits : @device_step.DeviceStepLimits
  manifest_limits : @execution_manifest_file.ExecutionManifestFileLimits
  i8_limits : @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionLimits
  bootstrap_limits : @device_step.DeviceBootstrapLimits
  expected_total_memory_bytes : Int64
}

///|
fn validate_i8_source_envelope(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> Unit raise DeviceWorkerBootstrapError {
  if source.digest() != startup.bootstrap_source_digest() {
    raise Invalid(SourceDigest)
  }
  if source.recipe() != DenseLlamaI8PagedAotV6 ||
    source.kernel().luna_approval() is Some(_) ||
    source.i8_model_source() is None ||
    source.i8_execution() is None {
    raise Invalid(Recipe)
  }
}

///|
fn validate_i8_runtime_envelope(
  startup : @worker_wire.WorkerStartupContract,
  model : @spec.LlamaModelMetadata,
  execution : @worker_wire.WorkerI8BootstrapExecution,
) -> Unit raise DeviceWorkerBootstrapError {
  let spec = model.spec()
  let worker = startup.worker_limits()
  let inference = startup.inference_limits()
  if model.content_digest() != startup.identity().content() {
    raise Invalid(ModelIdentity)
  }
  if worker.max_prefill_rows() > worker.max_plan_rows() ||
    worker.max_decode_rows() > worker.max_plan_rows() ||
    worker.max_sequence_tokens() > spec.context_length() ||
    worker.max_token_id() >= spec.vocabulary_size() ||
    worker.max_completion_slots() < worker.max_plan_rows() {
    raise Invalid(WorkerEnvelope)
  }
  if inference.max_context_tokens() > spec.context_length() ||
    inference.max_input_tokens() > spec.context_length() ||
    inference.max_new_tokens() > spec.context_length() ||
    inference.max_token_id() >= spec.vocabulary_size() ||
    inference.max_token_id() > worker.max_token_id() ||
    inference.max_context_tokens() > worker.max_sequence_tokens() {
    raise Invalid(InferenceEnvelope)
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
}

///|
fn build_i8_limits(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  exact_count : Int,
  tensor_count : Int,
) -> (
  @device_layout.DeviceKvLimits,
  @device_step.DeviceStepLimits,
  @execution_manifest_file.ExecutionManifestFileLimits,
  @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionLimits,
  @device_step.DeviceBootstrapLimits,
  @numeric_weight_file.NumericWeightFileLimits,
) raise DeviceWorkerBootstrapError {
  let ceilings = source.ceilings()
  guard source.i8_execution() is Some(execution) else { raise Invalid(Recipe) }
  let worker = startup.worker_limits()
  let kv_limits = @device_layout.DeviceKvLimits::new(
    tokens_per_page=execution.tokens_per_page(),
    total_page_count=execution.total_page_count(),
    max_arena_bytes=ceilings.max_kv_arena_bytes(),
  ) catch {
    error => raise KvLayout(error)
  }
  let step_limits = @device_step.DeviceStepLimits::new(
    max_rows=worker.max_plan_rows(),
    max_tokens=worker.max_plan_tokens(),
    max_pages=worker.max_plan_pages(),
    max_sequence_tokens=worker.max_sequence_tokens(),
    max_token_id=worker.max_token_id(),
  ) catch {
    error => raise DeviceStep(error)
  }
  let artifact_limits = @artifact.KernelArtifactLimits::new(
    max_modules=exact_count,
    max_entry_points=exact_count,
    max_module_bytes=ceilings.max_module_bytes(),
    max_total_module_bytes=ceilings.max_total_module_bytes(),
    max_function_symbol_bytes=RECIPE_MAX_SYMBOL_BYTES,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let artifact_file_limits = @artifact_file.KernelArtifactFileLimits::new(
    max_manifest_bytes=ceilings.max_execution_manifest_bytes(),
    max_json_depth=RECIPE_JSON_DEPTH,
    max_path_bytes=4096,
    artifact=artifact_limits,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let memory_limits = @device_memory.DeviceMemoryLimits::new(
    activation_alignment=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_activation_arena_bytes(),
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let contract_limits = @launch_contract.LaunchContractLimits::new(
    max_profiles=1,
    max_contracts=exact_count,
    max_operands_per_contract=RECIPE_MAX_OPERANDS,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let manifest_limits = @execution_manifest_file.ExecutionManifestFileLimits::new(
    max_manifest_bytes=ceilings.max_execution_manifest_bytes(),
    max_json_depth=RECIPE_JSON_DEPTH,
    max_modules=exact_count,
    max_operations=exact_count,
    device_memory=memory_limits,
    launch_contract=contract_limits,
    artifact_file=artifact_file_limits,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let i8_limits = @i8_inert_capability_admission.I8InertKernelCapabilityAdmissionLimits::new(
    max_operations=exact_count,
    max_weights=tensor_count,
    max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let bootstrap_limits = @device_step.DeviceBootstrapLimits::new(
    max_modules=exact_count,
    max_entry_points=exact_count,
    max_steps=exact_count,
    max_operands_per_step=RECIPE_MAX_OPERANDS,
    max_symbol_bytes=RECIPE_MAX_SYMBOL_BYTES,
    max_canonical_bytes=RECIPE_MAX_CANONICAL_BYTES,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let numeric_limits = @numeric_weight_file.NumericWeightFileLimits::new(
    max_file_bytes=ceilings.max_weight_file_bytes(),
    max_header_bytes=RECIPE_MAX_WEIGHT_HEADER_BYTES,
    transfer_chunk_bytes=RECIPE_TRANSFER_CHUNK_BYTES,
    max_tensor_count=tensor_count,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  (
    kv_limits, step_limits, manifest_limits, i8_limits, bootstrap_limits, numeric_limits,
  )
}

///|
fn inspect_i8_model_inputs(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  model_root : @approved_fs.ApprovedRoot,
) -> DerivedI8Bootstrap raise DeviceWorkerBootstrapError {
  guard source.i8_model_source() is Some(model_source) &&
    source.i8_execution() is Some(execution) else {
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
  validate_i8_runtime_envelope(startup, model, execution)
  let batch = @llama.LlamaPagedBatchEnvelope::new(
    startup.worker_limits().max_plan_rows(),
  ) catch {
    error => raise Plan(error)
  }
  let plan = @llama.build_paged_symmetric_i8_weight_only_v1(
    model,
    @llama.LlamaI8WeightOnlyPolicy::untied_v1(),
    batch~,
  ) catch {
    error => raise Plan(error)
  }
  let exact_count = exact_graph_count(model.spec().layer_count())
  if plan.operations().length() != exact_count ||
    plan.identity() != startup.identity() {
    raise Invalid(ModelIdentity)
  }
  let (
    kv_limits,
    step_limits,
    manifest_limits,
    i8_limits,
    bootstrap_limits,
    numeric_limits,
  ) = build_i8_limits(startup, source, exact_count, plan.tensor_count())
  let kv_layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    kv_limits,
  ) catch {
    error => raise KvLayout(error)
  }
  let observed = @device.DeviceCapability::from_observed_facts(
    compute_major=execution.compute_major(),
    compute_minor=execution.compute_minor(),
    total_memory_bytes=execution.total_memory_bytes(),
    supports_bf16=execution.supports_bf16(),
    supports_cublas_lt=execution.supports_cublas_lt(),
  ) catch {
    _ => raise Invalid(NumericCapability)
  }
  let device_numeric = @device.admit_symmetric_i8_weight_only_per_output_channel_v1(
    observed,
  ) catch {
    _ => raise Invalid(NumericCapability)
  }
  let target = @catalog.DeviceTarget::from_capability(observed) catch {
    error => raise DeviceTarget(error)
  }
  let layout = @device_materialize.plan_numeric_layout(
    plan,
    alignment_bytes=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_weight_arena_bytes(),
  ) catch {
    _ => raise Invalid(NumericWeightArtifact)
  }
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    model_source.numeric_weights_locator(),
  ) catch {
    _ => raise Invalid(Locator)
  }
  let artifact_digest = @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
    model_source.numeric_weight_artifact_digest(),
  ) catch {
    _ => raise Invalid(Digest)
  }
  let inspection = @numeric_weight_file.inspect_file(
    model_root,
    plan,
    locator,
    expected_artifact_digest=artifact_digest,
    limits=numeric_limits,
    alignment_bytes=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_weight_arena_bytes(),
  ) catch {
    _ => raise Invalid(NumericWeightArtifact)
  }
  if inspection.layout() != layout {
    raise Invalid(NumericWeightArtifact)
  }
  {
    plan,
    inspection,
    kv_layout,
    device_numeric,
    target,
    step_limits,
    manifest_limits,
    i8_limits,
    bootstrap_limits,
    expected_total_memory_bytes: execution.total_memory_bytes(),
  }
}
