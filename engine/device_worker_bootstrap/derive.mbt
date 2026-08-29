///|
priv struct DerivedBootstrap {
  model : @spec.LlamaModelMetadata
  plan : @plan.ModelPlan
  weights : @device_materialize.DeviceWeightFileInspection
  kv_layout : @device_layout.DeviceKvLayout
  target : @catalog.DeviceTarget
  step_limits : @device_step.DeviceStepLimits
  execution_limits : @execution_manifest_file.ExecutionManifestFileLimits
  bootstrap_limits : @device_step.DeviceBootstrapLimits
}

///|
fn validate_source_envelope(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> Unit raise DeviceWorkerBootstrapError {
  if source.digest() != startup.bootstrap_source_digest() {
    raise Invalid(SourceDigest)
  }
  match source.recipe() {
    DenseLlamaPagedAotV2 =>
      if source.kernel().luna_approval() is Some(_) {
        raise Invalid(Recipe)
      }
    DenseLlamaPagedAotV3 =>
      if source.kernel().luna_approval() is None {
        raise Invalid(Recipe)
      }
    DenseLlamaBf16PagedAotExtensionsV10 =>
      if source.kernel().luna_approval() is Some(_) {
        raise Invalid(Recipe)
      }
    _ => raise Invalid(Recipe)
  }
}

///|
fn validate_runtime_envelope(
  startup : @worker_wire.WorkerStartupContract,
  model : @spec.LlamaModelMetadata,
  execution : @worker_wire.WorkerBootstrapExecution,
) -> Unit raise DeviceWorkerBootstrapError {
  let spec = model.spec()
  if model.content_digest() != startup.identity().content() {
    raise Invalid(ModelIdentity)
  }
  let worker = startup.worker_limits()
  let inference = startup.inference_limits()
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
  let context = spec.context_length().to_int64()
  let rows = worker.max_plan_rows().to_int64()
  let tokens_per_page = execution.tokens_per_page().to_int64()
  let pages_per_sequence = (context + tokens_per_page - 1L) / tokens_per_page
  if worker.max_plan_tokens().to_int64() > rows * context {
    raise Invalid(WorkerEnvelope)
  }
  if worker.max_plan_pages() > execution.total_page_count() ||
    worker.max_plan_pages().to_int64() > rows * pages_per_sequence {
    raise Invalid(PageEnvelope)
  }
}

///|
fn build_limits(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  exact_count : Int,
) -> (
  @device_layout.DeviceKvLimits,
  @device_step.DeviceStepLimits,
  @execution_manifest_file.ExecutionManifestFileLimits,
  @device_step.DeviceBootstrapLimits,
) raise DeviceWorkerBootstrapError {
  let ceilings = source.ceilings()
  let execution = source.execution()
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
  (kv_limits, step_limits, manifest_limits, bootstrap_limits)
}

///|
fn inspect_model_inputs(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  model_root : @approved_fs.ApprovedRoot,
) -> DerivedBootstrap raise DeviceWorkerBootstrapError {
  let model_source = source.model()
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
  let config_limit = ceilings
    .max_model_config_bytes()
    .to_int64()
    .min(RECIPE_MAX_CONFIG_BYTES)
  let config_limits = @config_file.ModelConfigFileLimits::new(
    maximum_bytes=config_limit,
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
  validate_runtime_envelope(startup, model, source.execution())
  let exact_count = exact_graph_count(model.spec().layer_count())
  let batch = @llama.LlamaPagedBatchEnvelope::new(
    startup.worker_limits().max_plan_rows(),
  ) catch {
    error => raise Plan(error)
  }
  let plan = @llama.build_paged(model, batch~) catch {
    error => raise Plan(error)
  }
  if plan.operations().length() != exact_count ||
    plan.identity() != startup.identity() {
    raise Invalid(ModelIdentity)
  }
  let (kv_limits, step_limits, manifest_limits, bootstrap_limits) = build_limits(
    startup, source, exact_count,
  )
  let kv_layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    kv_limits,
  ) catch {
    error => raise KvLayout(error)
  }
  let execution = source.execution()
  if !execution.supports_bf16() {
    raise Invalid(RecipeLimit)
  }
  let target = @catalog.DeviceTarget::new(
    compute_major=execution.compute_major(),
    compute_minor=execution.compute_minor(),
    supports_bf16=execution.supports_bf16(),
    supports_cublas_lt=execution.supports_cublas_lt(),
  ) catch {
    error => raise DeviceTarget(error)
  }
  let safetensors = weight_header_limits(ceilings, exact_count)
  let file_limits = @device_materialize.DeviceWeightFileLimits::new(
    max_file_bytes=ceilings.max_weight_file_bytes(),
    transfer_chunk_bytes=RECIPE_TRANSFER_CHUNK_BYTES,
    safetensors~,
  ) catch {
    error => raise Weights(error)
  }
  let weights_locator = @approved_fs.ApprovedRelativeLocator::new(
    model_source.weights_locator(),
  ) catch {
    _ => raise Invalid(Locator)
  }
  let weights = @device_materialize.inspect_file(
    model_root,
    model,
    plan,
    weights_locator,
    limits=file_limits,
    alignment_bytes=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_weight_arena_bytes(),
  ) catch {
    error => raise Weights(error)
  }
  {
    model,
    plan,
    weights,
    kv_layout,
    target,
    step_limits,
    execution_limits: manifest_limits,
    bootstrap_limits,
  }
}

///|
fn weight_header_limits(
  ceilings : @worker_wire.WorkerBootstrapCeilings,
  exact_count : Int,
) -> @safetensors.HeaderLimits raise DeviceWorkerBootstrapError {
  @safetensors.HeaderLimits::new(
    max_header_bytes=RECIPE_MAX_WEIGHT_HEADER_BYTES,
    max_tensor_count=exact_count,
    max_rank=RECIPE_MAX_RANK,
    max_name_length=RECIPE_MAX_TENSOR_NAME_BYTES,
    max_payload_bytes=ceilings.max_weight_file_bytes(),
    max_materialized_tensor_bytes=ceilings
      .max_weight_file_bytes()
      .min(ceilings.max_weight_arena_bytes()),
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
}
