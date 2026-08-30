///|
priv struct DerivedQwen3Bootstrap {
  plan : @plan.ModelPlan
  weights : @numeric_weight_file.NumericWeightFileInspection
  kv_layout : @device_layout.DeviceKvLayout
  target : @catalog.DeviceTarget
  step_limits : @device_step.DeviceStepLimits
  execution_limits : @execution_manifest_file.ExecutionManifestFileLimits
  bootstrap_limits : @device_step.DeviceBootstrapLimits
}

///|
fn validate_qwen3_source_envelope(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> @worker_wire.WorkerQwen3BootstrapModelSource raise DeviceWorkerBootstrapError {
  if source.digest() != startup.bootstrap_source_digest() {
    raise Invalid(SourceDigest)
  }
  if source.recipe() != DenseQwen3Bf16PagedAotV12 ||
    source.kernel().luna_approval() is Some(_) ||
    source.kernel().fused_runtime_source() is Some(_) {
    raise Invalid(Recipe)
  }
  match source.kernel().sampling_runtime() {
    HostSamplingRuntime | EmbeddedCudaGreedyV1 => ()
  }
  guard source.qwen3_model_source() is Some(model_source) else {
    raise Invalid(Recipe)
  }
  if !model_source.tied_embeddings() {
    raise Invalid(Recipe)
  }
  model_source
}

///|
fn read_qwen3_config(
  root : @approved_fs.ApprovedRoot,
  source : @worker_wire.WorkerQwen3BootstrapModelSource,
  content : @spec.ContentDigest,
  maximum : Int,
) -> @qwen3_spec.Qwen3ModelMetadata raise DeviceWorkerBootstrapError {
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    source.config_locator(),
  ) catch {
    _ => raise Invalid(Locator)
  }
  let file = root.open_file(locator) catch { _ => raise Invalid(Locator) }
  let outcome : Result[Bytes, DeviceWorkerBootstrapError] = Ok(
    file.read_immutable_snapshot(maximum_bytes=maximum.to_int64()) catch {
      _ => raise Invalid(RecipeLimit)
    },
  ) catch {
    error => Err(error)
  }
  let close_result : Result[Unit, DeviceWorkerBootstrapError] = Ok(
    file.close() catch {
      _ => raise Invalid(Locator)
    },
  ) catch {
    error => Err(error)
  }
  let bytes = match outcome {
    Ok(value) => value
    Err(error) => raise error
  }
  match close_result {
    Err(error) => raise error
    Ok(_) => ()
  }
  if @crypto.bytes_to_hex_string(@crypto.sha256(bytes)) !=
    source.config_digest() {
    raise Invalid(Digest)
  }
  @config_reader.parse_qwen3_bytes(
    bytes,
    content,
    max_bytes=maximum,
    max_depth=RECIPE_JSON_DEPTH,
  ) catch {
    error => raise Qwen3Config(error)
  }
}

///|
fn qwen3_runtime_envelope(
  startup : @worker_wire.WorkerStartupContract,
  metadata : @qwen3_spec.Qwen3ModelMetadata,
  execution : @worker_wire.WorkerBootstrapExecution,
) -> Unit raise DeviceWorkerBootstrapError {
  let spec = metadata.spec()
  let worker = startup.worker_limits()
  let inference = startup.inference_limits()
  if metadata.content_digest() != startup.identity().content() ||
    !spec.tied_embeddings() ||
    worker.max_sequence_tokens() > spec.context_length() ||
    worker.max_token_id() >= spec.vocabulary_size() ||
    inference.max_context_tokens() > spec.context_length() ||
    inference.max_token_id() >= spec.vocabulary_size() ||
    worker.max_plan_pages() > execution.total_page_count() {
    raise Invalid(WorkerEnvelope)
  }
}

///|
fn derive_qwen3_bootstrap(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  model_root : @approved_fs.ApprovedRoot,
) -> DerivedQwen3Bootstrap raise DeviceWorkerBootstrapError {
  let model_source = validate_qwen3_source_envelope(startup, source)
  let ceilings = source.ceilings()
  let metadata = read_qwen3_config(
    model_root,
    model_source,
    startup.identity().content(),
    ceilings.max_model_config_bytes(),
  )
  qwen3_runtime_envelope(startup, metadata, source.execution())
  let batch = @qwen3.Qwen3PagedBatchEnvelope::new(
    startup.worker_limits().max_plan_rows(),
  ) catch {
    error => raise Plan(error)
  }
  let plan = @qwen3.build_paged(metadata, batch~) catch {
    error => raise Plan(error)
  }
  if plan.identity() != startup.identity() ||
    plan.tensor_count() != metadata.spec().layer_count() * 11 + 2 ||
    plan.operations().length() != metadata.spec().layer_count() * 10 + 3 {
    raise Invalid(ModelIdentity)
  }
  let operation_count = plan.operations().length()
  let (kv_limits, step_limits, execution_limits, bootstrap_limits) = build_limits(
    startup, source, operation_count,
  )
  let kv_layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    kv_limits,
  ) catch {
    error => raise KvLayout(error)
  }
  let execution = source.execution()
  let target = @catalog.DeviceTarget::new(
    compute_major=execution.compute_major(),
    compute_minor=execution.compute_minor(),
    supports_bf16=execution.supports_bf16(),
    supports_cublas_lt=execution.supports_cublas_lt(),
  ) catch {
    error => raise DeviceTarget(error)
  }
  let weights_locator = @approved_fs.ApprovedRelativeLocator::new(
    model_source.numeric_weights_locator(),
  ) catch {
    _ => raise Invalid(Locator)
  }
  let artifact_digest = @numeric_weight_file.NumericWeightArtifactDigest::from_sha256(
    model_source.numeric_weight_artifact_digest(),
  ) catch {
    _ => raise Invalid(Digest)
  }
  let weight_limits = @numeric_weight_file.NumericWeightFileLimits::new(
    max_file_bytes=ceilings.max_weight_file_bytes(),
    max_header_bytes=RECIPE_MAX_WEIGHT_HEADER_BYTES,
    transfer_chunk_bytes=RECIPE_TRANSFER_CHUNK_BYTES,
    max_tensor_count=plan.tensor_count(),
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let inspection = @qwen3_weights.inspect_paged_file(
    model_root,
    metadata,
    weights_locator,
    expected_artifact_digest=artifact_digest,
    limits=weight_limits,
    alignment_bytes=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_weight_arena_bytes(),
    expected_route_manifest_sha256=model_source.weight_manifest_digest(),
    batch~,
  ) catch {
    error => raise Qwen3Weights(error)
  }
  if inspection.route_manifest_sha256() != model_source.weight_manifest_digest() {
    raise Invalid(NumericWeightArtifact)
  }
  {
    plan,
    weights: inspection.generic_inspection(),
    kv_layout,
    target,
    step_limits,
    execution_limits,
    bootstrap_limits,
  }
}
