///|
priv struct DerivedMistralBootstrap {
  plan : @plan.ModelPlan
  weights : @numeric_weight_file.NumericWeightFileInspection
  kv_layout : @device_layout.DeviceKvLayout
  target : @catalog.DeviceTarget
  step_limits : @device_step.DeviceStepLimits
  execution_limits : @execution_manifest_file.ExecutionManifestFileLimits
  bootstrap_limits : @device_step.DeviceBootstrapLimits
}

///|
fn validate_mistral_source_envelope(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> @worker_wire.WorkerMistralBootstrapModelSource raise DeviceWorkerBootstrapError {
  if source.digest() != startup.bootstrap_source_digest() {
    raise Invalid(SourceDigest)
  }
  if source.recipe() != DenseMistralBf16PagedAotV7 &&
    source.recipe() != DenseMistralBf16PagedAotExtensionsV11 {
    raise Invalid(Recipe)
  }
  guard source.mistral_model_source() is Some(model_source) else {
    raise Invalid(Recipe)
  }
  model_source
}

///|
fn read_mistral_config(
  root : @approved_fs.ApprovedRoot,
  source : @worker_wire.WorkerMistralBootstrapModelSource,
  content : @spec.ContentDigest,
  maximum : Int,
) -> @mistral_spec.MistralModelMetadata raise DeviceWorkerBootstrapError {
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
  @config_reader.parse_mistral_bytes(
    bytes,
    content,
    max_bytes=maximum,
    max_depth=RECIPE_JSON_DEPTH,
  ) catch {
    error => raise MistralConfig(error)
  }
}

///|
fn mistral_runtime_envelope(
  startup : @worker_wire.WorkerStartupContract,
  metadata : @mistral_spec.MistralModelMetadata,
  execution : @worker_wire.WorkerBootstrapExecution,
) -> Unit raise DeviceWorkerBootstrapError {
  let spec = metadata.spec()
  let worker = startup.worker_limits()
  let inference = startup.inference_limits()
  if metadata.content_digest() != startup.identity().content() ||
    worker.max_sequence_tokens() > spec.context_length() ||
    worker.max_token_id() >= spec.vocabulary_size() ||
    inference.max_context_tokens() > spec.context_length() ||
    inference.max_token_id() >= spec.vocabulary_size() ||
    worker.max_plan_pages() > execution.total_page_count() {
    raise Invalid(WorkerEnvelope)
  }
}

///|
fn derive_mistral_bootstrap(
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
  model_root : @approved_fs.ApprovedRoot,
) -> DerivedMistralBootstrap raise DeviceWorkerBootstrapError {
  let model_source = validate_mistral_source_envelope(startup, source)
  let ceilings = source.ceilings()
  let maximum = ceilings.max_model_config_bytes()
  let metadata = read_mistral_config(
    model_root,
    model_source,
    startup.identity().content(),
    maximum,
  )
  let observed_window = metadata.spec().sliding_window().unwrap_or(0)
  if observed_window != model_source.sliding_window_tokens() {
    raise Invalid(ModelIdentity)
  }
  mistral_runtime_envelope(startup, metadata, source.execution())
  let batch = @mistral.MistralPagedBatchEnvelope::new(
    startup.worker_limits().max_plan_rows(),
  ) catch {
    error => raise Plan(error)
  }
  let plan = @mistral.build_paged(metadata, batch~) catch {
    error => raise Plan(error)
  }
  if plan.identity() != startup.identity() {
    raise Invalid(ModelIdentity)
  }
  let exact_count = exact_graph_count(metadata.spec().layer_count())
  let (kv_limits, step_limits, execution_limits, bootstrap_limits) = build_limits(
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
    max_tensor_count=exact_count,
  ) catch {
    _ => raise Invalid(RecipeLimit)
  }
  let inspection = @mistral_weights.inspect_paged_file(
    model_root,
    metadata,
    weights_locator,
    expected_artifact_digest=artifact_digest,
    limits=weight_limits,
    alignment_bytes=RECIPE_ALIGNMENT_BYTES,
    max_arena_bytes=ceilings.max_weight_arena_bytes(),
    batch~,
  ) catch {
    error => raise MistralWeights(error)
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
