///|
priv struct TensorParallelRankChildAdmission {
  model_root : @approved_fs.ApprovedRoot
  kernel_root : @approved_fs.ApprovedRoot
  collective_runtime : @device.TensorCollectiveRuntime
  worker_plan : @tensor_parallel_device_worker.TensorParallelDeviceWorkerPlan
}

///|
fn validate_rank_child_model_envelope(
  startup : @worker_wire.WorkerStartupContract,
  model : @spec.LlamaModelMetadata,
  execution : @worker_wire.WorkerBootstrapExecution,
) -> Unit raise TensorParallelRankChildError {
  let spec = model.spec()
  let worker = startup.worker_limits()
  let inference = startup.inference_limits()
  if model.content_digest() != startup.identity().content() ||
    worker.max_prefill_rows() > worker.max_plan_rows() ||
    worker.max_decode_rows() > worker.max_plan_rows() ||
    worker.max_sequence_tokens() > spec.context_length() ||
    worker.max_token_id() >= spec.vocabulary_size() ||
    worker.max_completion_slots() < worker.max_plan_rows() ||
    inference.max_context_tokens() > spec.context_length() ||
    inference.max_input_tokens() > spec.context_length() ||
    inference.max_new_tokens() > spec.context_length() ||
    inference.max_token_id() > worker.max_token_id() ||
    inference.max_context_tokens() > worker.max_sequence_tokens() {
    raise Invalid(ModelEnvelope)
  }
  let context = spec.context_length().to_int64()
  let rows = worker.max_plan_rows().to_int64()
  let tokens_per_page = execution.tokens_per_page().to_int64()
  if context <= 0L || rows <= 0L || tokens_per_page <= 0L {
    raise Invalid(ModelEnvelope)
  }
  let pages_per_sequence = 1L + (context - 1L) / tokens_per_page
  if rows > 9223372036854775807L / context ||
    rows > 9223372036854775807L / pages_per_sequence {
    raise Invalid(ModelEnvelope)
  }
  if worker.max_plan_tokens().to_int64() > rows * context ||
    worker.max_plan_pages() > execution.total_page_count() ||
    worker.max_plan_pages().to_int64() > rows * pages_per_sequence {
    raise Invalid(ModelEnvelope)
  }
}

///|
fn load_rank_child_model(
  model_root : @approved_fs.ApprovedRoot,
  startup : @worker_wire.WorkerStartupContract,
  source : @worker_wire.EncodedBootstrapSource,
) -> (@spec.LlamaModelMetadata, @plan.ModelPlan) raise TensorParallelRankChildError {
  if !source.has_valid_tensor_parallel_recipe_binding() {
    raise Invalid(SourcePolicy)
  }
  let model_source = source.model()
  let locator = @approved_fs.ApprovedRelativeLocator::new(
    model_source.config_locator(),
  ) catch {
    error => raise Root(error)
  }
  let digest = @config_file.ModelConfigDigest::from_sha256(
    model_source.config_digest(),
  ) catch {
    error => raise Config(error)
  }
  let maximum = source
    .ceilings()
    .max_model_config_bytes()
    .to_int64()
    .min(RANK_CHILD_MAX_CONFIG_BYTES)
  let limits = @config_file.ModelConfigFileLimits::new(
    maximum_bytes=maximum,
    maximum_depth=RANK_CHILD_JSON_DEPTH,
  ) catch {
    error => raise Config(error)
  }
  let model = @config_file.load(
    model_root,
    locator,
    digest,
    startup.identity().content(),
    limits,
  ) catch {
    error => raise Config(error)
  }
  validate_rank_child_model_envelope(startup, model, source.execution())
  let batch = @llama.LlamaPagedBatchEnvelope::new(
    startup.worker_limits().max_plan_rows(),
  ) catch {
    error => raise Plan(error)
  }
  let plan = @llama.build_paged(model, batch~) catch {
    error => raise Plan(error)
  }
  if plan.identity() != startup.identity() {
    raise Invalid(ModelEnvelope)
  }
  (model, plan)
}

///|
fn admit_rank_child_target(
  source : @worker_wire.EncodedBootstrapSource,
  topology : @device_topology.AdmittedLocalTopology,
) -> @catalog.DeviceTarget raise TensorParallelRankChildError {
  let execution = source.execution()
  let target = @catalog.DeviceTarget::new(
    compute_major=execution.compute_major(),
    compute_minor=execution.compute_minor(),
    supports_bf16=execution.supports_bf16(),
    supports_cublas_lt=execution.supports_cublas_lt(),
  ) catch {
    _ => raise Invalid(SourcePolicy)
  }
  if !execution.supports_bf16() || target != topology.target() {
    raise Invalid(SourcePolicy)
  }
  target
}

///|
fn build_rank_child_admission(
  configured : TensorParallelRankChildConfigured,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> TensorParallelRankChildAdmission raise TensorParallelRankChildError {
  let contract = configured.contract
  let startup = contract.startup()
  let source = contract.bootstrap_source()
  let envelope = contract.rank_envelope()
  let topology = @device_topology.admit_local_topology_from_inventory(
    contract.topology(),
  ) catch {
    error => raise Topology(error)
  }
  let target = admit_rank_child_target(source, topology)
  let runtime = @tensor_parallel_worker_bootstrap.admit_runtime_policy(
    source, topology,
  ) catch {
    error => raise Runtime(error)
  }
  let rank = envelope.rank()
  guard runtime.rank_capacity(rank) is Some(rank_capacity) else {
    raise Invalid(SourcePolicy)
  }
  let (model, model_plan) = load_rank_child_model(model_root, startup, source)
  let operation_count = model_plan.operations().length()
  let rank_weight_ceiling = rank_capacity
    .weight_reserved_bytes()
    .min(source.ceilings().max_weight_arena_bytes())
  if rank_weight_ceiling <= 0L {
    raise Invalid(SourcePolicy)
  }
  let weights_locator = @approved_fs.ApprovedRelativeLocator::new(
    source.model().weights_locator(),
  ) catch {
    error => raise Root(error)
  }
  let weight_limits = rank_child_weight_limits(
    source, operation_count, rank_weight_ceiling,
  )
  let inspection = @device_materialize.inspect_sharded_file(
    model_root,
    model,
    model_plan,
    weights_locator,
    rank~,
    world_size=envelope.world_size(),
    alignment_bytes=RANK_CHILD_ALIGNMENT_BYTES,
    max_arena_bytes=rank_weight_ceiling,
    limits=weight_limits,
  ) catch {
    error => raise Weights(error)
  }
  let rank_device = @tensor_parallel_device_plan.TensorParallelRankDevice::from_local_topology(
    topology,
    rank~,
  ) catch {
    error => raise DevicePlan(error)
  }
  let device_plan = @tensor_parallel_device_plan.build(
    model_plan,
    inspection.rank_plan(),
    inspection,
    rank_device,
    target,
  ) catch {
    error => raise DevicePlan(error)
  }
  let model_generation = @worker_protocol.ModelPlanGeneration::from_wire(
    startup.model_generation_value(),
  ) catch {
    _ => raise Invalid(ModelEnvelope)
  }
  let group_generation = @rank_group_protocol.RankGroupGeneration::from_wire(
    envelope.group_generation_value(),
  ) catch {
    _ => raise Invalid(ModelEnvelope)
  }
  let kv_plan = @tensor_parallel_kv_plan.admit_from_topology(
    model_plan,
    model_generation,
    topology,
    runtime,
    tokens_per_page=source.execution().tokens_per_page(),
    total_page_count=source.execution().total_page_count(),
  ) catch {
    error => raise Kv(error)
  }
  let activation_ceiling = rank_capacity
    .activation_workspace_reserved_bytes()
    .min(source.ceilings().max_activation_arena_bytes())
  let limits = rank_child_execution_limits(
    startup, source, operation_count, activation_ceiling,
  )
  let kernel_locator = @approved_fs.ApprovedRelativeLocator::new(
    source.kernel().manifest_locator(),
  ) catch {
    error => raise Root(error)
  }
  let kernel_digest = @execution_manifest_file.ExecutionManifestDigest::from_sha256(
    source.kernel().manifest_digest(),
  ) catch {
    error => raise Manifest(error)
  }
  let local_layout = kv_plan.rank_layout(rank) catch {
    error => raise Kv(error)
  }
  let local_binding = kv_plan.rank_binding(rank) catch {
    error => raise Kv(error)
  }
  let execution = @execution_manifest_file.load_tensor_parallel(
    kernel_root,
    kernel_locator,
    kernel_digest,
    model_plan,
    model_generation,
    inspection.rank_plan(),
    inspection,
    device_plan,
    local_layout,
    local_binding,
    target,
    limits.0,
    limits.1,
    limits.2,
    limits.3,
    external_approval=source.kernel().luna_approval(),
  ) catch {
    error => raise Manifest(error)
  }
  let version = envelope.collective_runtime_version_code()
  let collective_runtime = @device.admit_tensor_collective_runtime(
    minimum_version_code=version,
    maximum_version_code=version,
  ) catch {
    error => raise Device(error)
  }
  if collective_runtime.collective_runtime_version_code() != version {
    raise Invalid(SourcePolicy)
  }
  let worker_plan = @tensor_parallel_device_worker.admit_plan(
    model_plan,
    model_generation,
    group_generation,
    topology,
    startup,
    envelope,
    inspection,
    device_plan,
    execution.contracts(),
    execution.artifacts(),
    execution.profile(),
    limits.3,
    kv_plan,
    envelope.group_digest(),
    envelope.rendezvous_id(),
    collective_runtime_version_code=version,
  ) catch {
    error => raise Worker(error)
  }
  { model_root, kernel_root, collective_runtime, worker_plan }
}
