///|
const PAGED_GREEDY_RESULT_ROW_BYTES : Int64 = 8L

///|
/// Joins an inert offline sampler to one exact authenticated graph bundle.
/// The executor reauthenticates this opaque join before loading the symbol.
pub fn admit_paged_cuda_greedy_sampling(
  model_plan : @plan.ModelPlan,
  artifacts : @artifact.KernelArtifactBundle,
  artifact : @luna_cuda_sampling_aot.GreedySamplingCudaArtifact,
) -> PagedSamplingExecutionPolicy raise DeviceStepError {
  let boundary = @luna_cuda_sampling_aot.admit_host_sampling_boundary(
    model_plan,
  ) catch {
    _ => raise InvalidExecutor(SamplingArtifact)
  }
  if artifacts.model_identity() != model_plan.identity() ||
    artifacts.device_target() != artifact.target() ||
    boundary.vocabulary_size() != artifact.vocabulary_size() {
    raise InvalidExecutor(SamplingArtifact)
  }
  let mut module_index = -1
  for index, module_ in artifacts.modules() {
    if module_.digest() == artifact.module_digest() &&
      module_.module_bytes() == artifact.module_bytes() {
      if module_index >= 0 {
        raise InvalidExecutor(SamplingArtifact)
      }
      module_index = index
    }
  }
  if module_index < 0 {
    raise InvalidExecutor(SamplingArtifact)
  }
  PagedCudaGreedySampling({
    model_identity: artifacts.model_identity(),
    target: artifacts.device_target(),
    catalog_version: artifacts.catalog_version(),
    module_index,
    module_digest: artifact.module_digest(),
    module_bytes: artifact.module_bytes(),
    function_symbol: artifact.function_symbol(),
    vocabulary_size: artifact.vocabulary_size(),
    maximum_rows: artifact.maximum_rows(),
    result_row_bytes: artifact.result_row_bytes(),
  })
}

///|
/// Selects the product-owned greedy reducer co-compiled into the exact
/// language-model-head module by the BF16 release producer. The module is
/// already content-authenticated by the full-graph artifact bundle; this join
/// adds no caller-selected bytes or symbol and opens no runtime resource.
pub fn admit_paged_embedded_cuda_greedy_sampling(
  model_plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  artifacts : @artifact.KernelArtifactBundle,
) -> PagedSamplingExecutionPolicy raise DeviceStepError {
  let boundary = @luna_cuda_sampling_aot.admit_host_sampling_boundary(
    model_plan,
  ) catch {
    _ => raise InvalidExecutor(SamplingArtifact)
  }
  if blueprint.identity() != model_plan.identity() ||
    artifacts.model_identity() != model_plan.identity() ||
    blueprint.target() != artifacts.device_target() ||
    blueprint.catalog_version() != artifacts.catalog_version() ||
    blueprint.limits().max_rows() <= 0 {
    raise InvalidExecutor(SamplingArtifact)
  }
  let mut module_index = -1
  for step in blueprint.steps() {
    if step.operation_kind() == LanguageModelHead {
      if module_index >= 0 || step.operation_id() != boundary.producer() {
        raise InvalidExecutor(SamplingArtifact)
      }
      module_index = step.artifact_module_index()
    }
  }
  guard module_index >= 0 &&
    artifacts.modules().get(module_index) is Some(module_) else {
    raise InvalidExecutor(SamplingArtifact)
  }
  PagedCudaGreedySampling({
    model_identity: model_plan.identity(),
    target: artifacts.device_target(),
    catalog_version: artifacts.catalog_version(),
    module_index,
    module_digest: module_.digest(),
    module_bytes: module_.module_bytes(),
    function_symbol: @luna_cuda_sampling_aot.greedy_sampling_cuda_function_symbol(),
    vocabulary_size: boundary.vocabulary_size(),
    maximum_rows: blueprint.limits().max_rows(),
    result_row_bytes: PAGED_GREEDY_RESULT_ROW_BYTES,
  })
}

///|
/// Replays the startup-only sampling join against the exact worker graph.
/// No hash, filesystem operation, allocation, module load, or device call is
/// performed here.
pub fn validate_paged_sampling_execution_policy(
  model_plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  artifacts : @artifact.KernelArtifactBundle,
  policy : PagedSamplingExecutionPolicy,
) -> Unit raise DeviceStepError {
  match policy {
    PagedHostSampling => ()
    PagedCudaGreedySampling(admission) => {
      let expected = admit_paged_embedded_cuda_greedy_sampling(
        model_plan, blueprint, artifacts,
      )
      if expected != policy ||
        admission.function_symbol !=
        @luna_cuda_sampling_aot.greedy_sampling_cuda_function_symbol() {
        raise InvalidExecutor(SamplingArtifact)
      }
    }
    PagedCudaRestrictiveTopKV2(admission) =>
      validate_paged_cuda_device_sampling_admission(
        model_plan, blueprint, artifacts, admission,
      )
  }
}

///|
fn append_paged_steps(
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  tail : PreparedPagedGraphStep,
) -> ReadOnlyArray[PreparedPagedGraphStep] {
  let result : Array[PreparedPagedGraphStep] = []
  for step in steps {
    result.push(step)
  }
  result.push(tail)
  ReadOnlyArray::from_array(result)
}

///|
fn prepare_paged_cuda_greedy_step(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  limits : DeviceStepLimits,
  admission : PagedCudaGreedySamplingAdmission,
  artifacts : @artifact.KernelArtifactBundle,
  completion : PagedCompletionScratch,
  descriptor : DeviceStepResources,
  activation : @device.Allocation,
  resources : PagedGraphResources,
) -> PreparedPagedGraphStep raise DeviceStepError {
  let boundary = @luna_cuda_sampling_aot.admit_host_sampling_boundary(
    model_plan,
  ) catch {
    _ => raise InvalidExecutor(SamplingArtifact)
  }
  guard artifacts.modules().get(admission.module_index) is Some(module_image) &&
    resources.modules.get(admission.module_index) is Some(module_owner) &&
    module_owner.open else {
    raise InvalidExecutor(SamplingArtifact)
  }
  if admission.model_identity != model_plan.identity() ||
    admission.target != artifacts.device_target() ||
    admission.catalog_version != artifacts.catalog_version() ||
    module_image.digest() != admission.module_digest ||
    module_image.module_bytes() != admission.module_bytes ||
    !admission.target.matches_capability(context.capability()) ||
    admission.vocabulary_size != completion.vocabulary_size ||
    admission.maximum_rows != limits.max_rows() ||
    admission.result_row_bytes != PAGED_GREEDY_RESULT_ROW_BYTES {
    raise InvalidExecutor(SamplingArtifact)
  }
  let result_bytes = limits.max_rows().to_int64() *
    PAGED_GREEDY_RESULT_ROW_BYTES
  if result_bytes <= 0L ||
    result_bytes / PAGED_GREEDY_RESULT_ROW_BYTES != limits.max_rows().to_int64() {
    raise InvalidExecutor(SamplingArtifact)
  }
  let function_resource = module_owner.resource.load_function(
    @utf8.encode(admission.function_symbol),
  ) catch {
    error => raise ExecutorDevice(Function, error)
  }
  let function_index = resources.functions.length()
  resources.functions.push({
    resource: function_resource,
    module_index: admission.module_index,
    open: true,
  })
  let result_allocation = context.allocate(result_bytes) catch {
    error => raise ExecutorDevice(SamplingResult, error)
  }
  resources.greedy_sampling_results = Some({
    resource: result_allocation,
    open: true,
  })
  let terminal_bytes = completion.row_bytes * limits.max_tokens().to_int64()
  if terminal_bytes <= 0L ||
    completion.final_output_offset > activation.byte_count() ||
    terminal_bytes > activation.byte_count() - completion.final_output_offset {
    raise InvalidExecutor(SamplingOutput)
  }
  let arguments = [
    @device.KernelArgument::device_region(
      descriptor.counts,
      offset=0L,
      byte_count=4L * 4L,
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      descriptor.query_row_offsets,
      offset=0L,
      byte_count=(limits.max_rows() + 1).to_int64() * 4L,
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      activation,
      offset=completion.final_output_offset,
      byte_count=terminal_bytes,
      alignment=2L,
    ),
    @device.KernelArgument::device_region(
      result_allocation,
      offset=0L,
      byte_count=result_bytes,
      alignment=4L,
    ),
  ] catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
  let packed = @device.KernelArguments::new(arguments) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
  let dimensions = @device.KernelLaunchDimensions::new(
    grid_x=limits.max_rows(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise InvalidExecutor(LaunchDimensions)
  }
  completion.mode = CudaGreedyCompletion(
    FixedArray::make(result_bytes.to_int(), b'\x00'),
  )
  {
    operation_id: boundary.producer(),
    function_index,
    dimensions,
    arguments: packed,
  }
}

///|
fn prepare_paged_sampling_execution(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  limits : DeviceStepLimits,
  policy : PagedSamplingExecutionPolicy,
  artifacts : @artifact.KernelArtifactBundle,
  completion : PagedCompletionScratch,
  descriptor : DeviceStepResources,
  activation : @device.Allocation,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  resources : PagedGraphResources,
) -> ReadOnlyArray[PreparedPagedGraphStep] raise DeviceStepError {
  match policy {
    PagedHostSampling => steps
    PagedCudaGreedySampling(admission) =>
      append_paged_steps(
        steps,
        prepare_paged_cuda_greedy_step(
          context, model_plan, limits, admission, artifacts, completion, descriptor,
          activation, resources,
        ),
      )
    PagedCudaRestrictiveTopKV2(admission) =>
      append_paged_steps(
        steps,
        prepare_paged_cuda_device_sampling_step(
          context, model_plan, blueprint, admission, artifacts, completion, descriptor,
          activation, resources,
        ),
      )
  }
}

///|
fn finish_paged_sampling_preparation(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  artifacts : @artifact.KernelArtifactBundle,
  limits : DeviceStepLimits,
  policy : PagedSamplingExecutionPolicy,
  completion : PagedCompletionScratch,
  descriptor : DeviceStepResources,
  activation : @device.Allocation,
  model_steps : ReadOnlyArray[PreparedPagedGraphStep],
  decode_model_steps : ReadOnlyArray[PreparedPagedGraphStep]?,
  decode_split_model_steps : ReadOnlyArray[PreparedPagedGraphStep]?,
  resources : PagedGraphResources,
  graph_policy : PagedGraphExecutionPolicy,
  fused_residual_spans : ReadOnlyArray[ApprovedFusedResidualRmsNormSpan],
  diagnostic_policy : PagedExecutionDiagnosticPolicy,
  decode_split_minimum_context_tokens? : Int = PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
) -> PagedGraphPreparation raise DeviceStepError {
  let steps = prepare_paged_sampling_execution(
    context, model_plan, blueprint, limits, policy, artifacts, completion, descriptor,
    activation, model_steps, resources,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let decode_steps = match decode_model_steps {
    Some(model_decode) =>
      Some(append_paged_sampling_suffix(model_steps, steps, model_decode))
    None => None
  }
  let decode_split_steps = match decode_split_model_steps {
    Some(model_decode_split) =>
      Some(append_paged_sampling_suffix(model_steps, steps, model_decode_split))
    None => None
  }
  let graph_bucket_table = new_paged_graph_bucket_table(blueprint.profile())
  let ordered_dispatch = prepare_paged_ordered_executors(
    context,
    steps,
    resources,
    graph_policy,
    graph_bucket_table,
    blueprint.profile(),
    decode_steps~,
    decode_split_steps~,
    decode_split_minimum_context_tokens~,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let graph_telemetry = new_paged_graph_telemetry(
    open_paged_ordered_executor(
      resources,
      ordered_dispatch.telemetry_owner_index,
    ).execution_mode(),
    blueprint.graph_shape,
  )
  PagedReady({
    resources,
    completion,
    fp8_frame_envelope: None,
    fp8_frame_admission: None,
    fp8_scale_evidence: None,
    graph_telemetry,
    graph_bucket_table,
    ordered_dispatch,
    fused_residual_spans,
    fused_dispatch_canary: prepare_fused_dispatch_diagnostics(
      diagnostic_policy, fused_residual_spans,
    ),
    lifecycle: GraphReady,
  })
}

///|
fn copy_paged_cuda_greedy_results(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  row_count : Int,
) -> Unit raise DeviceStepError {
  guard executor.completion.mode is CudaGreedyCompletion(host) else { return }
  guard executor.resources.greedy_sampling_results is Some(owner) &&
    owner.open &&
    row_count >= 0 &&
    row_count <= executor.completion.selected_tokens.length() else {
    poison_paged_completion(executor, descriptor)
    raise InvalidExecutor(SamplingOutput)
  }
  descriptor.context.copy_to_fixed_host(
    owner.resource,
    host,
    0L,
    0L,
    row_count.to_int64() * PAGED_GREEDY_RESULT_ROW_BYTES,
  ) catch {
    error =>
      poison_paged_completion_and_raise(
        executor,
        descriptor,
        ExecutorDevice(ActivationReadback, error),
      )
  }
}

///|
fn select_paged_cuda_greedy_result(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  row : Int,
) -> Int raise DeviceStepError {
  guard executor.completion.mode is CudaGreedyCompletion(host) &&
    row >= 0 &&
    row < executor.completion.selected_tokens.length() else {
    return poison_paged_completion_and_raise(
      executor,
      descriptor,
      InvalidExecutor(SamplingOutput),
    )
  }
  let token = read_paged_i32(host, row * 2)
  let status = read_paged_i32(host, row * 2 + 1)
  if status >= 0 {
    return poison_paged_completion_and_raise(
      executor,
      descriptor,
      ExecutorSampling(NonFiniteLogit(row~, token_id=status)),
    )
  }
  if status != -1 || token < 0 || token >= executor.completion.vocabulary_size {
    return poison_paged_completion_and_raise(
      executor,
      descriptor,
      InvalidExecutor(SamplingOutput),
    )
  }
  token
}
