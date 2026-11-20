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
  let segmented = admission.function_symbol ==
    @luna_cuda_sampling_aot.segmented_greedy_partial_symbol()
  let (output_allocation, output_bytes, segments) = if segmented {
    let plan = @luna_cuda_sampling_aot.plan_segmented_greedy_cuda(
      admission.vocabulary_size,
      limits.max_rows(),
    ) catch {
      _ => raise InvalidExecutor(SamplingArtifact)
    }
    let workspace = allocate_paged_sampling_resource(
      context,
      plan.workspace_bytes(),
      SamplingWorkspace,
    )
    resources.device_sampling_workspace = Some(workspace)
    let (_, count) = plan.partition()
    (workspace.resource, plan.workspace_bytes(), count)
  } else {
    (result_allocation, result_bytes, 1)
  }
  let terminal_bytes = completion.row_bytes * limits.max_tokens().to_int64()
  if terminal_bytes <= 0L ||
    completion.final_output_offset > activation.byte_count() ||
    terminal_bytes > activation.byte_count() - completion.final_output_offset {
    raise InvalidExecutor(SamplingOutput)
  }
  let arguments = [
    @device.KernelArgument::device_region(
      descriptor.counts,
      offset=if sampling_descriptor_is_projected(resources) {
        byte_capacity(OUTPUT_COUNTS_CELL)
      } else {
        0L
      },
      byte_count=4L * 4L,
      alignment=4L,
    ),
    @device.KernelArgument::device_region(
      if sampling_descriptor_is_projected(resources) {
        descriptor.counts
      } else {
        descriptor.query_row_offsets
      },
      offset=if sampling_descriptor_is_projected(resources) {
        byte_capacity(OUTPUT_ROW_ENDS_CELL)
      } else {
        0L
      },
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
      output_allocation,
      offset=0L,
      byte_count=output_bytes,
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
    grid_y=segments,
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
    PagedCudaGreedySampling(admission) => {
      let prepared = append_paged_steps(
        steps,
        prepare_paged_cuda_greedy_step(
          context, model_plan, limits, admission, artifacts, completion, descriptor,
          activation, resources,
        ),
      )
      if admission.function_symbol ==
        @luna_cuda_sampling_aot.segmented_greedy_partial_symbol() {
        append_paged_steps(
          prepared,
          prepare_paged_segmented_greedy_merge(
            prepared[prepared.length() - 1].operation_id,
            limits,
            admission,
            descriptor,
            resources,
          ),
        )
      } else {
        prepared
      }
    }
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
  prefill_wide_model_steps : ReadOnlyArray[PreparedPagedGraphStep]?,
  prefill_split_model_steps : ReadOnlyArray[PreparedPagedGraphStep]?,
  deep_prefill_split_model_steps : ReadOnlyArray[PreparedPagedGraphStep]?,
  decode_model_steps : ReadOnlyArray[PreparedPagedGraphStep]?,
  decode_split_model_steps : ReadOnlyArray[PreparedPagedGraphStep]?,
  resources : PagedGraphResources,
  graph_policy : PagedGraphExecutionPolicy,
  fused_residual_spans : ReadOnlyArray[ApprovedFusedResidualRmsNormSpan],
  diagnostic_policy : PagedExecutionDiagnosticPolicy,
  measured_attention_candidates~ : ReadOnlyArray[Int32]? = None,
  prefill_wide_minimum_query_tokens~ : Int32 = PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
  prefill_split_minimum_context_tokens~ : Int32 = PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
  prefill_split_query_tile~ : Int32 = 1,
  prefill_split_partition_count~ : Int32 = 1,
  deep_prefill_split_minimum_context_tokens~ : Int32 = PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
  deep_prefill_split_query_tile~ : Int32 = 1,
  deep_prefill_split_partition_count~ : Int32 = 1,
  decode_split_minimum_context_tokens~ : Int32 = PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD,
  decode_split_maximum_batch_rows~ : Int32 = 2147483647,
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
  let prefill_wide_steps = match prefill_wide_model_steps {
    Some(model_prefill_wide) =>
      Some(append_paged_sampling_suffix(model_steps, steps, model_prefill_wide))
    None => None
  }
  let prefill_split_steps = match prefill_split_model_steps {
    Some(model_prefill_split) =>
      Some(
        append_paged_sampling_suffix(model_steps, steps, model_prefill_split),
      )
    None => None
  }
  let deep_prefill_split_steps = match deep_prefill_split_model_steps {
    Some(model_prefill_split) =>
      Some(
        append_paged_sampling_suffix(model_steps, steps, model_prefill_split),
      )
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
    measured_attention_candidates~,
    prefill_wide_steps~,
    prefill_wide_minimum_query_tokens~,
    prefill_split_steps~,
    prefill_split_minimum_context_tokens~,
    prefill_split_query_tile~,
    prefill_split_partition_count~,
    deep_prefill_split_steps~,
    deep_prefill_split_minimum_context_tokens~,
    deep_prefill_split_query_tile~,
    deep_prefill_split_partition_count~,
    decode_steps~,
    decode_split_steps~,
    decode_split_minimum_context_tokens~,
    decode_split_maximum_batch_rows~,
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
