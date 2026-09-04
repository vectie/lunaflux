///|
fn validate_numeric_bf16_executor_evidence(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  blueprint : PagedKvExecutionBlueprint,
  expected_layout : @device_materialize.DeviceWeightLayout,
  weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority,
) -> @device.Allocation raise DeviceStepError {
  if blueprint.scope() != FullGraph ||
    !blueprint.target().matches_capability(context.capability()) ||
    model_plan.identity() != blueprint.identity() ||
    weights.identity() != blueprint.identity() ||
    weights.numeric_schema_digest() != model_plan.numeric_binding().digest() ||
    weights.layout() != expected_layout {
    raise InvalidExecutor(WeightIdentity)
  }
  let allocation = weights.allocation() catch { _ => raise NumericWeightClose }
  let layout = weights.layout()
  if layout.numeric_schema_digest() != Some(weights.numeric_schema_digest()) ||
    allocation.byte_count() != blueprint.weight_arena_bytes() ||
    layout.length() != model_plan.tensor_count() {
    raise InvalidExecutor(WeightLayout)
  }
  for step in blueprint.steps() {
    for operand in step.operands() {
      if operand.space() == WeightArena {
        guard operand.role() is WeightInput(reference) &&
          layout.region(reference) is Some(region) &&
          region.offset_bytes() == operand.offset_bytes() &&
          region.byte_length() == operand.byte_count() else {
          raise executor_operation_error(step.operation_id(), WeightRegion)
        }
      }
    }
  }
  allocation
}

///|
fn prepare_numeric_bf16_graph_resources(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  blueprint : PagedKvExecutionBlueprint,
  memory_plan : @device_memory.DeviceMemoryPlan,
  artifacts : @artifact.KernelArtifactBundle,
  weight : @device.Allocation,
  resources : PagedGraphResources,
  has_accepted_predecessor : Bool,
  accepted_predecessor_value : UInt64,
  graph_policy : PagedGraphExecutionPolicy,
  sampling_policy : PagedSamplingExecutionPolicy,
  fused_residual_spans : ReadOnlyArray[ApprovedFusedResidualRmsNormSpan],
  fused_qkv_readonly_span : ApprovedFusedQkvReadOnlyAttentionSpan?,
  qwen_fused_attention_spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
  decode_split_bindings : ReadOnlyArray[
    @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
  ],
  decode_split_maximum_workspace_bytes : Int64,
) -> PagedGraphPreparation raise DeviceStepError {
  validate_predecessor_seed(
    has_accepted_predecessor, accepted_predecessor_value,
  )
  let completion = prepare_paged_completion_scratch(
    model_plan, memory_plan, limits,
  )
  let (functions, function_indices, split_decode_indices) = collect_paged_functions(
    blueprint,
  )
  let descriptor_outcome = prepare(
    context,
    model_plan,
    model_generation,
    layout,
    limits,
    has_accepted_predecessor~,
    accepted_predecessor_value~,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let descriptor = match descriptor_outcome {
    Ready(owner) => owner
    CleanupRequired(failed) => {
      resources.descriptor = Some(Failed(failed))
      return paged_construction_failed(resources, failed.cause())
    }
  }
  resources.descriptor = Some(Complete(descriptor))
  let lease = context.lease_allocation(weight) catch {
    error =>
      return paged_construction_failed(
        resources,
        ExecutorDevice(WeightLease, error),
      )
  }
  resources.weight_lease = Some({ resource: lease, open: true })
  let stream = context.create_stream() catch {
    error =>
      return paged_construction_failed(resources, ExecutorDevice(Stream, error))
  }
  resources.stream = Some({ resource: stream, open: true })
  let activation = context.allocate(
    blueprint.activation_workspace_arena_bytes(),
  ) catch {
    error =>
      return paged_construction_failed(
        resources,
        ExecutorDevice(ActivationWorkspaceArena, error),
      )
  }
  resources.activation_workspace = Some({ resource: activation, open: true })
  let kv = context.allocate(layout.arena_bytes()) catch {
    error =>
      return paged_construction_failed(
        resources,
        ExecutorDevice(PersistentKvArena, error),
      )
  }
  resources.persistent_kv = Some({ resource: kv, open: true })
  if activation.byte_count() != blueprint.activation_workspace_arena_bytes() ||
    kv.byte_count() != layout.arena_bytes() {
    return paged_construction_failed(resources, InvalidExecutor(AllocationSize))
  }
  validate_paged_physical_regions(
    context,
    blueprint,
    descriptor.resources,
    weight,
    activation,
    kv,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  load_paged_modules(context, artifacts, resources) catch {
    error => return paged_construction_failed(resources, error)
  }
  load_paged_functions(functions, resources) catch {
    error => return paged_construction_failed(resources, error)
  }
  let (model_steps, phase_decode_model_steps, partitioned_prefill_model_steps) = prepare_paged_production_steps(
    context,
    blueprint,
    function_indices,
    split_decode_indices,
    descriptor.resources,
    weight,
    activation,
    kv,
    resources,
    fused_residual_spans,
    fused_qkv_readonly_span,
    qwen_fused_attention_spans,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let decode_baseline = match phase_decode_model_steps {
    Some(steps) => steps
    None => model_steps
  }
  let split_decode_model_steps = prepare_paged_decode_split_steps(
    context,
    blueprint,
    decode_baseline,
    decode_split_bindings,
    decode_split_maximum_workspace_bytes,
    descriptor.resources,
    weight,
    activation,
    kv,
    resources,
    fused_qkv_readonly_span,
    qwen_fused_attention_spans,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let decode_split_minimum_context_tokens = paged_decode_split_minimum_context_tokens(
    decode_split_bindings,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let prefill_split_minimum_context_tokens = qwen_partitioned_prefill_minimum_context_tokens(
    blueprint, qwen_fused_attention_spans,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  resources.graph_launch_contract = prepare_qwen_graph_launch_contract(
    blueprint, model_steps, qwen_fused_attention_spans,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  finish_paged_sampling_preparation(
    context,
    model_plan,
    blueprint,
    artifacts,
    limits,
    sampling_policy,
    completion,
    descriptor.resources,
    activation,
    model_steps,
    partitioned_prefill_model_steps,
    phase_decode_model_steps,
    split_decode_model_steps,
    resources,
    graph_policy,
    fused_residual_spans,
    ProductionFastPath,
    prefill_split_minimum_context_tokens~,
    decode_split_minimum_context_tokens~,
  )
}

///|
/// Consumes an exact authenticated generic BF16 numeric-weight authority and
/// prepares the existing dense paged executor. This is family-free: the model
/// plan, execution manifest, layout, and every weight operand are rejoined
/// before any module/function readiness can exist.
pub fn prepare_numeric_bf16_paged_graph_executor_v1(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  profile_id : @launch_contract.KernelProfileId,
  weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority,
  contracts : @launch_contract.PagedKvLaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  has_accepted_predecessor~ : Bool,
  accepted_predecessor_value~ : UInt64,
  luna_authorization? : @luna_artifact_admission.LunaAotKernelAdmission? = None,
  graph_policy? : PagedGraphExecutionPolicy = PagedOrderedEagerOnly,
  sampling_policy? : PagedSamplingExecutionPolicy = PagedHostSampling,
  fused_residual_spans? : ReadOnlyArray[ApprovedFusedResidualRmsNormSpan] = ReadOnlyArray::from_array([],
  ),
  fused_qkv_readonly_span? : ApprovedFusedQkvReadOnlyAttentionSpan? = None,
  qwen_fused_attention_spans? : ReadOnlyArray[ApprovedQwenFusedAttentionSpan] = ReadOnlyArray::from_array([],
  ),
  decode_split_bindings? : ReadOnlyArray[
    @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
  ] = ReadOnlyArray::from_array([]),
  decode_split_maximum_workspace_bytes? : Int64 = PAGED_DECODE_SPLIT_MAXIMUM_WORKSPACE_BYTES,
) -> PagedGraphPreparation raise DeviceStepError {
  let resources = empty_paged_resources()
  resources.numeric_weights = Some(Bf16NumericWeights(weights))
  let blueprint = admit_paged_graph_execution_blueprint(
    model_plan,
    limits,
    layout,
    static_plan,
    memory_plan,
    profile_id,
    contracts,
    artifacts,
    luna_authorization~,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let weight = validate_numeric_bf16_executor_evidence(
    context,
    model_plan,
    blueprint,
    static_plan.weight_layout(),
    weights,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  for span in fused_residual_spans {
    if span.model_generation() != model_generation ||
      span.target() != blueprint.target() ||
      !fused_dispatch_policy_allows_graph(ProductionFastPath, graph_policy) ||
      !fused_dispatch_policy_allows_span(ProductionFastPath, span.kernel_abi()) {
      return paged_construction_failed(
        resources,
        InvalidExecutor(ResourceState),
      )
    }
  }
  match fused_qkv_readonly_span {
    Some(span) =>
      if span.model_generation() != model_generation ||
        span.target() != blueprint.target() {
        return paged_construction_failed(
          resources,
          InvalidExecutor(ResourceState),
        )
      }
    None => ()
  }
  for span in qwen_fused_attention_spans {
    if span.model_generation() != model_generation ||
      span.target() != blueprint.target() {
      return paged_construction_failed(
        resources,
        InvalidExecutor(ResourceState),
      )
    }
  }
  prepare_numeric_bf16_graph_resources(
    context, model_plan, model_generation, layout, limits, blueprint, memory_plan,
    artifacts, weight, resources, has_accepted_predecessor, accepted_predecessor_value,
    graph_policy, sampling_policy, fused_residual_spans, fused_qkv_readonly_span,
    qwen_fused_attention_spans, decode_split_bindings, decode_split_maximum_workspace_bytes,
  ) catch {
    error => paged_construction_failed(resources, error)
  }
}
