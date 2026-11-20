///|
fn empty_paged_resources() -> PagedGraphResources {
  {
    descriptor: None,
    stream: None,
    weight_lease: None,
    activation_workspace: None,
    qwen_fused_attention_sidecar: None,
    prefill_partition_workspace: None,
    deep_prefill_partition_workspace: None,
    decode_split_workspace: None,
    persistent_kv: None,
    host_cache: None,
    streaming: None,
    fused_dispatch_canary: None,
    greedy_sampling_results: None,
    device_sampling_parameters: None,
    device_sampling_workspace: None,
    device_sampling_results: None,
    modules: [],
    functions: [],
    projection_variants: [],
    attention_phase_replacements: [],
    attention_partitioned_decode_companions: [],
    ordered_executors: [],
    vendor_projection_exact_rows: 0,
    vendor_projection_rules: [],
    vendor_capture_budget: effect_capture_budget(GraphMemoryUnsupported),
    effect_operation_prefix: 0,
    effect_preparation_prefix: 0,
    effect_capture_budget: effect_capture_budget(GraphMemoryUnsupported),
    graph_launch_contract: None,
    numeric_weights: None,
  }
}

///|
fn executor_operation_error(
  operation : @plan.OperationId,
  rule : PagedGraphExecutorRule,
) -> DeviceStepError {
  InvalidExecutorOperation(operation, rule)
}

///|
fn validate_executor_evidence(
  context : @device.Context,
  blueprint : PagedKvExecutionBlueprint,
  weights : @device_materialize.DeviceWeights,
) -> Unit raise DeviceStepError {
  if blueprint.scope() != FullGraph {
    raise InvalidExecutor(BlueprintScope)
  }
  if !blueprint.target().matches_capability(context.capability()) {
    raise InvalidExecutor(ContextTarget)
  }
  if weights.identity().content() != blueprint.identity().content() {
    raise InvalidExecutor(WeightIdentity)
  }
  let weight_layout = weights.layout()
  if weight_layout.arena_bytes() != blueprint.weight_arena_bytes() ||
    weight_layout.alignment_bytes() != blueprint.weight_alignment() ||
    weights.allocation().byte_count() != blueprint.weight_arena_bytes() {
    raise InvalidExecutor(WeightLayout)
  }
  for step in blueprint.steps() {
    for operand in step.operands() {
      if operand.space() == WeightArena {
        guard operand.role() is WeightInput(reference) &&
          weights.region(reference) is Some(region) &&
          region.offset_bytes() == operand.offset_bytes() &&
          region.byte_length() == operand.byte_count() else {
          raise executor_operation_error(step.operation_id(), WeightRegion)
        }
      }
    }
  }
}

///|
fn descriptor_allocation(
  resources : DeviceStepResources,
  stage : DeviceStepStage,
) -> @device.Allocation {
  match stage {
    Counts => resources.counts
    TokenIds => resources.token_ids
    TokenPositions => resources.token_positions
    QueryRowOffsets => resources.query_row_offsets
    SequenceLengths => resources.sequence_lengths
    PageOffsets => resources.page_offsets
    PhysicalPageIndices => resources.physical_page_indices
  }
}

///|
fn allocation_for_paged_operand(
  binding : PagedKvOperandBinding,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation_workspace : @device.Allocation,
  persistent_kv : @device.Allocation,
) -> @device.Allocation {
  match binding.space() {
    StepDescriptor(stage) => descriptor_allocation(descriptor, stage)
    WeightArena => weight
    ActivationWorkspaceArena => activation_workspace
    PersistentKvArena => persistent_kv
  }
}

///|
fn validate_paged_physical_regions(
  context : @device.Context,
  blueprint : PagedKvExecutionBlueprint,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation_workspace : @device.Allocation,
  persistent_kv : @device.Allocation,
) -> Unit raise DeviceStepError {
  for step in blueprint.steps() {
    for binding in step.operands() {
      context.validate_allocation_region(
        allocation_for_paged_operand(
          binding, descriptor, weight, activation_workspace, persistent_kv,
        ),
        offset=binding.offset_bytes(),
        byte_count=binding.byte_count(),
        alignment=binding.alignment(),
      ) catch {
        error => raise ExecutorDevice(KernelArguments, error)
      }
    }
  }
}
