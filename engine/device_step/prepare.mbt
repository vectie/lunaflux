///|
fn set_resource(
  resources : PartialDeviceStepResources,
  stage : DeviceStepStage,
  allocation : @device.Allocation,
) -> Unit {
  match stage {
    Counts => resources.counts = Some(allocation)
    TokenIds => resources.token_ids = Some(allocation)
    TokenPositions => resources.token_positions = Some(allocation)
    QueryRowOffsets => resources.query_row_offsets = Some(allocation)
    SequenceLengths => resources.sequence_lengths = Some(allocation)
    PageOffsets => resources.page_offsets = Some(allocation)
    PhysicalPageIndices => resources.physical_page_indices = Some(allocation)
  }
}

///|
fn allocate_one(
  context : @device.Context,
  resources : PartialDeviceStepResources,
  stage : DeviceStepStage,
  cells : Int,
) -> @device.Allocation raise DeviceStepError {
  let bytes = byte_capacity(cells)
  let allocation = context.allocate(bytes) catch {
    error => raise Device(stage, error)
  }
  set_resource(resources, stage, allocation)
  if allocation.byte_count() != bytes {
    raise AllocationSizeMismatch(stage)
  }
  context.validate_allocation_region(
    allocation,
    offset=0L,
    byte_count=bytes,
    alignment=DEVICE_BUFFER_ALIGNMENT,
  ) catch {
    error => raise Device(stage, error)
  }
  allocation
}

///|
fn validate_plan_binding(
  model_plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
) -> FixedArray[@plan.KernelCapabilityId] raise DeviceStepError {
  if model_plan.kv_execution() != PagedKeyValue {
    raise InvalidModelPlan
  }
  if layout.identity() != model_plan.identity() ||
    layout.geometry() != model_plan.kv_cache_geometry() ||
    layout.stable_layout_version() != 1 ||
    layout.tokens_per_page() <= 0 ||
    layout.total_page_count() > INT32_MAX {
    raise InvalidLayout
  }
  let shapes = model_plan.shape_constraints()
  if limits.max_rows() > shapes.max_batch_rows() {
    raise InvalidLimit(Rows)
  }
  if limits.max_sequence_tokens() > shapes.max_sequence_tokens() {
    raise InvalidLimit(SequenceTokens)
  }
  if limits.max_token_id() >= shapes.vocabulary_size() {
    raise InvalidLimit(TokenId)
  }
  let source = model_plan.required_capabilities()
  if source.is_empty() {
    raise InvalidModelPlan
  }
  let capabilities = FixedArray::make(source.length(), source[0])
  for index, capability in source {
    capabilities[index] = capability
  }
  capabilities
}

///|
fn construction_failed(
  resources : PartialDeviceStepResources,
  cause : DeviceStepError,
) -> DeviceStepPreparation raise DeviceStepError {
  try close_partial(resources) catch {
    cleanup_failure =>
      CleanupRequired({
        resources,
        cause,
        cleanup_failure,
        lifecycle: ClosingFailed,
      })
  } noraise {
    _ => raise cause
  }
}

///|
fn validate_predecessor_seed(
  has_accepted_predecessor : Bool,
  accepted_predecessor_value : UInt64,
) -> Unit raise DeviceStepError {
  if (!has_accepted_predecessor && accepted_predecessor_value != 0UL) ||
    (
      has_accepted_predecessor &&
      accepted_predecessor_value == 18446744073709551615UL
    ) {
    raise InvalidPlan(PlanSequence)
  }
}

///|
/// Allocates all seven device operands at startup after validating the paged
/// model plan, generation, layout, and capacity inputs. Construction either
/// returns one complete owner, raises after balanced cleanup, or retains an
/// explicit cleanup owner when cleanup itself fails.
pub fn prepare(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  layout : @device_layout.DeviceKvLayout,
  limits : DeviceStepLimits,
  has_accepted_predecessor~ : Bool,
  accepted_predecessor_value~ : UInt64,
) -> DeviceStepPreparation raise DeviceStepError {
  validate_predecessor_seed(
    has_accepted_predecessor, accepted_predecessor_value,
  )
  let required_capabilities = validate_plan_binding(model_plan, layout, limits)
  let resources = empty_partial_resources()
  let counts = allocate_one(context, resources, Counts, COUNTS_LENGTH) catch {
    error => return construction_failed(resources, error)
  }
  let token_ids = allocate_one(
    context,
    resources,
    TokenIds,
    limits.max_tokens(),
  ) catch {
    error => return construction_failed(resources, error)
  }
  let token_positions = allocate_one(
    context,
    resources,
    TokenPositions,
    limits.max_tokens(),
  ) catch {
    error => return construction_failed(resources, error)
  }
  let query_row_offsets = allocate_one(
    context,
    resources,
    QueryRowOffsets,
    row_offset_cells(limits),
  ) catch {
    error => return construction_failed(resources, error)
  }
  let sequence_lengths = allocate_one(
    context,
    resources,
    SequenceLengths,
    limits.max_rows(),
  ) catch {
    error => return construction_failed(resources, error)
  }
  let page_offsets = allocate_one(
    context,
    resources,
    PageOffsets,
    row_offset_cells(limits),
  ) catch {
    error => return construction_failed(resources, error)
  }
  let physical_page_indices = allocate_one(
    context,
    resources,
    PhysicalPageIndices,
    limits.max_pages(),
  ) catch {
    error => return construction_failed(resources, error)
  }
  let complete : DeviceStepResources = {
    counts,
    token_ids,
    token_positions,
    query_row_offsets,
    sequence_lengths,
    page_offsets,
    physical_page_indices,
  }
  Ready({
    context,
    identity: model_plan.identity(),
    model_generation_value: model_generation.value(),
    layout,
    required_capabilities,
    limits,
    host: make_host_buffers(limits),
    resources: complete,
    lifecycle: Ready,
    stage_epoch: 0UL,
    has_accepted_predecessor,
    accepted_predecessor_value,
    staged_sequence_value: 0UL,
  })
}
