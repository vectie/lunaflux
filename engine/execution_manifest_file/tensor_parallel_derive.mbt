///|
fn tensor_parallel_reject_envelope() -> Unit raise ExecutionManifestFileError {
  raise UnsupportedModelEnvelope
}

///|
fn validate_tensor_parallel_device_plan(
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
) -> Unit raise ExecutionManifestFileError {
  if device_plan.identity() != rank_plan.identity() ||
    device_plan.arena_bytes() != rank_plan.arena_bytes() ||
    device_plan.materialized_bytes() != rank_plan.materialized_bytes() ||
    device_plan.tensor_count() != rank_plan.tensor_count() ||
    device_plan.operation_count() != rank_plan.operation_count() ||
    device_plan.collective_count() != rank_plan.collective_count() {
    tensor_parallel_reject_envelope()
  }
  for index in 0..<rank_plan.tensor_count() {
    let reference = @plan.TensorRef::new(index) catch {
      _ => raise UnsupportedModelEnvelope
    }
    guard rank_plan.tensor(reference) is Some(tensor) &&
      device_plan.tensor(tensor.reference()) is Some(view) else {
      tensor_parallel_reject_envelope()
      return
    }
    let extent = tensor.extent()
    let region = tensor.region()
    if view.reference() != tensor.reference() ||
      view.placement() != tensor.placement() ||
      view.full_rows() != extent.full_rows() ||
      view.full_columns() != extent.full_columns() ||
      view.local_row_start() != extent.row_start() ||
      view.local_rows() != extent.row_count() ||
      view.local_column_start() != extent.column_start() ||
      view.local_columns() != extent.column_count() ||
      view.region_offset_bytes() != region.offset_bytes() ||
      view.region_byte_length() != region.byte_length() {
      tensor_parallel_reject_envelope()
    }
  }
  for index in 0..<rank_plan.operation_count() {
    let id = @plan.OperationId::new(index) catch {
      _ => raise UnsupportedModelEnvelope
    }
    guard rank_plan.operation(id) is Some(expected) &&
      device_plan.operation(id) is Some(actual) else {
      tensor_parallel_reject_envelope()
      return
    }
    if actual != expected {
      tensor_parallel_reject_envelope()
    }
  }
  for index in 0..<rank_plan.collective_count() {
    if device_plan.collective(index) != rank_plan.collective(index) {
      tensor_parallel_reject_envelope()
    }
  }
}

///|
fn validate_tensor_parallel_kv(
  model : @plan.ModelPlan,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  local_layout : @device_layout.DeviceKvLayout,
  binding : @tensor_parallel_kv_plan.TensorParallelKvRankBinding,
  model_generation : @worker_protocol.ModelPlanGeneration,
  target : @catalog.DeviceTarget,
) -> @device_layout.DeviceKvLayout raise ExecutionManifestFileError {
  let topology = rank_plan.topology()
  let full = model.kv_cache_geometry()
  let world = topology.world_size()
  if world <= 1 || full.kv_head_count() % world != 0 {
    tensor_parallel_reject_envelope()
  }
  let local_heads = full.kv_head_count() / world
  let local_geometry = @plan.KvCacheGeometry::new(
    layer_count=full.layer_count(),
    kv_head_count=local_heads,
    head_dimension=full.head_dimension(),
    dtype=full.dtype(),
    layout=full.layout(),
  ) catch {
    _ => raise UnsupportedModelEnvelope
  }
  if local_layout.identity() != model.identity() ||
    !local_layout.is_canonical_for(model.identity(), local_geometry) ||
    local_layout.geometry() != local_geometry ||
    binding.identity() != model.identity() ||
    binding.generation_value() != model_generation.value() ||
    binding.rank() != topology.rank() ||
    binding.world_size() != world ||
    binding.process_ordinal() != device_plan.device_ordinal() ||
    binding.target() != target ||
    binding.global_kv_head_count() != full.kv_head_count() ||
    binding.local_kv_head_count() != local_heads ||
    binding.layer_count() != full.layer_count() ||
    binding.head_dimension() != full.head_dimension() ||
    binding.layout_version() != local_layout.stable_layout_version() ||
    binding.tokens_per_page() != local_layout.tokens_per_page() ||
    binding.total_page_count() != local_layout.total_page_count() ||
    binding.page_logical_bytes() != local_layout.page_logical_bytes() ||
    binding.page_stride_bytes() != local_layout.page_stride_bytes() ||
    binding.component_stride_bytes() != local_layout.component_stride_bytes() ||
    binding.layer_stride_bytes() != local_layout.layer_stride_bytes() ||
    binding.arena_bytes() != local_layout.arena_bytes() ||
    binding.kv_reservation_bytes() < local_layout.arena_bytes() {
    tensor_parallel_reject_envelope()
  }
  let full_limits = @device_layout.DeviceKvLimits::new(
    tokens_per_page=local_layout.tokens_per_page(),
    total_page_count=local_layout.total_page_count(),
    max_arena_bytes=9223372036854775807L,
  ) catch {
    _ => raise UnsupportedModelEnvelope
  }
  let full_layout = @device_layout.plan(model.identity(), full, full_limits) catch {
    _ => raise UnsupportedModelEnvelope
  }
  if full_layout.arena_bytes() <= local_layout.arena_bytes() ||
    full_layout.page_logical_bytes() <= local_layout.page_logical_bytes() {
    tensor_parallel_reject_envelope()
  }
  full_layout
}

///|
fn validate_tensor_parallel_envelope(
  model : @plan.ModelPlan,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  inspection : @device_materialize.DeviceShardedWeightFileInspection,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  local_layout : @device_layout.DeviceKvLayout,
  binding : @tensor_parallel_kv_plan.TensorParallelKvRankBinding,
  model_generation : @worker_protocol.ModelPlanGeneration,
  target : @catalog.DeviceTarget,
  step_limits : @device_step.DeviceStepLimits,
  limits : ExecutionManifestFileLimits,
) -> @device_layout.DeviceKvLayout raise ExecutionManifestFileError {
  let topology = rank_plan.topology()
  let shape = model.shape_constraints()
  if model.kv_execution() != PagedKeyValue ||
    model.identity() != rank_plan.identity() ||
    inspection.rank_plan() != rank_plan ||
    device_plan.rank() != topology.rank() ||
    device_plan.world_size() != topology.world_size() ||
    device_plan.target() != target ||
    shape.max_batch_rows() != step_limits.max_rows() ||
    step_limits.max_sequence_tokens() > shape.max_sequence_tokens() ||
    step_limits.max_token_id() != shape.vocabulary_size() - 1 ||
    step_limits.max_pages() > local_layout.total_page_count() ||
    step_limits.max_tokens().to_int64() >
    step_limits.max_rows().to_int64() * shape.max_sequence_tokens().to_int64() ||
    model.operations().is_empty() ||
    model.operations().length() > limits.max_operations() {
    tensor_parallel_reject_envelope()
  }
  let pages_per_row = (
      shape.max_sequence_tokens().to_int64() +
      local_layout.tokens_per_page().to_int64() -
      1L
    ) /
    local_layout.tokens_per_page().to_int64()
  if step_limits.max_pages().to_int64() >
    step_limits.max_rows().to_int64() * pages_per_row {
    tensor_parallel_reject_envelope()
  }
  validate_tensor_parallel_device_plan(rank_plan, device_plan)
  validate_tensor_parallel_kv(
    model, rank_plan, device_plan, local_layout, binding, model_generation, target,
  )
}

///|
fn derive_tensor_parallel_profile(
  claims : ManifestClaims,
  step_limits : @device_step.DeviceStepLimits,
) -> @launch_contract.PagedKernelProfile raise ExecutionManifestFileError {
  @launch_contract.PagedKernelProfile::new(
    id=claims.profile_id,
    max_query_rows=step_limits.max_rows(),
    max_query_tokens=step_limits.max_tokens(),
    max_page_table_entries=step_limits.max_pages(),
  ) catch {
    _ => raise AdmissionRejected(TensorParallelLaunchContracts)
  }
}

///|
fn derive_tensor_parallel_contracts(
  claims : ManifestClaims,
  model : @plan.ModelPlan,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  local_layout : @device_layout.DeviceKvLayout,
  resolved : @catalog.ResolvedKernelCatalog,
  target : @catalog.DeviceTarget,
  profile : @launch_contract.PagedKernelProfile,
  limits : @tensor_parallel_launch_contract.TensorParallelLaunchContractLimits,
) -> @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet raise ExecutionManifestFileError {
  let topology = rank_plan.topology()
  let inputs : Array[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput,
  ] = []
  for index, operation in model.operations() {
    let claim = claims.operations[index]
    let module_claim = claims.modules[claim.module_index]
    guard rank_plan.operation(operation.id()) is Some(geometry) else {
      raise AdmissionRejected(TensorParallelLaunchContracts)
    }
    inputs.push(
      @tensor_parallel_launch_contract.TensorParallelAotLaunchContractInput::new(
        profile_id=claims.profile_id,
        operation_id=operation.id(),
        entry_point=@catalog.AotKernelEntryPoint::new(
          family=@catalog.AotKernelFamily::new(
            module_digest=module_claim.digest,
            family_id=claim.family_id,
          ),
          entry_point_id=claim.entry_point_id,
        ),
        dimensions=claim.dimensions,
        rank=topology.rank(),
        world_size=topology.world_size(),
        device_target=target,
        max_query_rows=profile.max_query_rows(),
        max_query_tokens=profile.max_query_tokens(),
        max_page_table_entries=profile.max_page_table_entries(),
        input_width=geometry.input_width(),
        output_width=geometry.output_width(),
        inner_width=geometry.inner_width(),
        query_head_count=geometry.query_head_count(),
        kv_head_count=geometry.kv_head_count(),
        head_dimension=geometry.head_dimension(),
        kv_layout_version=local_layout.stable_layout_version(),
        tokens_per_page=local_layout.tokens_per_page(),
      ),
    )
  }
  @tensor_parallel_launch_contract.admit_paged_rank(
    model,
    rank_plan,
    local_layout,
    resolved,
    expected_rank=topology.rank(),
    expected_world_size=topology.world_size(),
    expected_target=target,
    [profile],
    inputs,
    limits,
  ) catch {
    _ => raise AdmissionRejected(TensorParallelLaunchContracts)
  }
}
