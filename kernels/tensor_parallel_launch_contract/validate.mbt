///|
fn validate_local_layout(
  model : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  operations : ReadOnlyArray[@tensor_parallel_plan.TensorParallelOperationPlan],
  world_size : Int,
) -> Unit raise TensorParallelLaunchContractError {
  if layout.identity() != model.identity() {
    raise invalid(Identity)
  }
  let full = model.kv_cache_geometry()
  if world_size <= 1 || full.kv_head_count() % world_size != 0 {
    raise invalid(KvLayout)
  }
  let rank_kv_head_count = full.kv_head_count() / world_size
  let rank_geometry = @plan.KvCacheGeometry::new(
    layer_count=full.layer_count(),
    kv_head_count=rank_kv_head_count,
    head_dimension=full.head_dimension(),
    dtype=full.dtype(),
    layout=full.layout(),
  ) catch {
    _ => raise invalid(KvLayout)
  }
  if !layout.is_canonical_for(model.identity(), rank_geometry) ||
    layout.stable_layout_version() != full.layout().stable_version() {
    raise invalid(KvLayout)
  }
  let full_limits = @device_layout.DeviceKvLimits::new(
    tokens_per_page=layout.tokens_per_page(),
    total_page_count=layout.total_page_count(),
    max_arena_bytes=TP_MAX_INT64,
  ) catch {
    _ => raise invalid(KvLayout)
  }
  let full_layout = @device_layout.plan(model.identity(), full, full_limits) catch {
    _ => raise invalid(CapacityOverflow)
  }
  if layout.page_logical_bytes() >= full_layout.page_logical_bytes() ||
    layout.arena_bytes() >= full_layout.arena_bytes() {
    raise invalid(FullKvReplication)
  }
  for index, operation in operations {
    if operation.kv_head_count() > 0 &&
      (
        operation.kv_head_count() != rank_kv_head_count ||
        operation.head_dimension() != full.head_dimension()
      ) {
      raise invalid(OperationShape, operation=index)
    }
  }
}

///|
fn validate_profiles(
  profiles : ArrayView[@launch_contract.PagedKernelProfile],
  model : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  limits : TensorParallelLaunchContractLimits,
) -> Unit raise TensorParallelLaunchContractError {
  if profiles.is_empty() || profiles.length() > limits.max_profiles {
    raise invalid(ProfileCount)
  }
  let shape = model.shape_constraints()
  let rows = Int64::from_int(shape.max_batch_rows())
  let sequence = Int64::from_int(shape.max_sequence_tokens())
  if rows > TP_MAX_INT64 / sequence {
    raise invalid(CapacityOverflow)
  }
  let token_limit = rows * sequence
  let tokens_per_page = Int64::from_int(layout.tokens_per_page())
  if sequence > TP_MAX_INT64 - (tokens_per_page - 1L) {
    raise invalid(CapacityOverflow)
  }
  let pages_per_row = (sequence + tokens_per_page - 1L) / tokens_per_page
  if rows > TP_MAX_INT64 / pages_per_row {
    raise invalid(CapacityOverflow)
  }
  let page_limit = rows * pages_per_row
  for index, profile in profiles {
    if profile.max_query_rows() > shape.max_batch_rows() ||
      Int64::from_int(profile.max_query_tokens()) > token_limit ||
      Int64::from_int(profile.max_page_table_entries()) > page_limit {
      raise invalid(ProfileEnvelope, profile=index)
    }
    for earlier in 0..<index {
      if profiles[earlier].id() == profile.id() {
        raise invalid(DuplicateProfile, profile=index)
      }
    }
  }
}

///|
fn validate_binding(
  operation : @plan.PlanOperation,
  binding : @catalog.KernelBinding,
  layout : @device_layout.DeviceKvLayout,
) -> @catalog.AotKernelFamily raise TensorParallelLaunchContractError {
  let operation_id = operation.id().as_int()
  if binding.operation_id() != operation.id() ||
    binding.capability() != operation.required_capability() ||
    binding.semantic_version() != @catalog.KernelSemanticVersion::v3() {
    raise invalid(Binding, operation=operation_id)
  }
  guard binding.execution_semantics() is Some(semantics) else {
    raise invalid(ExecutionSemantics, operation=operation_id)
  }
  if semantics.kv_layer() != operation.kv_layer() ||
    semantics.runtime_inputs() != operation.runtime_inputs() ||
    semantics.persistent_state() != operation.persistent_state() {
    raise invalid(ExecutionSemantics, operation=operation_id)
  }
  match semantics.kv_layout() {
    Some(kv) =>
      if kv.layout() != layout.geometry().layout() ||
        kv.stable_version() != layout.stable_layout_version() ||
        kv.tokens_per_page() != layout.tokens_per_page() {
        raise invalid(ExecutionSemantics, operation=operation_id)
      }
    None =>
      if operation.persistent_state() != NoPersistentState {
        raise invalid(ExecutionSemantics, operation=operation_id)
      }
  }
  match binding.implementation() {
    VendorBf16GemmImplicitHeuristicV1 =>
      raise invalid(VendorImplementation, operation=operation_id)
    ContentAddressedAot(family) => family
  }
}

///|
fn validate_catalog(
  model : @plan.ModelPlan,
  resolved : @catalog.ResolvedKernelCatalog,
  layout : @device_layout.DeviceKvLayout,
  expected_target : @catalog.DeviceTarget,
) -> ReadOnlyArray[@catalog.AotKernelFamily] raise TensorParallelLaunchContractError {
  if resolved.catalog_version() != @catalog.CatalogVersion::v3() {
    raise invalid(CatalogVersion)
  }
  if resolved.model_identity() != model.identity() {
    raise invalid(Identity)
  }
  if resolved.device_target() != expected_target {
    raise invalid(DeviceTarget)
  }
  let operations = model.operations()
  let bindings = resolved.bindings()
  if bindings.length() != operations.length() {
    raise invalid(Binding, operation=bindings.length())
  }
  let families : Array[@catalog.AotKernelFamily] = []
  for index, operation in operations {
    families.push(validate_binding(operation, bindings[index], layout))
  }
  ReadOnlyArray::from_array(families)
}

///|
fn validate_inputs(
  profiles : ArrayView[@launch_contract.PagedKernelProfile],
  operations : ReadOnlyArray[@plan.PlanOperation],
  operation_plans : ReadOnlyArray[
    @tensor_parallel_plan.TensorParallelOperationPlan,
  ],
  families : ReadOnlyArray[@catalog.AotKernelFamily],
  layout : @device_layout.DeviceKvLayout,
  expected_rank : Int,
  expected_world_size : Int,
  expected_target : @catalog.DeviceTarget,
  inputs : ArrayView[TensorParallelAotLaunchContractInput],
  limits : TensorParallelLaunchContractLimits,
) -> Unit raise TensorParallelLaunchContractError {
  if operations.length() <= 0 ||
    profiles.length() > limits.max_contracts / operations.length() {
    raise invalid(ContractCount)
  }
  let expected_count = profiles.length() * operations.length()
  if expected_count > limits.max_contracts || inputs.length() != expected_count {
    raise invalid(ContractCount, operation=inputs.length())
  }
  let mut index = 0
  for profile_index, profile in profiles {
    for operation_index, operation in operations {
      let item = inputs[index]
      let geometry = operation_plans[operation_index]
      if item.profile_id != profile.id() || item.operation_id != operation.id() {
        raise invalid(
          ContractOrder,
          profile=profile_index,
          operation=operation_index,
        )
      }
      if item.rank != expected_rank ||
        item.world_size != expected_world_size ||
        item.device_target != expected_target ||
        item.max_query_rows != profile.max_query_rows() ||
        item.max_query_tokens != profile.max_query_tokens() ||
        item.max_page_table_entries != profile.max_page_table_entries() ||
        item.input_width != geometry.input_width() ||
        item.output_width != geometry.output_width() ||
        item.inner_width != geometry.inner_width() ||
        item.query_head_count != geometry.query_head_count() ||
        item.kv_head_count != geometry.kv_head_count() ||
        item.head_dimension != geometry.head_dimension() ||
        item.kv_layout_version != layout.stable_layout_version() ||
        item.tokens_per_page != layout.tokens_per_page() {
        raise invalid(
          ArtifactProfile,
          profile=profile_index,
          operation=operation_index,
        )
      }
      if item.entry_point.family() != families[operation_index] {
        raise invalid(
          ArtifactFamily,
          profile=profile_index,
          operation=operation_index,
        )
      }
      index += 1
    }
  }
}
