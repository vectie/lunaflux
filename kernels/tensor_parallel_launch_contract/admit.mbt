///|
/// Admits an exact rank-local AOT launch contract for every paged profile and
/// admitted semantic operation. This function allocates only immutable startup
/// metadata; it performs no module load, device allocation, or launch.
pub fn admit_paged_rank(
  model_plan : @plan.ModelPlan,
  rank_plan : @tensor_parallel_plan.TensorParallelRankPlan,
  device_kv_layout : @device_layout.DeviceKvLayout,
  resolved : @catalog.ResolvedKernelCatalog,
  expected_rank~ : Int,
  expected_world_size~ : Int,
  expected_target~ : @catalog.DeviceTarget,
  profiles : ArrayView[@launch_contract.PagedKernelProfile],
  inputs : ArrayView[TensorParallelAotLaunchContractInput],
  limits : TensorParallelLaunchContractLimits,
) -> TensorParallelAotLaunchContractSet raise TensorParallelLaunchContractError {
  let operation_plans = validate_operation_plans(model_plan, rank_plan)
  validate_rank_plan(model_plan, rank_plan, expected_rank, expected_world_size)
  validate_local_layout(
    model_plan, device_kv_layout, operation_plans, expected_world_size,
  )
  validate_profiles(profiles, model_plan, device_kv_layout, limits)
  let families = validate_catalog(
    model_plan, resolved, device_kv_layout, expected_target,
  )
  let operations = model_plan.operations()
  validate_inputs(
    profiles, operations, operation_plans, families, device_kv_layout, expected_rank,
    expected_world_size, expected_target, inputs, limits,
  )
  let collectives = validate_collective_facts(
    rank_collective_facts(rank_plan),
    operation_plans,
  )
  let bindings = resolved.bindings()
  let contracts : Array[TensorParallelAotLaunchContract] = []
  let rank_constants = build_rank_constants(
    expected_rank, expected_world_size, expected_target, device_kv_layout,
  )
  let mut input_index = 0
  for profile_index, profile in profiles {
    for operation_index, operation in operations {
      let geometry = operation_plans[operation_index]
      let item = inputs[input_index]
      contracts.push({
        profile,
        operation_id: operation.id(),
        operation_kind: operation.kind(),
        capability: operation.required_capability(),
        entry_point: item.entry_point,
        dimensions: item.dimensions,
        input_width: geometry.input_width(),
        output_width: geometry.output_width(),
        inner_width: geometry.inner_width(),
        query_head_count: geometry.query_head_count(),
        kv_head_count: geometry.kv_head_count(),
        head_dimension: geometry.head_dimension(),
        operands: build_operands(
          operation,
          bindings[operation_index],
          rank_plan,
          device_kv_layout,
          profile,
          profile_index,
          geometry,
          limits.max_operands_per_contract,
        ),
        collective_after: collectives[operation_index],
      })
      input_index += 1
    }
  }
  {
    model_identity: model_plan.identity(),
    rank: expected_rank,
    world_size: expected_world_size,
    device_target: expected_target,
    catalog_version: resolved.catalog_version(),
    device_kv_layout,
    rank_constants,
    profiles: ReadOnlyArray::from_array(profiles),
    contracts: ReadOnlyArray::from_array(contracts),
  }
}
