///|
fn plan_rank_constants(
  contracts : ArrayView[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  ],
  expected_version : Int,
) -> TensorParallelScalarRegion raise TensorParallelExecutionPlanError {
  let mut byte_count = -1L
  let mut alignment = -1L
  for operation, contract in contracts {
    let mut found = 0
    for operand_index, operand in contract.operands() {
      match operand.role() {
        RankConstants(version~) => {
          if version != expected_version ||
            (byte_count >= 0L && byte_count != operand.byte_count()) ||
            (alignment >= 0L && alignment != operand.alignment()) {
            raise rejected(RankConstants, operation~, operand=operand_index)
          }
          byte_count = operand.byte_count()
          alignment = operand.alignment()
          found += 1
        }
        _ => ()
      }
    }
    if found != 1 {
      raise rejected(RankConstants, operation~)
    }
  }
  if byte_count <= 0L || alignment <= 0L {
    raise rejected(RankConstants)
  }
  scalar_region(RankConstantsSpace, 0L, byte_count, alignment)
}

///|
fn plan_workspace(
  contracts : ArrayView[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  ],
  activation_bytes : Int64,
  limits : TensorParallelExecutionPlanLimits,
) -> (TensorParallelScalarRegion, Int64) raise TensorParallelExecutionPlanError {
  let mut byte_count = 0L
  let mut alignment = 1L
  for operation, contract in contracts {
    let mut found = 0
    for operand_index, operand in contract.operands() {
      if operand.role() is Workspace {
        if operand.byte_count() <= 0L || !valid_alignment(operand.alignment()) {
          raise rejected(WorkspaceRegion, operation~, operand=operand_index)
        }
        byte_count = byte_count.max(operand.byte_count())
        alignment = alignment.max(operand.alignment())
        found += 1
      }
    }
    if found > 1 {
      raise rejected(WorkspaceRegion, operation~)
    }
  }
  let offset = checked_align(activation_bytes, alignment)
  let workspace = scalar_region(
    ActivationWorkspaceSpace,
    offset,
    byte_count,
    alignment,
  )
  let total = workspace.end_offset_bytes()
  if total > limits.max_activation_workspace_bytes {
    raise rejected(WorkspaceRegion)
  }
  (workspace, total)
}

///|
fn plan_memory(
  model : @plan.ModelPlan,
  contracts : ArrayView[
    @tensor_parallel_launch_contract.TensorParallelAotLaunchContract,
  ],
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  limits : TensorParallelExecutionPlanLimits,
) -> PlanningMemory raise TensorParallelExecutionPlanError {
  let (descriptors, descriptor_bytes) = plan_descriptors(contracts, limits)
  let rank_constants = plan_rank_constants(
    contracts,
    launch_set.rank_constants().version(),
  )
  let facts = collect_activation_facts(
    model,
    contracts,
    launch_set.rank(),
    launch_set.world_size(),
  )
  let slots = assign_activation_slots(facts)
  let (activations, activation_bytes) = place_activations(facts, slots)
  let (workspace, activation_workspace_bytes) = plan_workspace(
    contracts, activation_bytes, limits,
  )
  {
    descriptors,
    descriptor_bytes,
    rank_constants,
    activations,
    activation_workspace_bytes,
    workspace,
  }
}

///|
fn activation_region(
  memory : PlanningMemory,
  reference : @plan.ActivationRef,
  operation : Int,
  operand : Int,
) -> TensorParallelScalarRegion raise TensorParallelExecutionPlanError {
  let index = reference.as_int()
  if index < 0 || index >= memory.activations.length() {
    raise rejected(ActivationRegion, operation~, operand~)
  }
  let binding = memory.activations[index]
  if binding.reference.as_int() != index {
    raise rejected(ActivationRegion, operation~, operand~)
  }
  binding.region
}
