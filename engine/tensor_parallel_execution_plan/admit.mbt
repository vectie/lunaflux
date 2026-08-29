///|
/// Derives one immutable rank-local tensor-parallel physical execution plan.
///
/// Admission cross-authenticates semantic graph, rank/world/device evidence,
/// selected AOT profile, exact modules and symbols, sharded weight regions,
/// local KV geometry, operand spaces, and collective order. It performs only
/// bounded startup allocation. It opens no device, module, communicator,
/// filesystem authority, or backend handle.
pub fn admit(
  model : @plan.ModelPlan,
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  artifacts : @artifact.KernelArtifactBundle,
  selected_profile : @launch_contract.PagedKernelProfile,
  limits : TensorParallelExecutionPlanLimits,
) -> TensorParallelExecutionPlan raise TensorParallelExecutionPlanError {
  validate_identity_surface(model, launch_set, device_plan, artifacts, limits)
  validate_device_tensors(device_plan)
  let contracts = select_contracts(
    model, launch_set, device_plan, selected_profile, limits,
  )
  let memory = plan_memory(model, contracts, launch_set, limits)
  let operations = build_operation_bindings(
    contracts, launch_set, device_plan, artifacts, memory,
  )
  let collectives = build_collective_bindings(
    model, contracts, launch_set, device_plan, memory, limits,
  )
  let provisional : TensorParallelExecutionPlan = {
    identity: model.identity(),
    rank: launch_set.rank(),
    world_size: launch_set.world_size(),
    device_ordinal: device_plan.device_ordinal(),
    target: launch_set.device_target(),
    catalog_version: launch_set.catalog_version(),
    profile: selected_profile,
    digest: { sha256: "" },
    descriptor_bytes: memory.descriptor_bytes,
    rank_constants: memory.rank_constants,
    weight_arena_bytes: device_plan.arena_bytes(),
    kv_arena_bytes: launch_set.device_kv_layout().arena_bytes(),
    activation_workspace_bytes: memory.activation_workspace_bytes,
    workspace: memory.workspace,
    descriptors: memory.descriptors,
    activations: memory.activations,
    operations,
    collectives,
  }
  {
    ..provisional,
    digest: derive_digest(
      provisional, model, launch_set, contracts, device_plan, artifacts,
    ),
  }
}
