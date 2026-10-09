///|
/// Preserves local topology admission while sharing the resource-free executor
/// join with network workers. Neither path creates backend resources here.
pub fn admit_plan(
  model : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  group_generation : @rank_group_protocol.RankGroupGeneration,
  topology : @device_topology.AdmittedLocalTopology,
  startup : @worker_wire.WorkerStartupContract,
  envelope : @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope,
  inspection : @device_materialize.DeviceShardedWeightFileInspection,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  selected_profile : @launch_contract.PagedKernelProfile,
  execution_limits : @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits,
  kv_plan : @tensor_parallel_kv_plan.TensorParallelKvPlan,
  expected_group_digest : @tensor_parallel_worker_bootstrap.TensorParallelGroupBootstrapDigest,
  expected_rendezvous_id : @device.TensorCollectiveGroupId,
  collective_runtime_version_code~ : Int,
) -> TensorParallelDeviceWorkerPlan raise TensorParallelDeviceWorkerError {
  admit_bound_plan(
    model,
    model_generation,
    group_generation,
    topology.world_size(),
    envelope.rank(),
    startup,
    None,
    inspection,
    device_plan,
    launch_set,
    artifacts,
    selected_profile,
    execution_limits,
    kv_plan,
    (execution, kv_binding) => {
      envelope.authenticate_local(
        model,
        model_generation,
        group_generation,
        topology,
        startup,
        inspection,
        device_plan,
        execution,
        kv_binding,
        expected_group_digest~,
        expected_rendezvous_id~,
        expected_collective_runtime_version_code=collective_runtime_version_code,
      ) catch {
        error => raise Bootstrap(error)
      }
    },
    expected_rendezvous_id,
    collective_runtime_version_code~,
  )
}
