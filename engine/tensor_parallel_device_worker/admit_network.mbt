///|
/// Binds a remote rank to the same executor used for local tensor parallelism.
/// The caller must authenticate the control channel and enforce the remote
/// lease; this pure admission does not grant network or live device authority.
pub fn admit_network_plan(
  model : @plan.ModelPlan,
  compiled : @two_node_plan.TwoNodePlan,
  remote : @remote_startup.RemoteRankStartup,
  envelope : @tensor_parallel_worker_bootstrap.TensorParallelRankBootstrapEnvelope,
  inspection : @device_materialize.DeviceShardedWeightFileInspection,
  device_plan : @tensor_parallel_device_plan.TensorParallelDevicePlan,
  launch_set : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
  selected_profile : @launch_contract.PagedKernelProfile,
  execution_limits : @tensor_parallel_execution_plan.TensorParallelExecutionPlanLimits,
  kv_plan : @tensor_parallel_kv_plan.TensorParallelKvPlan,
  expected_group_digest : @tensor_parallel_worker_bootstrap.TensorParallelGroupBootstrapDigest,
  collective_runtime_version_code~ : Int,
) -> TensorParallelDeviceWorkerPlan raise TensorParallelDeviceWorkerError {
  let model_generation = @worker_protocol.ModelPlanGeneration::from_wire(
    remote.worker().model_generation_value(),
  ) catch {
    _ => raise Invalid(ModelGeneration)
  }
  let group_generation = @rank_group_protocol.RankGroupGeneration::from_wire(
    remote.group_generation_value(),
  ) catch {
    _ => raise Invalid(GroupGeneration)
  }
  let admitted = admit_bound_plan(
    model,
    model_generation,
    group_generation,
    2,
    remote.rank(),
    remote.worker(),
    Some(remote),
    inspection,
    device_plan,
    launch_set,
    artifacts,
    selected_profile,
    execution_limits,
    kv_plan,
    (execution, _binding) => {
      envelope.authenticate_network(
        model,
        compiled,
        remote,
        inspection,
        device_plan,
        execution,
        kv_plan,
        expected_group_digest~,
        expected_collective_runtime_version_code=collective_runtime_version_code,
      ) catch {
        error => raise Bootstrap(error)
      }
    },
    remote.rendezvous(),
    collective_runtime_version_code~,
  )
  { ..admitted, remote_envelope: Some(envelope), }
}
