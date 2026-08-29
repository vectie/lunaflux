///|
fn prepare_tensor_parallel_group_transport(
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  inputs : TensorParallelGroupPlanningInputs,
  plan_buffer_identities : ArrayView[@worker_protocol.PlanBufferIdentity],
  previous_plan_sequence_value~ : UInt64,
) -> TensorParallelGroupTransport raise TensorParallelGroupTransportError {
  let executable = @worker_executable_fixture.admit(
    inputs.admitted_rank_child_activation_path,
  ) catch {
    _ => rejected(Process)
  }
  prepare_tensor_parallel_group_transport_approved(
    model_root,
    kernel_root,
    inputs,
    executable,
    plan_buffer_identities,
    previous_plan_sequence_value~,
  )
}
