///|
fn prepare_owned_tensor_parallel_luna_online_instance_with_clock(
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  blueprint : @core.SchedulerBlueprint,
  inputs : @tensor_parallel_group_transport.TensorParallelGroupPlanningInputs,
  approved_executable : @worker_executable_file.WorkerExecutableAdmission,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  clock : @monotonic_clock.MonotonicClock,
  restart_policy : @worker_service.WorkerRestartBackoffPolicy,
) -> LunaOnlineInstancePreparation raise LunaOnlineInstancePreparationError {
  if inputs.inference_limits() != inference {
    raise luna_preparation_failure(
      LunaOnlineInvalid(LunaOnlineInferenceEnvelope),
    )
  }
  if expected_model != blueprint.model_identity() {
    raise luna_preparation_failure(LunaOnlineInvalid(LunaOnlineModelIdentity))
  }
  let (instance, prepared) = new_pending_luna_online_preparation(
    expected_tokenizer, expected_model, inference, blueprint,
  )
  let owned = @worker_service.prepare_owned_tensor_parallel_approved(
    blueprint, inputs, approved_executable, model_root, kernel_root, restart_policy,
  ) catch {
    _ => raise luna_preparation_failure(LunaOnlineServicePreparationFailure)
  }
  publish_owned_luna_online_preparation(instance, prepared, owned, clock)
}

///|
/// Converges the existing rank-group transport owner on the same persistent
/// online instance used by the single-rank route. No listener or readiness
/// authority is published unless the complete rank group is ready.
pub fn prepare_owned_tensor_parallel_luna_online_instance_approved(
  expected_tokenizer : @tokenizer.TokenizerDigest,
  expected_model : @spec.ModelIdentity,
  inference : @inference.InferenceLimits,
  blueprint : @core.SchedulerBlueprint,
  inputs : @tensor_parallel_group_transport.TensorParallelGroupPlanningInputs,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
  restart_policy : @worker_service.WorkerRestartBackoffPolicy,
) -> LunaOnlineInstancePreparation raise LunaOnlineInstancePreparationError {
  prepare_owned_tensor_parallel_luna_online_instance_with_clock(
    expected_tokenizer,
    expected_model,
    inference,
    blueprint,
    inputs,
    executable,
    model_root,
    kernel_root,
    @monotonic_clock.MonotonicClock::system(),
    restart_policy,
  )
}
