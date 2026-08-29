///|
pub(all) enum I8ReleaseWorkerRule {
  SourceRecipe
  ModelIdentity
  BootstrapIdentity
  BootstrapSourceIdentity
  DeviceAssignment
  ModelGeneration
  WorkerEnvelope
  InferenceEnvelope
  NumericWeightEnvelope
} derive(Debug, Eq)

///|
pub(all) suberror I8ReleaseWorkerError {
  Invalid(I8ReleaseWorkerRule)
  Prepare(@worker_service.OwnedWorkerServicePreparationError)
  OnlinePrepare(@online_session.LunaOnlineInstancePreparationError)
} derive(Debug, Eq)

///|
fn validate_release_join(
  instance : @instance_admission.I8RuntimeInstanceAdmission,
) -> Unit raise I8ReleaseWorkerError {
  let runtime = instance.runtime_descriptor()
  let blueprint = instance.scheduler_blueprint()
  let binding = instance.worker_binding()
  let source = runtime.bootstrap_source()
  let startup = runtime.startup_contract()
  let execution = runtime.execution()
  let inspection = runtime.weight_inspection()
  if source.recipe() != DenseLlamaI8PagedAotV6 ||
    source.i8_model_source() is None ||
    source.i8_execution() is None {
    raise Invalid(SourceRecipe)
  }
  if source.digest() != startup.bootstrap_source_digest() ||
    binding.bootstrap_source_digest() != startup.bootstrap_source_digest() {
    raise Invalid(BootstrapSourceIdentity)
  }
  if runtime.bootstrap_digest() != startup.bootstrap_digest() ||
    binding.bootstrap_digest() != startup.bootstrap_digest() ||
    execution.bootstrap_manifest().digest() != startup.bootstrap_digest() {
    raise Invalid(BootstrapIdentity)
  }
  if runtime.model_identity() != startup.identity() ||
    runtime.model_identity() != blueprint.model_identity() ||
    execution.model_plan().identity() != startup.identity() ||
    inspection.identity() != startup.identity() {
    raise Invalid(ModelIdentity)
  }
  if inspection.layout() != execution.weight_layout() ||
    inspection.numeric_schema_digest() !=
    execution.model_plan().numeric_binding().digest() ||
    inspection.artifact_digest() != execution.weight_artifact_digest() {
    raise Invalid(NumericWeightEnvelope)
  }
  if runtime.device_ordinal() != startup.device_ordinal() ||
    binding.device_ordinal() != startup.device_ordinal() ||
    execution.bootstrap_manifest().device_ordinal() != startup.device_ordinal() ||
    runtime.target() != execution.bootstrap_manifest().target() {
    raise Invalid(DeviceAssignment)
  }
  if blueprint.model_plan_generation().value() !=
    startup.model_generation_value() {
    raise Invalid(ModelGeneration)
  }
  if blueprint.worker_protocol_limits() != startup.worker_limits() {
    raise Invalid(WorkerEnvelope)
  }
  if binding.inference_limits() != startup.inference_limits() {
    raise Invalid(InferenceEnvelope)
  }
}

///|
priv struct RetainedReleaseInputs[B, E] {
  blueprint : B
  executable : E
}

///|
fn[B, E] retain_exact_release_inputs(
  blueprint : B,
  executable : E,
) -> RetainedReleaseInputs[B, E] {
  { blueprint, executable }
}

///|
/// Starts the existing rooted, restartable, serialized worker-service path
/// only from one exact I8 runtime-instance admission and one independently
/// verified opaque worker-executable admission.
///
/// The child independently decodes recipe v6, rebuilds and reauthenticates the
/// numeric artifact, re-observes the physical context, and publishes readiness
/// only after complete executor construction. This host wiring is not physical
/// CUDA correctness, sanitizer, leak, soak, or benchmark evidence.
pub fn prepare_owned(
  instance : @instance_admission.I8RuntimeInstanceAdmission,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> @worker_service.OwnedWorkerServicePreparation raise I8ReleaseWorkerError {
  validate_release_join(instance)
  let runtime = instance.runtime_descriptor()
  let retained = retain_exact_release_inputs(
    instance.scheduler_blueprint(),
    executable,
  )
  @worker_service.prepare_owned_approved(
    retained.blueprint,
    instance.worker_binding(),
    retained.executable,
    runtime.startup_contract(),
    runtime.bootstrap_source(),
    instance.worker_process_limits(),
    model_root,
    kernel_root,
    instance.restart_policy(),
  ) catch {
    error => raise Prepare(error)
  }
}

///|
/// Prepares the sole existing online framed scheduler/service/worker
/// composition from the same two opaque admissions. Every scheduler,
/// tokenizer, protocol, source, process, and restart input is derived from the
/// retained I8 instance; callers cannot substitute a raw downstream value.
pub fn prepare_owned_online_framed(
  instance : @instance_admission.I8RuntimeInstanceAdmission,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> @online_session.LunaOnlineFramedServicePreparation raise I8ReleaseWorkerError {
  validate_release_join(instance)
  let runtime = instance.runtime_descriptor()
  let retained = retain_exact_release_inputs(
    instance.scheduler_blueprint(),
    executable,
  )
  @online_session.prepare_owned_luna_online_framed_service_approved(
    instance.tokenizer(),
    instance.tokenizer_digest(),
    runtime.model_identity(),
    instance.worker_binding().inference_limits(),
    instance.framed_limits(),
    instance.preparation_lane_count(),
    instance.preparation_step_budget(),
    instance.preparation_work_limit(),
    instance.preparation_storage_budget(),
    instance.event_step_budget(),
    retained.blueprint,
    instance.worker_binding(),
    retained.executable,
    runtime.startup_contract(),
    runtime.bootstrap_source(),
    instance.worker_process_limits(),
    model_root,
    kernel_root,
    instance.restart_policy(),
  ) catch {
    error => raise OnlinePrepare(error)
  }
}
