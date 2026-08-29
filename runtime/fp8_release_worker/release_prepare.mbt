///|
pub(all) enum Fp8ReleaseWorkerRule {
  SourceRecipe
  ModelIdentity
  BootstrapIdentity
  BootstrapSourceIdentity
  DeviceAssignment
  ModelGeneration
  WorkerEnvelope
  InferenceEnvelope
  ReusableRecipeIdentity
  ReleaseAuthorityIdentity
} derive(Debug, Eq)

///|
pub(all) suberror Fp8ReleaseWorkerError {
  Invalid(Fp8ReleaseWorkerRule)
  Prepare(@worker_service.OwnedWorkerServicePreparationError)
  OnlinePrepare(@online_session.LunaOnlineInstancePreparationError)
} derive(Debug, Eq)

///|
fn validate_release_join_v3(
  instance : @instance_admission.Fp8RuntimeInstanceAdmission,
) -> Unit raise Fp8ReleaseWorkerError {
  let runtime = instance.runtime_descriptor()
  let source = runtime.bootstrap_source()
  let startup = runtime.startup_contract()
  let execution = runtime.execution()
  guard source.fp8_execution() is Some(source_execution) else {
    raise Invalid(SourceRecipe)
  }
  if source.recipe() != DenseLlamaFp8ReusablePagedAotApprovedV9 ||
    source.kernel().luna_approval() is None {
    raise Invalid(SourceRecipe)
  }
  if source.digest() != startup.bootstrap_source_digest() ||
    instance.worker_binding().bootstrap_source_digest() !=
    startup.bootstrap_source_digest() {
    raise Invalid(BootstrapSourceIdentity)
  }
  if runtime.bootstrap_digest() != startup.bootstrap_digest() ||
    instance.worker_binding().bootstrap_digest() != startup.bootstrap_digest() {
    raise Invalid(BootstrapIdentity)
  }
  if runtime.model_identity() != startup.identity() ||
    execution.model_plan().identity() != startup.identity() ||
    execution.weight_inspection().identity() != startup.identity() ||
    instance.scheduler_blueprint().model_identity() != startup.identity() {
    raise Invalid(ModelIdentity)
  }
  if source_execution.reusable_recipe_digest() !=
    execution.reusable_recipe().digest().as_hex() {
    raise Invalid(ReusableRecipeIdentity)
  }
  if startup.bootstrap_digest().as_hex() !=
    execution.release_authority().digest().as_hex() {
    raise Invalid(ReleaseAuthorityIdentity)
  }
  if runtime.device_ordinal() != startup.device_ordinal() {
    raise Invalid(DeviceAssignment)
  }
  if instance.scheduler_blueprint().model_plan_generation().value() !=
    startup.model_generation_value() {
    raise Invalid(ModelGeneration)
  }
  if instance.scheduler_blueprint().worker_protocol_limits() !=
    startup.worker_limits() {
    raise Invalid(WorkerEnvelope)
  }
  if instance.worker_binding().inference_limits() != startup.inference_limits() {
    raise Invalid(InferenceEnvelope)
  }
}

///|
/// Starts the existing rooted/restartable service from exact v3 authority.
pub fn prepare_owned_online_framed_v3(
  instance : @instance_admission.Fp8RuntimeInstanceAdmission,
  executable : @worker_executable_file.WorkerExecutableAdmission,
  model_root : @approved_fs.ApprovedRoot,
  kernel_root : @approved_fs.ApprovedRoot,
) -> @online_session.LunaOnlineFramedServicePreparation raise Fp8ReleaseWorkerError {
  validate_release_join_v3(instance)
  let runtime = instance.runtime_descriptor()
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
    instance.scheduler_blueprint(),
    instance.worker_binding(),
    executable,
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
