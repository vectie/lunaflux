///|
fn fail_preparation_v3(
  resources : Fp8DeviceWorkerV3Resources,
  cause : Fp8DeviceWorkerV3Error,
) -> Fp8DeviceWorkerPreparationV3 raise Fp8DeviceWorkerV3Error {
  try close_resources_v3(resources) catch {
    cleanup_failure =>
      Fp8DeviceWorkerCleanupRequiredV3({
        resources,
        cause,
        cleanup_failure,
        lifecycle: WorkerClosingFailed,
      })
  } noraise {
    _ => raise cause
  }
}

///|
pub fn prepare_v3(
  received : @worker_wire.WorkerStartupContract,
  model_root : @approved_fs.ApprovedRoot,
  plan : Fp8DeviceWorkerPlanV3,
) -> Fp8DeviceWorkerPreparationV3 raise Fp8DeviceWorkerV3Error {
  if received != plan.startup {
    raise Invalid(StartupContract)
  }
  let resources : Fp8DeviceWorkerV3Resources = {
    executor: None,
    numeric: None,
    context: None,
  }
  let context = @device.open_context(plan.startup.device_ordinal()) catch {
    error => raise Device(error)
  }
  resources.context = Some(context)
  let execution = plan.execution
  if context.ordinal() != plan.startup.device_ordinal() {
    return fail_preparation_v3(resources, Invalid(DeviceOrdinal))
  }
  if context.capability().total_memory_bytes() !=
    plan.expected_total_memory_bytes ||
    !execution
    .release_authority()
    .target()
    .matches_capability(context.capability()) {
    return fail_preparation_v3(resources, Invalid(DeviceTarget))
  }
  let observed = @device.admit_finite_fp8_e4m3_w8a8(context.capability()) catch {
    _ => return fail_preparation_v3(resources, Invalid(DeviceNumericCapability))
  }
  if observed.digest() !=
    execution.reusable_recipe().startup().device_numeric_capability().digest() {
    return fail_preparation_v3(resources, Invalid(DeviceNumericCapability))
  }
  let loaded = @numeric_weight_file.load_inspected_file(
    context,
    model_root,
    execution.weight_inspection(),
  ) catch {
    error => return fail_preparation_v3(resources, NumericWeights(error))
  }
  let raw = match loaded {
    NumericWeightsReady(owner) => {
      resources.numeric = Some(Raw(owner))
      owner
    }
    NumericWeightsCleanupRequired(failed) => {
      resources.numeric = Some(Failed(failed))
      return fail_preparation_v3(resources, NumericWeights(failed.cause()))
    }
  }
  let authority = raw.authenticate(
    expected_identity=execution.model_plan().identity(),
    expected_schema_digest=execution.model_plan().numeric_binding().digest(),
    expected_layout=execution.weight_inspection().layout(),
    expected_artifact_digest=execution.weight_inspection().artifact_digest(),
  ) catch {
    error => return fail_preparation_v3(resources, NumericWeights(error))
  }
  resources.numeric = Some(Authenticated(authority))
  let blueprint = @device_step.admit_fp8_paged_graph_execution_blueprint_v3(
    execution.release_authority(),
    execution.device_step_limits(),
    execution.static_plan(),
    execution.memory_plan(),
    execution.contracts(),
  ) catch {
    error => return fail_preparation_v3(resources, Executor(error))
  }
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    plan.startup.model_generation_value(),
  ) catch {
    error => return fail_preparation_v3(resources, Protocol(error))
  }
  // Ownership transfers into the executor construction on every outcome.
  resources.numeric = None
  let outcome = @device_step.prepare_fp8_paged_graph_executor_v3(
    context,
    generation,
    blueprint,
    authority,
    has_accepted_predecessor=plan.startup.has_predecessor(),
    accepted_predecessor_value=plan.startup.predecessor_value(),
  ) catch {
    error => return fail_preparation_v3(resources, Executor(error))
  }
  match outcome {
    PagedReady(executor) => resources.executor = Some(Complete(executor))
    PagedCleanupRequired(failed) => {
      resources.executor = Some(Failed(failed))
      return fail_preparation_v3(resources, Executor(failed.cause()))
    }
  }
  Fp8DeviceWorkerReadyV3({
    startup: plan.startup,
    resources,
    lifecycle: WorkerReady,
  })
}

///|
pub fn Fp8DeviceWorkerOwnerV3::readiness_contract(
  self : Self,
) -> @worker_wire.WorkerStartupContract raise Fp8DeviceWorkerV3Error {
  if !fp8_resources_publishable_v3(
      self.lifecycle,
      self.resources.executor is Some(Complete(_)),
      self.resources.numeric is Some(_),
      self.resources.context is Some(_),
    ) {
    raise Invalid(ResourceState)
  }
  self.startup
}

///|
fn fp8_resources_publishable_v3(
  lifecycle : Fp8DeviceWorkerV3Lifecycle,
  executor_complete : Bool,
  numeric_pending : Bool,
  context_present : Bool,
) -> Bool {
  lifecycle == WorkerReady &&
  executor_complete &&
  !numeric_pending &&
  context_present
}
