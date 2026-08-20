///|
fn exact_startup_contract(
  startup : @worker_wire.WorkerStartupContract,
  manifest : @device_step.DeviceWorkerBootstrapManifest,
) -> @worker_wire.WorkerStartupContract raise DeviceWorkerError {
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    startup.model_generation_value(),
  ) catch {
    error => raise Protocol(error)
  }
  let expected = manifest.worker_startup_contract(
    generation,
    bootstrap_source_digest=startup.bootstrap_source_digest(),
    startup.worker_limits(),
    startup.inference_limits(),
    has_predecessor=startup.has_predecessor(),
    predecessor_value=startup.predecessor_value(),
  ) catch {
    error => raise Bootstrap(error)
  }
  if expected != startup {
    raise Invalid(StartupContract)
  }
  expected
}

///|
fn validate_received_startup(
  received : @worker_wire.WorkerStartupContract,
  expected : @worker_wire.WorkerStartupContract,
) -> Unit raise DeviceWorkerError {
  if received != expected {
    raise Invalid(StartupContract)
  }
}

///|
fn validate_model_identity(
  model : @spec.LlamaModelMetadata,
  execution_identity : @spec.ModelIdentity,
) -> Unit raise DeviceWorkerError {
  let canonical = @spec.ModelIdentity::new(
    model.identity().content(),
    model.spec().canonical_paged_plan_digest(),
  ) catch {
    _ => raise Invalid(ModelIdentity)
  }
  if execution_identity != canonical {
    raise Invalid(ModelIdentity)
  }
}

///|
fn validate_weight_evidence(
  model : @spec.LlamaModelMetadata,
  inspection : @device_materialize.DeviceWeightFileInspection,
  execution : @execution_manifest_file.PagedExecutionAdmission,
) -> Unit raise DeviceWorkerError {
  let inspected_layout = inspection.layout()
  let execution_layout = execution.weight_layout()
  if inspection.content_digest() != model.identity().content() ||
    inspected_layout != execution_layout {
    raise Invalid(WeightEnvelope)
  }
}

///|
/// Admits independently prepared immutable device-worker evidence without
/// opening a filesystem, native, or device resource. The exact weight-file
/// inspection and aggregate paged execution admission are retained privately.
pub fn admit_plan(
  startup~ : @worker_wire.WorkerStartupContract,
  model~ : @spec.LlamaModelMetadata,
  weight_inspection~ : @device_materialize.DeviceWeightFileInspection,
  execution~ : @execution_manifest_file.PagedExecutionAdmission,
  bootstrap_limits~ : @device_step.DeviceBootstrapLimits,
) -> DeviceWorkerPlan raise DeviceWorkerError {
  let blueprint = execution.blueprint()
  validate_model_identity(model, blueprint.identity())
  validate_weight_evidence(model, weight_inspection, execution)
  let manifest = @device_step.admit_device_worker_bootstrap(
    blueprint,
    execution.artifacts(),
    device_ordinal=startup.device_ordinal(),
    bootstrap_limits,
  ) catch {
    error => raise Bootstrap(error)
  }
  let admitted_startup = exact_startup_contract(startup, manifest)
  { startup: admitted_startup, weight_inspection, execution, manifest }
}

///|
/// Requires the received contract to equal the admitted inert plan before any
/// resource opens, then owns the exact configured device, fully re-admits and
/// streams the retained inspected descendant into final device storage, and
/// prepares the complete paged graph before returning readiness.
pub fn prepare(
  received : @worker_wire.WorkerStartupContract,
  model_root : @approved_fs.ApprovedRoot,
  plan : DeviceWorkerPlan,
) -> DeviceWorkerPreparation raise DeviceWorkerError {
  validate_received_startup(received, plan.startup)
  let ready_contract = plan.startup
  let execution = plan.execution
  let resources : DeviceWorkerResources = {
    executor: None,
    weights: None,
    context: None,
  }
  let context = @device.open_context(plan.manifest.device_ordinal()) catch {
    error => raise Device(error)
  }
  resources.context = Some(context)
  if context.ordinal() != plan.manifest.device_ordinal() {
    return preparation_failed(resources, Invalid(DeviceOrdinal))
  }
  if !plan.manifest.target().matches_capability(context.capability()) {
    return preparation_failed(resources, Invalid(DeviceTarget))
  }
  let weight_outcome = @device_materialize.load_inspected_file(
    context,
    model_root,
    plan.weight_inspection,
  ) catch {
    error => return preparation_failed(resources, Weights(error))
  }
  let weights = match weight_outcome {
    WeightsReady(owner) => {
      resources.weights = Some(Complete(owner))
      owner
    }
    WeightsCleanupRequired(failed) => {
      resources.weights = Some(Failed(failed))
      return preparation_failed(resources, Weights(failed.cause()))
    }
  }
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    plan.startup.model_generation_value(),
  ) catch {
    error => return preparation_failed(resources, Protocol(error))
  }
  let executor_outcome = @device_step.prepare_paged_graph_executor(
    context,
    execution.model_plan(),
    generation,
    execution.layout(),
    execution.device_step_limits(),
    execution.static_plan(),
    execution.memory_plan(),
    execution.profile_id(),
    weights,
    execution.contracts(),
    execution.artifacts(),
    has_accepted_predecessor=plan.startup.has_predecessor(),
    accepted_predecessor_value=plan.startup.predecessor_value(),
  ) catch {
    error => return preparation_failed(resources, Executor(error))
  }
  match executor_outcome {
    PagedReady(executor) => resources.executor = Some(Complete(executor))
    PagedCleanupRequired(failed) => {
      resources.executor = Some(Failed(failed))
      return preparation_failed(resources, Executor(failed.cause()))
    }
  }
  DeviceWorkerReady({
    startup: ready_contract,
    resources,
    lifecycle: WorkerReady,
  })
}

///|
/// Returns the exact configured contract only while every owned device/model/
/// executor resource is live. This is the sole readiness publication surface.
pub fn DeviceWorkerOwner::readiness_contract(
  self : DeviceWorkerOwner,
) -> @worker_wire.WorkerStartupContract raise DeviceWorkerError {
  if self.lifecycle != WorkerReady {
    raise InvalidLifecycle(self.lifecycle)
  }
  if !resources_complete(self.resources) {
    raise Invalid(ResourceState)
  }
  self.startup
}
