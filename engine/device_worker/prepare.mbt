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
  )
  if execution_identity != canonical {
    raise Invalid(ModelIdentity)
  }
}

///|
fn admit_model_locator(
  locator : StringView,
) -> @approved_fs.ApprovedRelativeLocator raise DeviceWorkerError {
  @approved_fs.ApprovedRelativeLocator::new(locator) catch {
    _ => raise Invalid(ModelLocator)
  }
}

///|
/// Admits all immutable device-worker evidence without opening a native or
/// device resource. The strict relative model locator is copied into plan-owned
/// startup storage; approved-root resolution occurs during preparation.
pub fn admit_plan(
  startup~ : @worker_wire.WorkerStartupContract,
  model~ : @spec.LlamaModelMetadata,
  model_locator~ : StringView,
  weight_file_limits~ : @device_materialize.DeviceWeightFileLimits,
  weight_alignment_bytes~ : Int64,
  max_weight_arena_bytes~ : Int64,
  model_plan~ : @plan.ModelPlan,
  layout~ : @device_layout.DeviceKvLayout,
  device_step_limits~ : @device_step.DeviceStepLimits,
  static_plan~ : @device_plan.StaticDevicePlan,
  memory_plan~ : @device_memory.DeviceMemoryPlan,
  profile_id~ : @launch_contract.KernelProfileId,
  contracts~ : @launch_contract.PagedKvLaunchContractSet,
  artifacts~ : @artifact.KernelArtifactBundle,
  bootstrap_limits~ : @device_step.DeviceBootstrapLimits,
) -> DeviceWorkerPlan raise DeviceWorkerError {
  let admitted_model_locator = admit_model_locator(model_locator)
  let blueprint = @device_step.admit_paged_graph_execution_blueprint(
    model_plan, device_step_limits, layout, static_plan, memory_plan, profile_id,
    contracts, artifacts,
  ) catch {
    error => raise Executor(error)
  }
  validate_model_identity(model, blueprint.identity())
  if weight_alignment_bytes != blueprint.weight_alignment() ||
    max_weight_arena_bytes < blueprint.weight_arena_bytes() {
    raise Invalid(WeightEnvelope)
  }
  let manifest = @device_step.admit_device_worker_bootstrap(
    blueprint,
    artifacts,
    device_ordinal=startup.device_ordinal(),
    bootstrap_limits,
  ) catch {
    error => raise Bootstrap(error)
  }
  let admitted_startup = exact_startup_contract(startup, manifest)
  {
    startup: admitted_startup,
    model,
    model_locator: admitted_model_locator,
    weight_file_limits,
    weight_alignment_bytes,
    max_weight_arena_bytes,
    model_plan,
    layout,
    device_step_limits,
    static_plan,
    memory_plan,
    profile_id,
    contracts,
    artifacts,
    manifest,
  }
}

///|
/// Inspects the approved model descendant before opening a device resource,
/// then owns the exact configured device, fully re-admits and streams that
/// descendant into final device storage, and prepares the complete paged graph
/// before returning readiness. The received startup contract is compared with
/// the admitted inert plan before any filesystem or device resource opens.
pub fn prepare(
  received : @worker_wire.WorkerStartupContract,
  model_root : @approved_fs.ApprovedRoot,
  plan : DeviceWorkerPlan,
) -> DeviceWorkerPreparation raise DeviceWorkerError {
  validate_received_startup(received, plan.startup)
  let ready_contract = plan.startup
  let weight_inspection = @device_materialize.inspect_file(
    model_root,
    plan.model,
    plan.model_locator,
    limits=plan.weight_file_limits,
    alignment_bytes=plan.weight_alignment_bytes,
    max_arena_bytes=plan.max_weight_arena_bytes,
  ) catch {
    error => raise Weights(error)
  }
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
    context, model_root, weight_inspection,
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
    plan.model_plan,
    generation,
    plan.layout,
    plan.device_step_limits,
    plan.static_plan,
    plan.memory_plan,
    plan.profile_id,
    weights,
    plan.contracts,
    plan.artifacts,
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
