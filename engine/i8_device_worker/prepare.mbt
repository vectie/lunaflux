///|
fn validate_received_i8_startup(
  received : @worker_wire.WorkerStartupContract,
  expected : @worker_wire.WorkerStartupContract,
) -> Unit raise I8DeviceWorkerError {
  if received != expected {
    raise Invalid(StartupContract)
  }
}

///|
fn observe_exact_numeric_capability(
  context : @device.Context,
  plan : I8DeviceWorkerPlan,
) -> @device.DeviceNumericCapability raise I8DeviceWorkerError {
  let manifest = plan.execution.bootstrap_manifest()
  if context.ordinal() != manifest.device_ordinal() {
    raise Invalid(DeviceOrdinal)
  }
  if !manifest.target().matches_capability(context.capability()) {
    raise Invalid(DeviceTarget)
  }
  if context.capability().total_memory_bytes() !=
    plan.expected_total_memory_bytes {
    raise Invalid(DeviceTarget)
  }
  let observed = @device.admit_symmetric_i8_weight_only_per_output_channel_v1(
    context.capability(),
  ) catch {
    _ => raise Invalid(DeviceNumericCapability)
  }
  if observed != plan.execution.device_numeric_capability() ||
    observed.digest() != manifest.device_numeric_digest() {
    raise Invalid(DeviceNumericCapability)
  }
  observed
}

///|
fn load_exact_numeric_weights(
  context : @device.Context,
  model_root : @approved_fs.ApprovedRoot,
  plan : I8DeviceWorkerPlan,
  resources : I8DeviceWorkerResources,
) -> @numeric_weight_file.NumericDeviceWeights raise I8DeviceWorkerError {
  let outcome = @numeric_weight_file.load_inspected_file(
    context,
    model_root,
    plan.inspection,
  ) catch {
    error => raise NumericWeights(error)
  }
  match outcome {
    NumericWeightsReady(weights) => {
      commit_owner_transition(resources, OwnerSlotsEmpty, RawNumericOwner, () => {
        resources.numeric = Some(Raw(weights))
      })
      weights
    }
    NumericWeightsCleanupRequired(failed) => {
      commit_owner_transition(resources, OwnerSlotsEmpty, FailedNumericOwner, () => {
        resources.numeric = Some(Failed(failed))
      })
      raise NumericWeights(failed.cause())
    }
  }
}

///|
fn authenticate_exact_numeric_weights(
  weights : @numeric_weight_file.NumericDeviceWeights,
  plan : I8DeviceWorkerPlan,
  resources : I8DeviceWorkerResources,
) -> @numeric_weight_file.AuthenticatedNumericWeightAuthority raise I8DeviceWorkerError {
  let authority = weights.authenticate(
    expected_identity=plan.model_plan.identity(),
    expected_schema_digest=plan.model_plan.numeric_binding().digest(),
    expected_layout=plan.execution.weight_layout(),
    expected_artifact_digest=plan.execution.weight_artifact_digest(),
  ) catch {
    error => raise NumericWeights(error)
  }
  commit_owner_transition(
    resources,
    RawNumericOwner,
    AuthenticatedNumericOwner,
    () => resources.numeric = Some(Authenticated(authority)),
  )
  authority
}

///|
fn join_exact_loaded_weights(
  authority : @numeric_weight_file.AuthenticatedNumericWeightAuthority,
  plan : I8DeviceWorkerPlan,
  resources : I8DeviceWorkerResources,
) -> @device_step.I8LoadedExecutionBlueprint raise I8DeviceWorkerError {
  let loaded = plan.execution.join_loaded_weights(authority) catch {
    error => raise ExecutionManifest(error)
  }
  commit_owner_transition(
    resources,
    AuthenticatedNumericOwner,
    LoadedNumericOwner,
    () => resources.numeric = Some(Loaded(loaded)),
  )
  loaded
}

///|
fn prepare_exact_i8_executor(
  context : @device.Context,
  transfer : @execution_manifest_file.I8PagedExecutorTransferV4,
  plan : I8DeviceWorkerPlan,
  resources : I8DeviceWorkerResources,
) -> Unit raise I8DeviceWorkerError {
  let generation = @worker_protocol.ModelPlanGeneration::from_wire(
    plan.startup.model_generation_value(),
  ) catch {
    error => raise Protocol(error)
  }
  commit_owner_transition(
    resources,
    ExecutorTransferOwner,
    ExecutorConstructionOwner,
    () => resources.numeric = None,
  )
  let outcome = @execution_manifest_file.prepare_paged_i8_v4_executor(
    context,
    transfer,
    generation,
    has_accepted_predecessor=plan.startup.has_predecessor(),
    accepted_predecessor_value=plan.startup.predecessor_value(),
  ) catch {
    error => {
      commit_owner_transition(
        resources,
        ExecutorConstructionOwner,
        OwnerSlotsEmpty,
        () => (),
      )
      raise ExecutionManifest(error)
    }
  }
  match outcome {
    PagedReady(executor) =>
      commit_owner_transition(
        resources,
        ExecutorConstructionOwner,
        ReadyExecutorOwner,
        () => resources.executor = Some(Complete(executor)),
      )
    PagedCleanupRequired(failed) => {
      commit_owner_transition(
        resources,
        ExecutorConstructionOwner,
        FailedExecutorOwner,
        () => resources.executor = Some(Failed(failed)),
      )
      raise Executor(failed.cause())
    }
  }
}

///|
/// Opens the exact context, re-observes its numeric capability, loads and
/// reauthenticates the retained numeric artifact, joins the four-way authority,
/// and prepares the schema-v4 paged executor before readiness can exist.
pub fn prepare(
  received : @worker_wire.WorkerStartupContract,
  model_root : @approved_fs.ApprovedRoot,
  plan : I8DeviceWorkerPlan,
) -> I8DeviceWorkerPreparation raise I8DeviceWorkerError {
  validate_received_i8_startup(received, plan.startup)
  let resources : I8DeviceWorkerResources = {
    executor: None,
    numeric: None,
    context: None,
    owner_phase: OwnerSlotsEmpty,
  }
  let context = @device.open_context(
    plan.execution.bootstrap_manifest().device_ordinal(),
  ) catch {
    error => raise Device(error)
  }
  resources.context = Some(context)
  ignore(
    observe_exact_numeric_capability(context, plan) catch {
      error => return preparation_failed(resources, error)
    },
  )
  let weights = load_exact_numeric_weights(context, model_root, plan, resources) catch {
    error => return preparation_failed(resources, error)
  }
  let authority = authenticate_exact_numeric_weights(weights, plan, resources) catch {
    error => return preparation_failed(resources, error)
  }
  let loaded = join_exact_loaded_weights(authority, plan, resources) catch {
    error => return preparation_failed(resources, error)
  }
  let transfer = plan.execution.bind_loaded_executor_transfer(loaded) catch {
    error => return preparation_failed(resources, ExecutionManifest(error))
  }
  commit_owner_transition(
    resources,
    LoadedNumericOwner,
    ExecutorTransferOwner,
    () => resources.numeric = Some(ExecutorTransfer(transfer)),
  ) catch {
    error => return preparation_failed(resources, error)
  }
  prepare_exact_i8_executor(context, transfer, plan, resources) catch {
    error => return preparation_failed(resources, error)
  }
  I8DeviceWorkerReady({
    startup: plan.startup,
    resources,
    lifecycle: WorkerReady,
  })
}

///|
/// Sole readiness surface: every exact device/executor dependency must remain
/// live and the owner must not have faulted or begun cleanup.
pub fn I8DeviceWorkerOwner::readiness_contract(
  self : I8DeviceWorkerOwner,
) -> @worker_wire.WorkerStartupContract raise I8DeviceWorkerError {
  if !i8_resources_publishable(
      self.lifecycle,
      self.resources.executor is Some(Complete(_)),
      self.resources.numeric is Some(_),
      self.resources.context is Some(_),
      self.resources.owner_phase == ReadyExecutorOwner,
    ) {
    if self.lifecycle != WorkerReady {
      raise InvalidLifecycle(self.lifecycle)
    }
    raise Invalid(ResourceState)
  }
  self.startup
}
