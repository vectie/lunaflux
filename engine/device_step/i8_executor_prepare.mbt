///|
fn I8LoadedExecutionBlueprint::authenticated_allocation(
  self : I8LoadedExecutionBlueprint,
) -> @device.Allocation raise DeviceStepError {
  self.weights.allocation() catch {
    _ => raise NumericWeightClose
  }
}

///|
fn validate_i8_executor_evidence(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  loaded : I8LoadedExecutionBlueprint,
) -> @device.Allocation raise DeviceStepError {
  let blueprint = loaded.blueprint()
  let artifacts = blueprint.artifacts
  if !blueprint.target().matches_capability(context.capability()) {
    raise InvalidExecutor(ContextTarget)
  }
  let observed = @device.admit_symmetric_i8_weight_only_per_output_channel_v1(
    context.capability(),
  ) catch {
    _ => raise InvalidExecutor(ContextTarget)
  }
  if observed.digest() != device_numeric.digest() ||
    observed.digest() != blueprint.device_numeric_digest() {
    raise InvalidExecutor(ContextTarget)
  }
  if model_plan.identity() != blueprint.identity() ||
    model_plan.numeric_binding().digest() != blueprint.numeric_schema_digest() ||
    artifacts.model_identity() != blueprint.identity() ||
    artifacts.device_target() != blueprint.target() ||
    artifacts.catalog_version() != blueprint.catalog_version() {
    raise InvalidExecutor(WeightIdentity)
  }
  let weight = loaded.authenticated_allocation()
  let layout = blueprint.weight_layout()
  if weight.byte_count() != layout.arena_bytes() ||
    layout.length() != model_plan.tensor_count() {
    raise InvalidExecutor(WeightLayout)
  }
  for index in 0..<layout.length() {
    let reference = @plan.TensorRef::new(index) catch {
      _ => raise InvalidExecutor(WeightLayout)
    }
    guard layout.region(reference) is Some(region) &&
      region.reference() == reference &&
      region.offset_bytes() >= 0L &&
      region.byte_length() > 0L &&
      region.end_offset_bytes() <= weight.byte_count() else {
      raise InvalidExecutor(WeightRegion)
    }
    context.validate_allocation_region(
      weight,
      offset=region.offset_bytes(),
      byte_count=region.byte_length(),
      alignment=layout.alignment_bytes(),
    ) catch {
      error => raise ExecutorDevice(WeightLease, error)
    }
  }
  for step in blueprint.steps() {
    for operand in step.operands() {
      if operand.space() == WeightArena {
        let reference = match operand.role() {
          WeightInput(reference) => reference
          WeightScaleInput(_, scale) => scale.tensor()
          _ => raise executor_operation_error(step.operation_id(), WeightRegion)
        }
        guard layout.region(reference) is Some(region) &&
          region.offset_bytes() == operand.offset_bytes() &&
          region.byte_length() == operand.byte_count() else {
          raise executor_operation_error(step.operation_id(), WeightRegion)
        }
      }
    }
  }
  weight
}

///|
fn collect_i8_paged_functions(
  blueprint : I8PagedExecutionBlueprint,
) -> (ReadOnlyArray[PagedFunctionBlueprint], FixedArray[Int]) {
  let functions : Array[PagedFunctionBlueprint] = []
  let indices = FixedArray::make(blueprint.steps().length(), 0)
  for step_index, step in blueprint.steps() {
    match paged_function_index(functions, step.entry_point()) {
      Some(index) => indices[step_index] = index
      None => {
        functions.push({
          identity: step.entry_point(),
          module_index: step.artifact_module_index(),
          symbol: step.function_symbol(),
        })
        indices[step_index] = functions.length() - 1
      }
    }
  }
  (ReadOnlyArray::from_array(functions), indices)
}

///|
fn validate_i8_physical_regions(
  context : @device.Context,
  blueprint : I8PagedExecutionBlueprint,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation_workspace : @device.Allocation,
  persistent_kv : @device.Allocation,
) -> Unit raise DeviceStepError {
  for step in blueprint.steps() {
    for binding in step.operands() {
      context.validate_allocation_region(
        allocation_for_paged_operand(
          binding, descriptor, weight, activation_workspace, persistent_kv,
        ),
        offset=binding.offset_bytes(),
        byte_count=binding.byte_count(),
        alignment=binding.alignment(),
      ) catch {
        error => raise ExecutorDevice(KernelArguments, error)
      }
    }
  }
}

///|
fn prepare_i8_paged_steps(
  blueprint : I8PagedExecutionBlueprint,
  function_indices : FixedArray[Int],
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation_workspace : @device.Allocation,
  persistent_kv : @device.Allocation,
) -> ReadOnlyArray[PreparedPagedGraphStep] raise DeviceStepError {
  let prepared : Array[PreparedPagedGraphStep] = []
  for step_index, step in blueprint.steps() {
    let arguments : Array[@device.KernelArgument] = []
    for binding in step.operands() {
      arguments.push(
        @device.KernelArgument::device_region(
          allocation_for_paged_operand(
            binding, descriptor, weight, activation_workspace, persistent_kv,
          ),
          offset=binding.offset_bytes(),
          byte_count=binding.byte_count(),
          alignment=binding.alignment(),
        ) catch {
          error => raise ExecutorDevice(KernelArguments, error)
        },
      )
    }
    let source = step.dimensions()
    let dimensions = @device.KernelLaunchDimensions::new(
      grid_x=source.grid_x(),
      grid_y=source.grid_y(),
      grid_z=source.grid_z(),
      block_x=source.block_x(),
      block_y=source.block_y(),
      block_z=source.block_z(),
      shared_memory_bytes=source.shared_memory_bytes(),
    ) catch {
      _ => raise executor_operation_error(step.operation_id(), LaunchDimensions)
    }
    prepared.push({
      operation_id: step.operation_id(),
      function_index: function_indices[step_index],
      dimensions,
      arguments: @device.KernelArguments::new(arguments) catch {
        error => raise ExecutorDevice(KernelArguments, error)
      },
    })
  }
  ReadOnlyArray::from_array(prepared)
}

///|
/// Runs the exact ordered construction stages used by the production I8
/// executor. Native resource values stay captured by the caller; this narrow
/// seam exists so host tests can force every boundary without manufacturing a
/// CUDA context or exposing handles through the public API.
fn[Failure] run_i8_construction_stages(
  stages : ReadOnlyArray[() -> Unit raise DeviceStepError],
  on_failure : (DeviceStepError) -> Failure raise DeviceStepError,
) -> Failure? raise DeviceStepError {
  for stage in stages {
    try stage() catch {
      error => return Some(on_failure(error))
    } noraise {
      _ => ()
    }
  }
  None
}

///|
/// Prepares the software I8 paged executor from exact schema-v4 evidence and
/// the live authenticated weight owner. This establishes host-side resource
/// and launch wiring only; physical CUDA correctness/readiness remains a
/// separate release gate.
pub fn prepare_i8_paged_graph_executor_v1(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  model_generation : @worker_protocol.ModelPlanGeneration,
  device_numeric : @device.DeviceNumericCapability,
  loaded : I8LoadedExecutionBlueprint,
  has_accepted_predecessor~ : Bool,
  accepted_predecessor_value~ : UInt64,
  graph_policy? : PagedGraphExecutionPolicy = PagedOrderedEagerOnly,
) -> PagedGraphPreparation raise DeviceStepError {
  let resources = empty_paged_resources()
  // Ownership transfers at function entry. Every later admission or
  // construction failure therefore closes this exact owner here or returns
  // the retryable failed aggregate that retains it.
  resources.numeric_weights = Some(I8NumericWeights(loaded))
  let evidence = admit_i8_owned_preparation_evidence(
    context,
    model_plan,
    device_numeric,
    loaded,
    has_accepted_predecessor~,
    accepted_predecessor_value~,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let blueprint = evidence.blueprint
  let artifacts = blueprint.artifacts
  let weight = evidence.weight
  let completion = evidence.completion
  let functions = evidence.functions
  let function_indices = evidence.function_indices
  let graph_bucket_table = new_paged_graph_bucket_table(blueprint.profile())
  let descriptor : Ref[DeviceStepOwner?] = Ref(None)
  let activation_workspace : Ref[@device.Allocation?] = Ref(None)
  let persistent_kv : Ref[@device.Allocation?] = Ref(None)
  let prepared_steps : Ref[ReadOnlyArray[PreparedPagedGraphStep]?] = Ref(None)
  let ordered_dispatch : Ref[PagedOrderedExecutorDispatch?] = Ref(None)
  let stages : Array[() -> Unit raise DeviceStepError] = [
    () => {
      let outcome = prepare(
        context,
        model_plan,
        model_generation,
        blueprint.layout(),
        blueprint.limits(),
        has_accepted_predecessor~,
        accepted_predecessor_value~,
      )
      match outcome {
        Ready(owner) => {
          descriptor.val = Some(owner)
          resources.descriptor = Some(Complete(owner))
        }
        CleanupRequired(failed) => {
          resources.descriptor = Some(Failed(failed))
          raise failed.cause()
        }
      }
    },
    () => {
      let lease = context.lease_allocation(weight) catch {
        error => raise ExecutorDevice(WeightLease, error)
      }
      resources.weight_lease = Some({ resource: lease, open: true })
    },
    () => {
      let stream = context.create_stream() catch {
        error => raise ExecutorDevice(Stream, error)
      }
      resources.stream = Some({ resource: stream, open: true })
    },
    () => {
      let allocation = context.allocate(
        blueprint.activation_workspace_arena_bytes(),
      ) catch {
        error => raise ExecutorDevice(ActivationWorkspaceArena, error)
      }
      activation_workspace.val = Some(allocation)
      resources.activation_workspace = Some({ resource: allocation, open: true })
    },
    () => {
      let allocation = context.allocate(blueprint.layout().arena_bytes()) catch {
        error => raise ExecutorDevice(PersistentKvArena, error)
      }
      persistent_kv.val = Some(allocation)
      resources.persistent_kv = Some({ resource: allocation, open: true })
    },
    () => {
      guard descriptor.val is Some(descriptor_owner) &&
        activation_workspace.val is Some(activation) &&
        persistent_kv.val is Some(kv) else {
        raise InvalidExecutor(ResourceState)
      }
      if activation.byte_count() != blueprint.activation_workspace_arena_bytes() ||
        kv.byte_count() != blueprint.layout().arena_bytes() {
        raise InvalidExecutor(AllocationSize)
      }
      validate_i8_physical_regions(
        context,
        blueprint,
        descriptor_owner.resources,
        weight,
        activation,
        kv,
      )
    },
    () => load_paged_modules(context, artifacts, resources),
    () => load_paged_functions(functions, resources),
    () => {
      guard descriptor.val is Some(descriptor_owner) &&
        activation_workspace.val is Some(activation) &&
        persistent_kv.val is Some(kv) else {
        raise InvalidExecutor(ResourceState)
      }
      prepared_steps.val = Some(
        prepare_i8_paged_steps(
          blueprint,
          function_indices,
          descriptor_owner.resources,
          weight,
          activation,
          kv,
        ),
      )
    },
    () => {
      guard prepared_steps.val is Some(steps) else {
        raise InvalidExecutor(ResourceState)
      }
      ordered_dispatch.val = Some(
        prepare_paged_ordered_executors(
          context,
          steps,
          resources,
          graph_policy,
          graph_bucket_table,
          blueprint.profile(),
        ),
      )
    },
  ]
  match
    run_i8_construction_stages(ReadOnlyArray::from_array(stages), error => {
      paged_construction_failed(resources, error)
    }) {
    Some(failure) => failure
    None => {
      guard prepared_steps.val is Some(_) &&
        ordered_dispatch.val is Some(dispatch) else {
        return paged_construction_failed(
          resources,
          InvalidExecutor(ResourceState),
        )
      }
      PagedReady({
        resources,
        completion,
        fp8_frame_envelope: None,
        fp8_frame_admission: None,
        fp8_scale_evidence: None,
        graph_telemetry: new_paged_graph_telemetry(
          open_paged_ordered_executor(resources, dispatch.telemetry_owner_index).execution_mode(),
          None,
        ),
        graph_bucket_table,
        ordered_dispatch: dispatch,
        fused_residual_spans: ReadOnlyArray::from_array([]),
        fused_dispatch_canary: None,
        lifecycle: GraphReady,
      })
    }
  }
}

///|
priv struct I8OwnedPreparationEvidence {
  blueprint : I8PagedExecutionBlueprint
  weight : @device.Allocation
  completion : PagedCompletionScratch
  functions : ReadOnlyArray[PagedFunctionBlueprint]
  function_indices : FixedArray[Int]
}

///|
fn admit_i8_owned_preparation_evidence(
  context : @device.Context,
  model_plan : @plan.ModelPlan,
  device_numeric : @device.DeviceNumericCapability,
  loaded : I8LoadedExecutionBlueprint,
  has_accepted_predecessor~ : Bool,
  accepted_predecessor_value~ : UInt64,
) -> I8OwnedPreparationEvidence raise DeviceStepError {
  let blueprint = loaded.blueprint()
  validate_predecessor_seed(
    has_accepted_predecessor, accepted_predecessor_value,
  )
  let weight = validate_i8_executor_evidence(
    context, model_plan, device_numeric, loaded,
  )
  let completion = prepare_paged_completion_scratch(
    model_plan,
    blueprint.memory_plan,
    blueprint.limits(),
  )
  let (functions, function_indices) = collect_i8_paged_functions(blueprint)
  { blueprint, weight, completion, functions, function_indices }
}
