///|
fn validate_fp8_executor_weights_v3(
  context : @device.Context,
  blueprint : Fp8PagedExecutionBlueprintV3,
  weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority,
) -> @device.Allocation raise DeviceStepError {
  let authority = blueprint.authority()
  let recipe = authority.recipe()
  let plan = recipe.startup().model_plan()
  let observed = @device.admit_finite_fp8_e4m3_w8a8(context.capability()) catch {
    _ => raise InvalidExecutor(ContextTarget)
  }
  if !authority.target().matches_capability(context.capability()) ||
    observed.digest() != recipe.startup().device_numeric_capability().digest() ||
    weights.identity() != plan.identity() ||
    weights.numeric_schema_digest() != plan.numeric_binding().digest() ||
    weights.layout() != blueprint.weight_layout() ||
    weights.artifact_digest() != recipe.weight_inspection().artifact_digest() {
    raise InvalidExecutor(WeightIdentity)
  }
  let allocation = weights.allocation() catch { _ => raise NumericWeightClose }
  if allocation.byte_count() != blueprint.weight_layout().arena_bytes() {
    raise InvalidExecutor(WeightLayout)
  }
  for step in blueprint.steps() {
    for operand in step.operands() {
      if operand.space() == WeightArena {
        let reference = match operand.role() {
          WeightInput(reference) => reference
          WeightScaleInput(_, scale) => scale.tensor()
          _ => raise executor_operation_error(step.operation_id(), WeightRegion)
        }
        guard blueprint.weight_layout().region(reference) is Some(region) &&
          region.offset_bytes() == operand.offset_bytes() &&
          region.byte_length() == operand.byte_count() else {
          raise executor_operation_error(step.operation_id(), WeightRegion)
        }
      }
    }
  }
  allocation
}

///|
fn collect_fp8_paged_functions_v3(
  blueprint : Fp8PagedExecutionBlueprintV3,
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
fn validate_fp8_physical_regions_v3(
  context : @device.Context,
  blueprint : Fp8PagedExecutionBlueprintV3,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
) -> Unit raise DeviceStepError {
  for step in blueprint.steps() {
    for binding in step.operands() {
      context.validate_allocation_region(
        allocation_for_paged_operand(
          binding, descriptor, weight, activation, kv,
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
fn prepare_fp8_paged_steps_v3(
  blueprint : Fp8PagedExecutionBlueprintV3,
  function_indices : FixedArray[Int],
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
) -> ReadOnlyArray[PreparedPagedGraphStep] raise DeviceStepError {
  let result : Array[PreparedPagedGraphStep] = []
  for index, step in blueprint.steps() {
    let arguments : Array[@device.KernelArgument] = []
    for binding in step.operands() {
      arguments.push(
        @device.KernelArgument::device_region(
          allocation_for_paged_operand(
            binding, descriptor, weight, activation, kv,
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
    result.push({
      operation_id: step.operation_id(),
      function_index: function_indices[index],
      dimensions,
      arguments: @device.KernelArguments::new(arguments) catch {
        error => raise ExecutorDevice(KernelArguments, error)
      },
    })
  }
  ReadOnlyArray::from_array(result)
}

///|
/// Constructs the complete mixed paged graph under one descriptor, activation
/// arena, KV arena, stream, event, and ordered executor. Ownership of
/// `weights` transfers at entry and is retained on every cleanup failure.
pub fn prepare_fp8_paged_graph_executor_v3(
  context : @device.Context,
  model_generation : @worker_protocol.ModelPlanGeneration,
  blueprint : Fp8PagedExecutionBlueprintV3,
  weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority,
  has_accepted_predecessor~ : Bool,
  accepted_predecessor_value~ : UInt64,
  graph_policy? : PagedGraphExecutionPolicy = PagedOrderedEagerOnly,
  diagnostic_policy? : PagedExecutionDiagnosticPolicy = ProductionFastPath,
) -> PagedGraphPreparation raise DeviceStepError {
  let resources = empty_paged_resources()
  resources.numeric_weights = Some(Fp8NumericWeights(weights))
  let authority = blueprint.authority()
  let plan = authority.recipe().startup().model_plan()
  let weight = validate_fp8_executor_weights_v3(context, blueprint, weights) catch {
    error => return paged_construction_failed(resources, error)
  }
  validate_predecessor_seed(
    has_accepted_predecessor, accepted_predecessor_value,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let completion = prepare_paged_completion_scratch(
    plan,
    blueprint.memory_plan(),
    blueprint.limits(),
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let frame_envelope = make_fp8_frame_envelope_v3(
    authority,
    model_generation,
    blueprint.limits(),
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let scale_bytes = blueprint.fp8_workspace_bytes()
  if scale_bytes <= 0L || scale_bytes > 2147483647L {
    return paged_construction_failed(
      resources,
      InvalidExecutor(Fp8ScaleEvidence),
    )
  }
  let scale_evidence = prepare_fp8_scale_diagnostics(
    diagnostic_policy,
    blueprint.fp8_workspace_base(),
    scale_bytes,
  )
  let (functions, function_indices) = collect_fp8_paged_functions_v3(blueprint)
  let descriptor_outcome = prepare(
    context,
    plan,
    model_generation,
    blueprint.layout(),
    blueprint.limits(),
    has_accepted_predecessor~,
    accepted_predecessor_value~,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let descriptor = match descriptor_outcome {
    Ready(owner) => owner
    CleanupRequired(failed) => {
      resources.descriptor = Some(Failed(failed))
      return paged_construction_failed(resources, failed.cause())
    }
  }
  resources.descriptor = Some(Complete(descriptor))
  let lease = context.lease_allocation(weight) catch {
    error =>
      return paged_construction_failed(
        resources,
        ExecutorDevice(WeightLease, error),
      )
  }
  resources.weight_lease = Some({ resource: lease, open: true })
  let stream = context.create_stream() catch {
    error =>
      return paged_construction_failed(resources, ExecutorDevice(Stream, error))
  }
  resources.stream = Some({ resource: stream, open: true })
  let activation = context.allocate(
    blueprint.activation_workspace_arena_bytes(),
  ) catch {
    error =>
      return paged_construction_failed(
        resources,
        ExecutorDevice(ActivationWorkspaceArena, error),
      )
  }
  resources.activation_workspace = Some({ resource: activation, open: true })
  let kv = context.allocate(blueprint.layout().arena_bytes()) catch {
    error =>
      return paged_construction_failed(
        resources,
        ExecutorDevice(PersistentKvArena, error),
      )
  }
  resources.persistent_kv = Some({ resource: kv, open: true })
  if activation.byte_count() != blueprint.activation_workspace_arena_bytes() ||
    kv.byte_count() != blueprint.layout().arena_bytes() {
    return paged_construction_failed(resources, InvalidExecutor(AllocationSize))
  }
  validate_fp8_physical_regions_v3(
    context,
    blueprint,
    descriptor.resources,
    weight,
    activation,
    kv,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  load_paged_modules(context, blueprint.artifacts(), resources) catch {
    error => return paged_construction_failed(resources, error)
  }
  load_paged_functions(functions, resources) catch {
    error => return paged_construction_failed(resources, error)
  }
  let steps = prepare_fp8_paged_steps_v3(
    blueprint,
    function_indices,
    descriptor.resources,
    weight,
    activation,
    kv,
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let graph_bucket_table = new_paged_graph_bucket_table(blueprint.profile())
  let ordered_dispatch = prepare_paged_ordered_executors(
    context,
    steps,
    resources,
    graph_policy,
    graph_bucket_table,
    blueprint.profile(),
  ) catch {
    error => return paged_construction_failed(resources, error)
  }
  let graph_telemetry = new_paged_graph_telemetry(
    open_paged_ordered_executor(
      resources,
      ordered_dispatch.telemetry_owner_index,
    ).execution_mode(),
    None,
  )
  PagedReady({
    resources,
    completion,
    fp8_frame_envelope: Some(frame_envelope),
    fp8_frame_admission: Some({
      armed: false,
      descriptor_epoch: 0UL,
      wire_owner: None,
      wire_epoch: 0UL,
      sequence_value: 0UL,
      row_count: 0,
      token_count: 0,
      page_count: 0,
    }),
    fp8_scale_evidence: scale_evidence,
    graph_telemetry,
    graph_bucket_table,
    ordered_dispatch,
    fused_residual_spans: ReadOnlyArray::from_array([]),
    fused_dispatch_canary: None,
    lifecycle: GraphReady,
  })
}
