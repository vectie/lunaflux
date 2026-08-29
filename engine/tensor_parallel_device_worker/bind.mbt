///|
fn descriptor_stage(
  role : @tensor_parallel_execution_plan.TensorParallelDescriptorRole,
) -> @device_step.DeviceStepStage {
  match role {
    RuntimeDescriptor(StepCounts) => Counts
    RuntimeDescriptor(QueryPositions) => TokenPositions
    RuntimeDescriptor(QueryRowOffsets) => QueryRowOffsets
    RuntimeDescriptor(SequenceLengths) => SequenceLengths
    RuntimeDescriptor(PageTableOffsets) => PageOffsets
    RuntimeDescriptor(PageTablePageIndices) => PhysicalPageIndices
    ExternalDescriptor(TokenIds) => TokenIds
  }
}

///|
fn module_index_for_digest(
  artifacts : @artifact.KernelArtifactBundle,
  digest : @catalog.AotArtifactDigest,
) -> Int {
  for index, artifact_module in artifacts.modules() {
    if artifact_module.digest() == digest {
      return index
    }
  }
  -1
}

///|
fn load_modules(
  plan : TensorParallelDeviceWorkerPlan,
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  let context = require_context(resources)
  for artifact_module in plan.artifacts.modules() {
    let resource = context.load_module(artifact_module.module_bytes()) catch {
      error => raise Device(Module, error)
    }
    resources.modules.push({ resource, open: true })
  }
  if resources.modules.length() != plan.artifacts.module_count() {
    raise Invalid(ArtifactBinding)
  }
}

///|
fn find_function_blueprint(
  blueprints : ArrayView[TensorParallelFunctionBlueprint],
  module_index : Int,
  symbol : String,
) -> Int {
  for index, blueprint in blueprints {
    if blueprint.module_index == module_index && blueprint.symbol == symbol {
      return index
    }
  }
  -1
}

///|
fn collect_function_blueprints(
  plan : TensorParallelDeviceWorkerPlan,
) -> (ReadOnlyArray[TensorParallelFunctionBlueprint], FixedArray[Int]) raise TensorParallelDeviceWorkerError {
  let blueprints : Array[TensorParallelFunctionBlueprint] = []
  let indices = FixedArray::make(plan.execution.operation_count(), 0)
  for operation_index in 0..<plan.execution.operation_count() {
    let operation_id = @plan.OperationId::new(operation_index) catch {
      _ => raise Invalid(ExecutionPlan)
    }
    let operation = plan.execution.operation_exact(operation_id) catch {
      _ => raise Invalid(ExecutionPlan)
    }
    let module_index = module_index_for_digest(
      plan.artifacts,
      operation.module_digest(),
    )
    if module_index < 0 {
      raise Invalid(ArtifactBinding)
    }
    let existing = find_function_blueprint(
      blueprints,
      module_index,
      operation.function_symbol(),
    )
    if existing >= 0 {
      indices[operation_index] = existing
    } else {
      blueprints.push({ module_index, symbol: operation.function_symbol() })
      indices[operation_index] = blueprints.length() - 1
    }
  }
  (ReadOnlyArray::from_array(blueprints), indices)
}

///|
fn load_functions(
  blueprints : ReadOnlyArray[TensorParallelFunctionBlueprint],
  resources : TensorParallelPartialResources,
) -> Unit raise TensorParallelDeviceWorkerError {
  for blueprint in blueprints {
    if blueprint.module_index < 0 ||
      blueprint.module_index >= resources.modules.length() {
      raise Invalid(ArtifactBinding)
    }
    let loaded_module = resources.modules[blueprint.module_index]
    if !loaded_module.open {
      raise Invalid(ArtifactBinding)
    }
    let resource = loaded_module.resource.load_function(
      @utf8.encode(blueprint.symbol),
    ) catch {
      error => raise Device(Function, error)
    }
    resources.functions.push({
      resource,
      module_index: blueprint.module_index,
      open: true,
    })
  }
}

///|
fn allocation_argument(
  context : @device.Context,
  allocation : @device.Allocation,
  region : @tensor_parallel_execution_plan.TensorParallelScalarRegion,
  expected_space : @tensor_parallel_execution_plan.TensorParallelMemorySpace,
) -> @device.KernelArgument raise TensorParallelDeviceWorkerError {
  if region.space() != expected_space {
    raise Invalid(ExecutionPlan)
  }
  context.validate_allocation_region(
    allocation,
    offset=region.offset_bytes(),
    byte_count=region.byte_count(),
    alignment=region.alignment(),
  ) catch {
    error => raise Device(ActivationWorkspace, error)
  }
  @device.KernelArgument::device_region(
    allocation,
    offset=region.offset_bytes(),
    byte_count=region.byte_count(),
    alignment=region.alignment(),
  ) catch {
    error => raise Device(ActivationWorkspace, error)
  }
}

///|
fn physical_operand_argument(
  operand : @tensor_parallel_execution_plan.TensorParallelPhysicalOperand,
  resources : TensorParallelPartialResources,
) -> @device.KernelArgument raise TensorParallelDeviceWorkerError {
  let context = require_context(resources)
  let region = operand.region()
  match operand.role() {
    RuntimeInput(input) => {
      if region.space() != DescriptorSpace {
        raise Invalid(DescriptorBinding)
      }
      let descriptor = require_descriptor(resources)
      descriptor.descriptor_argument(
        descriptor_stage(RuntimeDescriptor(input)),
        byte_count=region.byte_count(),
        alignment=region.alignment(),
      ) catch {
        error => raise Descriptor(error)
      }
    }
    ExternalInput(input) => {
      if region.space() != DescriptorSpace {
        raise Invalid(DescriptorBinding)
      }
      let descriptor = require_descriptor(resources)
      descriptor.descriptor_argument(
        descriptor_stage(ExternalDescriptor(input)),
        byte_count=region.byte_count(),
        alignment=region.alignment(),
      ) catch {
        error => raise Descriptor(error)
      }
    }
    RankConstants(..) =>
      allocation_argument(
        context,
        require_allocation(resources.rank_constants),
        region,
        RankConstantsSpace,
      )
    WeightInput(_) =>
      allocation_argument(
        context,
        require_weights(resources).allocation(),
        region,
        WeightSpace,
      )
    ActivationInput(_) | ActivationOutput(_) | Workspace =>
      allocation_argument(
        context,
        require_allocation(resources.activation_workspace),
        region,
        ActivationWorkspaceSpace,
      )
    PersistentKvComponent(layer, component) => {
      if region.space() != PersistentKvSpace {
        raise Invalid(KvBinding)
      }
      let kv = require_kv(resources)
      let argument = kv.component_argument(context, layer, component) catch {
        error => raise Kv(error)
      }
      let layout = kv.layout()
      let component_offset = match component {
        Key => 0L
        Value => layout.component_stride_bytes()
      }
      let expected_offset = layer.as_int().to_int64() *
        layout.layer_stride_bytes() +
        component_offset
      if region.offset_bytes() != expected_offset ||
        region.byte_count() != layout.component_stride_bytes() ||
        region.alignment() != layout.segment_alignment_bytes() {
        raise Invalid(KvBinding)
      }
      argument
    }
  }
}

///|
fn physical_dimensions(
  source : @launch_contract.AotLaunchDimensions,
) -> @device.KernelLaunchDimensions raise TensorParallelDeviceWorkerError {
  @device.KernelLaunchDimensions::new(
    grid_x=source.grid_x(),
    grid_y=source.grid_y(),
    grid_z=source.grid_z(),
    block_x=source.block_x(),
    block_y=source.block_y(),
    block_z=source.block_z(),
    shared_memory_bytes=source.shared_memory_bytes(),
  ) catch {
    _ => raise Invalid(ExecutionPlan)
  }
}

///|
fn prepare_physical_steps(
  plan : TensorParallelDeviceWorkerPlan,
  resources : TensorParallelPartialResources,
) -> TensorParallelPreparedLaunches raise TensorParallelDeviceWorkerError {
  load_modules(plan, resources)
  let (blueprints, function_indices) = collect_function_blueprints(plan)
  load_functions(blueprints, resources)
  let steps : Array[TensorParallelPreparedStep] = []
  let launches : Array[@device.OrderedKernelLaunch] = []
  for operation_index in 0..<plan.execution.operation_count() {
    let operation_id = @plan.OperationId::new(operation_index) catch {
      _ => raise Invalid(ExecutionPlan)
    }
    let operation = plan.execution.operation_exact(operation_id) catch {
      _ => raise Invalid(ExecutionPlan)
    }
    let arguments : Array[@device.KernelArgument] = []
    for operand in operation.operands() {
      arguments.push(physical_operand_argument(operand, resources))
    }
    let packed = @device.KernelArguments::new(arguments) catch {
      error => raise Device(Function, error)
    }
    let function_index = function_indices[operation_index]
    if function_index < 0 || function_index >= resources.functions.length() {
      raise Invalid(ArtifactBinding)
    }
    let function = resources.functions[function_index]
    if !function.open {
      raise Invalid(ResourceState)
    }
    steps.push({ operation_id, })
    launches.push(
      @device.OrderedKernelLaunch::new(
        function.resource,
        physical_dimensions(operation.dimensions()),
        packed,
      ),
    )
  }
  {
    steps: ReadOnlyArray::from_array(steps),
    launches: ReadOnlyArray::from_array(launches),
  }
}
