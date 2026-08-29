///|
fn allocate_fp8_ephemeral(
  resources : Fp8DeviceResources,
  specs : ReadOnlyArray[Fp8EphemeralSpec],
) -> Unit raise Fp8DeviceExecutorError {
  for spec in specs {
    let allocation = resources.context.allocate(spec.byte_count) catch {
      error => raise Fp8DeviceFailure(Ephemeral, error)
    }
    resources.ephemeral.push({
      key: Some(spec.key),
      resource: allocation,
      byte_count: spec.byte_count,
      open: true,
    })
    resources.context.validate_allocation_region(
      allocation,
      offset=0L,
      byte_count=spec.byte_count,
      alignment=spec.alignment,
    ) catch {
      error => raise Fp8DeviceFailure(Ephemeral, error)
    }
  }
}

///|
fn load_fp8_modules(
  resources : Fp8DeviceResources,
  compiled : ArrayView[@fp8_release_authority.Fp8ReleaseOperationAuthorityV2],
) -> Unit raise Fp8DeviceExecutorError {
  for authority in compiled {
    if fp8_module_index(resources.modules, authority.module_digest()) is None {
      let module_ = resources.context.load_module(authority.module_bytes()) catch {
        error => raise Fp8DeviceFailure(Module, error)
      }
      resources.modules.push({
        digest: authority.module_digest(),
        resource: module_,
        open: true,
      })
    }
  }
}

///|
fn load_fp8_functions(
  resources : Fp8DeviceResources,
  compiled : ArrayView[@fp8_release_authority.Fp8ReleaseOperationAuthorityV2],
) -> Unit raise Fp8DeviceExecutorError {
  for authority in compiled {
    guard fp8_module_index(resources.modules, authority.module_digest())
      is Some(index) &&
      resources.modules[index].open else {
      raise InvalidFp8DeviceExecutor(ResourceState)
    }
    let function = resources.modules[index].resource.load_function(
      @utf8.encode(authority.function_symbol()),
    ) catch {
      error => raise Fp8DeviceFailure(Function, error)
    }
    resources.functions.push({
      module_index: index,
      resource: function,
      open: true,
    })
  }
}

///|
fn fp8_argument_for_operand(
  resources : Fp8DeviceResources,
  recipe : @fp8_runtime_recipe.Fp8RuntimeRecipeV2,
  launch : @fp8_launch_abi.Fp8StagedDynamicLaunchAbi,
  operand : @launch_contract.AotOperand,
) -> @device.KernelArgument raise Fp8DeviceExecutorError {
  let (allocation, offset) = match operand.role() {
    WeightInput(reference) => {
      guard recipe.weight_inspection().layout().region(reference)
        is Some(region) else {
        raise operation_error(launch.operation(), RawOperands)
      }
      (fp8_weight_allocation(resources), region.offset_bytes())
    }
    WeightScaleInput(_, scale) => {
      guard recipe.weight_inspection().layout().region(scale.tensor())
        is Some(region) else {
        raise operation_error(launch.operation(), RawOperands)
      }
      (fp8_weight_allocation(resources), region.offset_bytes())
    }
    Workspace =>
      (fp8_workspace_allocation(resources), launch.workspace().offset())
    RuntimeInput(input) =>
      (fp8_ephemeral_allocation(resources, Runtime(input)).resource, 0L)
    ExternalInput(input) =>
      (fp8_ephemeral_allocation(resources, External(input)).resource, 0L)
    ActivationInput(reference) | ActivationOutput(reference) =>
      (fp8_ephemeral_allocation(resources, Activation(reference)).resource, 0L)
    PersistentKvComponent(_, _) =>
      raise operation_error(launch.operation(), RawOperands)
  }
  @device.KernelArgument::device_region(
    allocation,
    offset~,
    byte_count=operand.byte_count(),
    alignment=operand.alignment(),
  ) catch {
    error => raise Fp8DeviceFailure(Arguments, error)
  }
}

///|
fn prepare_fp8_steps(
  resources : Fp8DeviceResources,
  recipe : @fp8_runtime_recipe.Fp8RuntimeRecipeV2,
) -> ReadOnlyArray[Fp8PreparedOperation] raise Fp8DeviceExecutorError {
  let steps : Array[Fp8PreparedOperation] = []
  for index, launch in recipe.launches() {
    let arguments : Array[@device.KernelArgument] = []
    let outputs : Array[@plan.ActivationRef] = []
    for operand in launch.raw_operands() {
      arguments.push(
        fp8_argument_for_operand(resources, recipe, launch, operand),
      )
      if operand.role() is ActivationOutput(reference) {
        outputs.push(reference)
      }
    }
    let source = launch.launch_dimensions()
    let dimensions = @device.KernelLaunchDimensions::new(
      grid_x=source.grid_x(),
      grid_y=source.grid_y(),
      grid_z=source.grid_z(),
      block_x=source.block_x(),
      block_y=source.block_y(),
      block_z=source.block_z(),
      shared_memory_bytes=source.shared_memory_bytes(),
    ) catch {
      _ => raise operation_error(launch.operation(), LaunchDimensions)
    }
    steps.push({
      operation: launch.operation(),
      function_index: index,
      dimensions,
      arguments: @device.KernelArguments::new(arguments) catch {
        error => raise Fp8DeviceFailure(Arguments, error)
      },
      workspace_offset: launch.workspace().offset(),
      workspace_bytes: launch.workspace().byte_length(),
      outputs: ReadOnlyArray::from_array(outputs),
    })
  }
  ReadOnlyArray::from_array(steps)
}

///|
fn run_fp8_construction_stages(
  stages : ReadOnlyArray[() -> Unit raise Fp8DeviceExecutorError],
) -> Fp8DeviceExecutorError? {
  for stage in stages {
    try stage() catch {
      error => return Some(error)
    } noraise {
      _ => ()
    }
  }
  None
}

///|
pub fn prepare_v2(
  context : @device.Context,
  release : @fp8_release_authority.Fp8ReleaseAuthorityV2,
  weights : @numeric_weight_file.AuthenticatedNumericWeightAuthority,
  limits : Fp8DeviceExecutorLimits,
) -> Fp8DeviceExecutorPreparation raise Fp8DeviceExecutorError {
  let recipe = release.recipe()
  let compiled = release.operations()
  let resources = empty_fp8_resources(context, weights)
  let weight_allocation : Ref[@device.Allocation?] = Ref(None)
  let specs : Ref[ReadOnlyArray[Fp8EphemeralSpec]?] = Ref(None)
  let steps : Ref[ReadOnlyArray[Fp8PreparedOperation]?] = Ref(None)
  let stages : Array[() -> Unit raise Fp8DeviceExecutorError] = [
    () => {
      validate_fp8_target(context, recipe)
      if release.recipe_digest() != recipe.digest() ||
        release.target() !=
        recipe.startup().kernel_artifacts().manifest().target() {
        raise InvalidFp8DeviceExecutor(ReleaseAuthority)
      }
      validate_fp8_release_authority(recipe, compiled, limits)
      weight_allocation.val = Some(validate_fp8_weights(recipe, weights))
      specs.val = Some(collect_ephemeral_specs(recipe, limits))
    },
    () => {
      guard weight_allocation.val is Some(allocation) else {
        raise InvalidFp8DeviceExecutor(ResourceState)
      }
      let lease = context.lease_allocation(allocation) catch {
        error => raise Fp8DeviceFailure(WeightLease, error)
      }
      resources.weight_lease = Some({ resource: lease, open: true })
    },
    () => {
      let stream = context.create_stream() catch {
        error => raise Fp8DeviceFailure(Stream, error)
      }
      resources.stream = Some({ resource: stream, open: true })
    },
    () => {
      let allocation = context.allocate(recipe.workspace_capacity_bytes()) catch {
        error => raise Fp8DeviceFailure(Workspace, error)
      }
      resources.workspace = Some({
        key: None,
        resource: allocation,
        byte_count: recipe.workspace_capacity_bytes(),
        open: true,
      })
      context.validate_allocation_region(
        allocation,
        offset=0L,
        byte_count=recipe.workspace_capacity_bytes(),
        alignment=4L,
      ) catch {
        error => raise Fp8DeviceFailure(Workspace, error)
      }
    },
    () => {
      guard specs.val is Some(value) else {
        raise InvalidFp8DeviceExecutor(ResourceState)
      }
      allocate_fp8_ephemeral(resources, value)
    },
    () => load_fp8_modules(resources, compiled),
    () => load_fp8_functions(resources, compiled),
    () => steps.val = Some(prepare_fp8_steps(resources, recipe)),
  ]
  match run_fp8_construction_stages(ReadOnlyArray::from_array(stages)) {
    Some(cause) => fp8_construction_failed(resources, cause)
    None =>
      match steps.val {
        Some(prepared) =>
          Fp8ExecutorReady({
            resources,
            recipe_digest: recipe.digest(),
            steps: prepared,
            scale_scratch: FixedArray::make(8, b'\x00'),
            epoch: 0UL,
            lifecycle: ExecutorReady,
          })
        None =>
          fp8_construction_failed(
            resources,
            InvalidFp8DeviceExecutor(ResourceState),
          )
      }
  }
}
