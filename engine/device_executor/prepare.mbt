///|
fn empty_resources() -> ExecutorResources {
  {
    stream: None,
    token_staging: None,
    arena: None,
    modules: [],
    functions: [],
    vendor_gemm: None,
    vendor_plans: [],
  }
}

///|
fn construction_failed(
  resources : ExecutorResources,
  cause : DeviceExecutorError,
) -> PreparationOutcome raise DeviceExecutorError {
  retain_on_cleanup_failure(resources, cause, close_all_retry_safe, (
    retained,
    original,
    cleanup_failure,
  ) => {
    CleanupRequired({
      resources: retained,
      cause: original,
      cleanup_failure,
      lifecycle: ClosingFailed,
    })
  })
}

///|
fn[Resource, Outcome] retain_on_cleanup_failure(
  resource : Resource,
  cause : DeviceExecutorError,
  cleanup : (Resource) -> Unit raise DeviceExecutorError,
  retain : (Resource, DeviceExecutorError, DeviceExecutorError) -> Outcome,
) -> Outcome raise DeviceExecutorError {
  try cleanup(resource) catch {
    cleanup_failure => retain(resource, cause, cleanup_failure)
  } noraise {
    _ => raise cause
  }
}

///|
fn allocation_for_space(
  space : @device_profile.ExecutionBufferSpace,
  token_staging : @device.Allocation,
  weight_allocation : @device.Allocation,
  arena : @device.Allocation,
) -> @device.Allocation {
  match space {
    TokenInput => token_staging
    WeightArena => weight_allocation
    ActivationWorkspaceArena => arena
  }
}

///|
fn validate_physical_region(
  context : @device.Context,
  binding : RegionBinding,
  token_staging : @device.Allocation,
  weight_allocation : @device.Allocation,
  arena : @device.Allocation,
) -> Unit raise DeviceExecutorError {
  context.validate_allocation_region(
    allocation_for_space(
      binding.allocation_space,
      token_staging,
      weight_allocation,
      arena,
    ),
    offset=binding.offset_bytes,
    byte_count=binding.byte_count,
    alignment=binding.alignment,
  ) catch {
    error => raise DeviceFailed(KernelArguments, error)
  }
}

///|
fn validate_blueprint_regions(
  context : @device.Context,
  blueprint : PreparationBlueprint,
  token_staging : @device.Allocation,
  weight_allocation : @device.Allocation,
  arena : @device.Allocation,
) -> Unit raise DeviceExecutorError {
  for step in blueprint.steps {
    match step {
      Aot(aot) =>
        for operand in aot.operands {
          validate_physical_region(
            context, operand, token_staging, weight_allocation, arena,
          )
        }
      Vendor(vendor) => {
        let left_bytes = checked_bf16_bytes(
          vendor.rows,
          vendor.inner,
          vendor.operation_id,
        )
        context.validate_allocation_region(
          arena,
          offset=vendor.left_offset,
          byte_count=left_bytes,
          alignment=16L,
        ) catch {
          error => raise DeviceFailed(KernelArguments, error)
        }
        let weight_bytes = checked_bf16_bytes(
          vendor.columns,
          vendor.inner,
          vendor.operation_id,
        )
        context.validate_allocation_region(
          weight_allocation,
          offset=vendor.weight_offset,
          byte_count=weight_bytes,
          alignment=16L,
        ) catch {
          error => raise DeviceFailed(KernelArguments, error)
        }
        let output_bytes = checked_bf16_bytes(
          vendor.rows,
          vendor.columns,
          vendor.operation_id,
        )
        context.validate_allocation_region(
          arena,
          offset=vendor.output_offset,
          byte_count=output_bytes,
          alignment=16L,
        ) catch {
          error => raise DeviceFailed(KernelArguments, error)
        }
        if vendor.workspace_bytes > 0L {
          context.validate_allocation_region(
            arena,
            offset=vendor.workspace_offset,
            byte_count=vendor.workspace_bytes,
            alignment=256L,
          ) catch {
            error => raise DeviceFailed(KernelArguments, error)
          }
        }
      }
    }
  }
}

///|
fn load_modules(
  context : @device.Context,
  artifacts : @artifact.KernelArtifactBundle,
  resources : ExecutorResources,
) -> Unit raise DeviceExecutorError {
  for image in artifacts.modules() {
    let resource = context.load_module(image.module_bytes()) catch {
      error => raise DeviceFailed(Module, error)
    }
    resources.modules.push({ resource, open: true })
  }
}

///|
fn load_functions(
  blueprint : PreparationBlueprint,
  resources : ExecutorResources,
) -> Unit raise DeviceExecutorError {
  for function in blueprint.functions {
    guard resources.modules.get(function.module_index) is Some(owner) else {
      raise InvalidPlan(ArtifactModule)
    }
    let resource = owner.resource.load_function(@utf8.encode(function.symbol)) catch {
      error => raise DeviceFailed(Function, error)
    }
    resources.functions.push({
      resource,
      module_index: function.module_index,
      open: true,
    })
  }
}

///|
fn prepare_steps(
  blueprint : PreparationBlueprint,
  token_staging : @device.Allocation,
  weight_allocation : @device.Allocation,
  arena : @device.Allocation,
  resources : ExecutorResources,
) -> ReadOnlyArray[PreparedStep] raise DeviceExecutorError {
  let steps : Array[PreparedStep] = []
  for step in blueprint.steps {
    match step {
      Aot(aot) => {
        let arguments : Array[@device.KernelArgument] = []
        for operand in aot.operands {
          let allocation = allocation_for_space(
            operand.allocation_space,
            token_staging,
            weight_allocation,
            arena,
          )
          let argument = @device.KernelArgument::device_region(
            allocation,
            offset=operand.offset_bytes,
            byte_count=operand.byte_count,
            alignment=operand.alignment,
          ) catch {
            error => raise DeviceFailed(KernelArguments, error)
          }
          arguments.push(argument)
        }
        let packed = @device.KernelArguments::new(arguments) catch {
          error => raise DeviceFailed(KernelArguments, error)
        }
        steps.push(
          Aot({
            function_index: aot.function_index,
            dimensions: aot.dimensions,
            arguments: packed,
          }),
        )
      }
      Vendor(vendor) => {
        guard resources.vendor_gemm is Some(handle) else {
          raise InvalidPlan(UnsupportedVendorOperation)
        }
        let resource = handle.resource.create_bf16_gemm_plan(
          vendor.rows,
          vendor.inner,
          vendor.columns,
          vendor.workspace_bytes,
        ) catch {
          error => raise DeviceFailed(VendorGemmPlan, error)
        }
        let plan_index = resources.vendor_plans.length()
        resources.vendor_plans.push({ resource, open: true })
        steps.push(
          Vendor({
            plan_index,
            left_offset: vendor.left_offset,
            weight_offset: vendor.weight_offset,
            output_offset: vendor.output_offset,
            workspace_offset: vendor.workspace_offset,
          }),
        )
      }
    }
  }
  ReadOnlyArray::from_array(steps)
}

///|
/// Validates every cross-package identity, exact kernel contract, and logical
/// region before opening the first native resource. After the reusable arenas
/// are allocated, their opaque physical addresses are checked before any
/// module, function, or vendor plan is opened. Construction is bounded and
/// cleaned in reverse order on failure.
pub fn prepare(
  context : @device.Context,
  static_plan : @device_plan.StaticDevicePlan,
  memory_plan : @device_memory.DeviceMemoryPlan,
  profile : @device_profile.DeviceExecutionProfile,
  profile_id : @launch_contract.KernelProfileId,
  weights : @device_materialize.DeviceWeights,
  contracts : @launch_contract.LaunchContractSet,
  artifacts : @artifact.KernelArtifactBundle,
) -> PreparationOutcome raise DeviceExecutorError {
  let blueprint = build_blueprint(
    context, static_plan, memory_plan, profile, profile_id, weights, contracts, artifacts,
  )
  let resources = empty_resources()
  let stream = context.create_stream() catch {
    error => raise DeviceFailed(Stream, error)
  }
  resources.stream = Some({ resource: stream, open: true })
  let token_staging = context.allocate(profile.token_staging_capacity_bytes()) catch {
    error =>
      return construction_failed(resources, DeviceFailed(TokenStaging, error))
  }
  resources.token_staging = Some({ resource: token_staging, open: true })
  let arena = @device_memory.allocate(context, memory_plan) catch {
    error =>
      return construction_failed(
        resources,
        MemoryFailed(ActivationWorkspaceArena, error),
      )
  }
  resources.arena = Some({ resource: arena, open: true })
  validate_blueprint_regions(
    context,
    blueprint,
    token_staging,
    weights.allocation(),
    arena.allocation(),
  ) catch {
    error => return construction_failed(resources, error)
  }
  load_modules(context, artifacts, resources) catch {
    error => return construction_failed(resources, error)
  }
  load_functions(blueprint, resources) catch {
    error => return construction_failed(resources, error)
  }
  if blueprint.has_vendor_steps {
    let resource = context.create_cublas_lt() catch {
      error =>
        return construction_failed(
          resources,
          DeviceFailed(VendorGemmHandle, error),
        )
    }
    resources.vendor_gemm = Some({ resource, open: true })
  }
  let prepared_steps = prepare_steps(
    blueprint,
    token_staging,
    weights.allocation(),
    arena.allocation(),
    resources,
  ) catch {
    error => return construction_failed(resources, error)
  }
  Ready({
    resources,
    weight_allocation: weights.allocation(),
    steps: prepared_steps,
    terminal_output: profile.final_output(),
    token_bytes: blueprint.token_staging_bytes,
    vocabulary_size: blueprint.vocabulary_size,
    output_valid: false,
    lifecycle: Ready,
  })
}
