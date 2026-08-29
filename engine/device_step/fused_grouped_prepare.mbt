///|
fn prepare_fused_residual_resources(
  context : @device.Context,
  span : ApprovedFusedResidualRmsNormSpan,
  resources : PagedGraphResources,
) -> Int raise DeviceStepError {
  let module_index = resources.modules.length()
  let loaded = context.load_module(span.approved_module_bytes()) catch {
    error => raise ExecutorDevice(Module, error)
  }
  resources.modules.push({ resource: loaded, open: true })
  let function = loaded.load_function(
    @utf8.encode(span.approved_function_symbol()),
  ) catch {
    error => raise ExecutorDevice(Function, error)
  }
  let function_index = resources.functions.length()
  resources.functions.push({ resource: function, module_index, open: true })
  if span.uses_qualification_canary() {
    let canary = context.allocate(4L) catch {
      error => raise ExecutorDevice(FusedDispatchCanary, error)
    }
    resources.fused_dispatch_canary = Some({ resource: canary, open: true })
  }
  function_index
}

///|
fn fused_residual_argument(
  binding : PagedKvOperandBinding,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation_workspace : @device.Allocation,
  persistent_kv : @device.Allocation,
) -> @device.KernelArgument raise DeviceStepError {
  @device.KernelArgument::device_region(
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

///|
fn fused_residual_argument_count_for_abi(
  abi : @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi,
) -> Int raise DeviceStepError {
  match abi {
    ResidualRmsNormAbiV1 => 7
    ResidualRmsNormProductionAbiV2 => 6
    _ => raise InvalidExecutor(ResourceState)
  }
}

///|
fn prepare_fused_residual_step(
  blueprint : PagedKvExecutionBlueprint,
  span : ApprovedFusedResidualRmsNormSpan,
  function_index : Int,
  resources : PagedGraphResources,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation_workspace : @device.Allocation,
  persistent_kv : @device.Allocation,
) -> PreparedPagedGraphStep raise DeviceStepError {
  let (residual_index, norm_index) = span.operation_indices()
  guard blueprint.steps().get(residual_index) is Some(residual) &&
    blueprint.steps().get(norm_index) is Some(norm) &&
    residual.operands().length() == 4 &&
    norm.operands().length() == 4 &&
    residual.operation_kind() == ResidualAdd &&
    norm.operation_kind() == RmsNorm else {
    raise InvalidExecutor(ResourceState)
  }
  let r = residual.operands()
  let n = norm.operands()
  guard r[0].role() == RuntimeInput(StepCounts) &&
    n[0].role() == RuntimeInput(StepCounts) &&
    r[1].role() is ActivationInput(left) &&
    r[2].role() is ActivationInput(right) &&
    r[3].role() is ActivationOutput(residual_output) &&
    n[1].role() == ActivationInput(residual_output) &&
    n[2].role() is WeightInput(norm_weight) &&
    n[3].role() is ActivationOutput(norm_output) else {
    raise executor_operation_error(residual.operation_id(), ResourceState)
  }
  ignore(left)
  ignore(right)
  ignore(norm_weight)
  ignore(norm_output)
  if r[0].space() != n[0].space() || r[0].offset_bytes() != n[0].offset_bytes() {
    raise executor_operation_error(residual.operation_id(), ResourceState)
  }
  let arguments : Array[@device.KernelArgument] = []
  for binding in [r[0], r[1], r[2], n[2], r[3], n[3]] {
    arguments.push(
      fused_residual_argument(
        binding, descriptor, weight, activation_workspace, persistent_kv,
      ),
    )
  }
  if span.uses_qualification_canary() {
    guard resources.fused_dispatch_canary is Some(canary) && canary.open else {
      raise InvalidExecutor(ResourceState)
    }
    arguments.push(
      @device.KernelArgument::device_region(
        canary.resource,
        offset=0L,
        byte_count=4L,
        alignment=4L,
      ) catch {
        error => raise ExecutorDevice(KernelArguments, error)
      },
    )
  } else if resources.fused_dispatch_canary is Some(_) {
    raise InvalidExecutor(ResourceState)
  }
  if arguments.length() !=
    fused_residual_argument_count_for_abi(span.kernel_abi()) {
    raise InvalidExecutor(ResourceState)
  }
  let launch = span.approved_launch()
  let dimensions = @device.KernelLaunchDimensions::new(
    grid_x=launch.grid_x(),
    grid_y=launch.grid_y(),
    grid_z=launch.grid_z(),
    block_x=launch.block_x(),
    block_y=launch.block_y(),
    block_z=launch.block_z(),
    shared_memory_bytes=launch.shared_memory_bytes(),
  ) catch {
    _ =>
      raise executor_operation_error(residual.operation_id(), LaunchDimensions)
  }
  let packed = @device.KernelArguments::new(arguments) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
  {
    operation_id: residual.operation_id(),
    function_index,
    dimensions,
    arguments: packed,
  }
}

///|
fn open_fused_dispatch_canary(
  resources : PagedGraphResources,
) -> @device.Allocation raise DeviceStepError {
  guard resources.fused_dispatch_canary is Some(owner) && owner.open else {
    raise InvalidExecutor(ResourceState)
  }
  owner.resource
}

///|
fn arm_fused_dispatch_canary(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
) -> Unit raise DeviceStepError {
  guard executor.fused_dispatch_canary is Some(state) else { return }
  for index in 0..<4 {
    state.host[index] = b'\x00'
  }
  let allocation = open_fused_dispatch_canary(executor.resources)
  descriptor.context.copy_from_fixed_host(allocation, state.host, 0L, 0L, 4L) catch {
    error => raise ExecutorDevice(FusedDispatchCanary, error)
  }
}

///|
fn validate_fused_dispatch_canary(
  executor : PagedGraphExecutor,
  descriptor : DeviceStepOwner,
  staged_epoch : UInt64,
) -> Unit raise DeviceStepError {
  guard executor.fused_dispatch_canary is Some(state) else { return }
  let token_count = StagedDeviceStep(descriptor, staged_epoch).token_count()
  ignore(fused_expected_dispatch_canary(state.per_token, token_count))
  let allocation = open_fused_dispatch_canary(executor.resources)
  descriptor.context.copy_to_fixed_host(allocation, state.host, 0L, 0L, 4L) catch {
    error => raise ExecutorDevice(FusedDispatchCanary, error)
  }
  let observed = fused_decode_dispatch_canary(state.host)
  require_fused_dispatch_canary_observation(
    state.per_token,
    token_count,
    observed,
  )
}

///|
fn fused_expected_dispatch_canary(
  per_token : UInt,
  token_count : Int,
) -> UInt raise DeviceStepError {
  if per_token == 0U ||
    token_count <= 0 ||
    token_count.reinterpret_as_uint() > 0xffffffffU / per_token {
    raise InvalidExecutor(ResourceState)
  }
  per_token * token_count.reinterpret_as_uint()
}

///|
fn fused_decode_dispatch_canary(host : FixedArray[Byte]) -> UInt {
  host[0].to_uint() |
  (host[1].to_uint() << 8) |
  (host[2].to_uint() << 16) |
  (host[3].to_uint() << 24)
}

///|
fn require_fused_dispatch_canary_observation(
  per_token : UInt,
  token_count : Int,
  observed : UInt,
) -> Unit raise DeviceStepError {
  if observed != fused_expected_dispatch_canary(per_token, token_count) {
    raise InvalidExecutor(ResourceState)
  }
}
