///|
fn paged_decode_split_loaded_module(
  context : @device.Context,
  binding : @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
  loaded : Array[PagedDecodeSplitLoadedModule],
  resources : PagedGraphResources,
) -> Int32 raise DeviceStepError {
  let digest = binding.contract().launches()[0].module_digest()
  for item in loaded {
    if item.digest == digest {
      return item.module_index
    }
  }
  let module_index = resources.modules.length()
  let module_ = context.load_module(binding.module_bytes()) catch {
    error => raise ExecutorDevice(Module, error)
  }
  resources.modules.push({ resource: module_, open: true, })
  loaded.push({ digest, module_index, })
  module_index
}

///|
fn paged_decode_split_loaded_function(
  module_index : Int32,
  symbol : String,
  loaded : Array[PagedDecodeSplitLoadedFunction],
  resources : PagedGraphResources,
) -> Int32 raise DeviceStepError {
  for item in loaded {
    if item.module_index == module_index && item.symbol == symbol {
      return item.function_index
    }
  }
  guard resources.modules.get(module_index) is Some(module_) && module_.open else {
    raise InvalidExecutor(ResourceState)
  }
  let function = module_.resource.load_function(@utf8.encode(symbol)) catch {
    error => raise ExecutorDevice(Function, error)
  }
  let function_index = resources.functions.length()
  resources.functions.push({ resource: function, module_index, open: true, })
  loaded.push({ module_index, symbol, function_index, })
  function_index
}

///|
fn paged_decode_split_argument(
  operation : @plan.OperationId,
  step : PagedKvExecutionStep,
  operand : @launch_contract.AotOperand,
  workspace : @device.Allocation,
  qwen_sidecar : @device.Allocation?,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
) -> @device.KernelArgument raise DeviceStepError {
  if operand.role() == Workspace {
    return @device.KernelArgument::device_region(
      workspace,
      offset=0L,
      byte_count=operand.byte_count(),
      alignment=operand.alignment(),
    ) catch {
      error => raise ExecutorDevice(KernelArguments, error)
    }
  }
  if operand.role() is ActivationInput(_) {
    match qwen_sidecar {
      Some(sidecar) =>
        return @device.KernelArgument::device_region(
          sidecar,
          offset=0L,
          byte_count=operand.byte_count(),
          alignment=operand.alignment(),
        ) catch {
          error => raise ExecutorDevice(KernelArguments, error)
        }
      None => ()
    }
  }
  let binding = unique_composite_binding(step, role => role == operand.role())
  if binding.byte_count() != operand.byte_count() ||
    binding.alignment() < operand.alignment() ||
    binding.offset_bytes() % operand.alignment() != 0L {
    raise executor_operation_error(operation, ResourceState)
  }
  composite_binding_argument(binding, descriptor, weight, activation, kv)
}

///|
fn validate_paged_decode_split_operand(
  operation : @plan.OperationId,
  step : PagedKvExecutionStep,
  operand : @launch_contract.AotOperand,
  workspace_bytes : Int64,
  workspace_alignment : Int64,
  qwen_sidecar_shape : (Int64, Int64)?,
) -> Unit raise DeviceStepError {
  if operand.role() == Workspace {
    if operand.byte_count() > workspace_bytes ||
      operand.alignment() > workspace_alignment {
      raise executor_operation_error(operation, ResourceState)
    }
    return
  }
  if operand.role() is ActivationInput(_) {
    match qwen_sidecar_shape {
      Some(shape) => {
        if operand.byte_count() != shape.0 || operand.alignment() > shape.1 {
          raise executor_operation_error(operation, ResourceState)
        }
        return
      }
      None => ()
    }
  }
  let binding = unique_composite_binding(step, role => role == operand.role())
  if binding.byte_count() != operand.byte_count() ||
    binding.alignment() < operand.alignment() ||
    binding.offset_bytes() % operand.alignment() != 0L {
    raise executor_operation_error(operation, ResourceState)
  }
}

///|
fn validate_paged_decode_split_bindings(
  blueprint : PagedKvExecutionBlueprint,
  bindings : ReadOnlyArray[
    @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
  ],
  workspace_bytes : Int64,
  workspace_alignment : Int64,
  fused_qkv_readonly_span : ApprovedFusedQkvReadOnlyAttentionSpan?,
  qwen_spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> Unit raise DeviceStepError {
  let admitted_operations : Array[Int32] = []
  for binding in bindings {
    let operation = binding.operation_id()
    if binding.target() != blueprint.target() ||
      binding.profile() != blueprint.profile() ||
      binding.device_kv_layout() != blueprint.layout() ||
      admitted_operations.contains(operation.as_int()) {
      raise executor_operation_error(operation, ResourceState)
    }
    paged_decode_split_rejects_composite_overlap(
      blueprint, operation, fused_qkv_readonly_span, qwen_spans,
    )
    let step = paged_decode_split_step(blueprint, operation)
    let qwen_sidecar_shape = paged_decode_split_qwen_sidecar_shape(
      blueprint, operation, qwen_spans,
    )
    let launches = binding.contract().launches()
    if launches.length() != 2 ||
      launches[0].stage() != Partial ||
      launches[1].stage() != Merge {
      raise executor_operation_error(operation, ResourceState)
    }
    for launch in launches {
      for operand in launch.operands() {
        validate_paged_decode_split_operand(
          operation, step, operand, workspace_bytes, workspace_alignment, qwen_sidecar_shape,
        )
      }
    }
    admitted_operations.push(operation.as_int())
  }
}

///|
fn prepare_paged_decode_split_launch(
  operation : @plan.OperationId,
  step : PagedKvExecutionStep,
  function_index : Int32,
  launch : @luna_attention_multilaunch_contract.AttentionOrderedLaunchContract,
  workspace : @device.Allocation,
  qwen_sidecar : @device.Allocation?,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
) -> PreparedPagedGraphStep raise DeviceStepError {
  let arguments : Array[@device.KernelArgument] = []
  for operand in launch.operands() {
    arguments.push(
      paged_decode_split_argument(
        operation, step, operand, workspace, qwen_sidecar, descriptor, weight, activation,
        kv,
      ),
    )
  }
  {
    operation_id: operation,
    function_index,
    dimensions: composite_launch_dimensions(launch.dimensions(), operation),
    arguments: @device.KernelArguments::new(arguments) catch {
      error => raise ExecutorDevice(KernelArguments, error)
    },
  }
}
