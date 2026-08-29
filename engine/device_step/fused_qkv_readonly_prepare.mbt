///|
fn prepare_composite_function(
  context : @device.Context,
  module_bytes : Bytes,
  symbol : String,
  resources : PagedGraphResources,
) -> Int raise DeviceStepError {
  let module_index = resources.modules.length()
  let loaded = context.load_module(module_bytes) catch {
    error => raise ExecutorDevice(Module, error)
  }
  resources.modules.push({ resource: loaded, open: true })
  let function = loaded.load_function(@utf8.encode(symbol)) catch {
    error => raise ExecutorDevice(Function, error)
  }
  let function_index = resources.functions.length()
  resources.functions.push({ resource: function, module_index, open: true })
  function_index
}

///|
fn prepare_fused_qkv_readonly_resources(
  context : @device.Context,
  span : ApprovedFusedQkvReadOnlyAttentionSpan,
  resources : PagedGraphResources,
) -> (Int, Int) raise DeviceStepError {
  let qkv = prepare_composite_function(
    context,
    span.qkv_module_bytes(),
    span.qkv_function_symbol(),
    resources,
  )
  let readonly_index = prepare_composite_function(
    context,
    span.readonly_module_bytes(),
    span.readonly_function_symbol(),
    resources,
  )
  (qkv, readonly_index)
}

///|
fn unique_composite_binding(
  step : PagedKvExecutionStep,
  accepts : (@launch_contract.AotOperandRole) -> Bool,
) -> PagedKvOperandBinding raise DeviceStepError {
  let mut found : PagedKvOperandBinding? = None
  for binding in step.operands() {
    if accepts(binding.role()) {
      if found is Some(_) {
        raise executor_operation_error(step.operation_id(), ResourceState)
      }
      found = Some(binding)
    }
  }
  guard found is Some(binding) else {
    raise executor_operation_error(step.operation_id(), ResourceState)
  }
  binding
}

///|
fn composite_binding_argument(
  binding : PagedKvOperandBinding,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
) -> @device.KernelArgument raise DeviceStepError {
  fused_residual_argument(binding, descriptor, weight, activation, kv)
}

///|
fn composite_launch_dimensions(
  source : @launch_contract.AotLaunchDimensions,
  operation : @plan.OperationId,
) -> @device.KernelLaunchDimensions raise DeviceStepError {
  @device.KernelLaunchDimensions::new(
    grid_x=source.grid_x(),
    grid_y=source.grid_y(),
    grid_z=source.grid_z(),
    block_x=source.block_x(),
    block_y=source.block_y(),
    block_z=source.block_z(),
    shared_memory_bytes=source.shared_memory_bytes(),
  ) catch {
    _ => raise executor_operation_error(operation, LaunchDimensions)
  }
}

///|
fn prepare_composite_launch(
  operation : @plan.OperationId,
  function_index : Int,
  launch : @launch_contract.AotLaunchDimensions,
  bindings : ArrayView[PagedKvOperandBinding],
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
) -> PreparedPagedGraphStep raise DeviceStepError {
  let arguments : Array[@device.KernelArgument] = []
  for binding in bindings {
    arguments.push(
      composite_binding_argument(binding, descriptor, weight, activation, kv),
    )
  }
  let packed = @device.KernelArguments::new(arguments) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
  {
    operation_id: operation,
    function_index,
    dimensions: composite_launch_dimensions(launch, operation),
    arguments: packed,
  }
}

///|
fn fused_qkv_production_argument_count() -> Int {
  13
}

///|
fn readonly_attention_production_argument_count() -> Int {
  10
}

///|
fn prepare_fused_qkv_readonly_steps(
  blueprint : PagedKvExecutionBlueprint,
  span : ApprovedFusedQkvReadOnlyAttentionSpan,
  qkv_function : Int,
  readonly_function : Int,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
) -> (PreparedPagedGraphStep, PreparedPagedGraphStep) raise DeviceStepError {
  let (qkv_index, rope_index, attention_index) = span.operation_indices()
  guard blueprint.steps().get(qkv_index) is Some(qkv_step) &&
    blueprint.steps().get(rope_index) is Some(rope_step) &&
    blueprint.steps().get(attention_index) is Some(attention_step) else {
    raise InvalidExecutor(ResourceState)
  }
  let qkv_counts = unique_composite_binding(qkv_step, role => {
    role == RuntimeInput(StepCounts)
  })
  let rope_counts = unique_composite_binding(rope_step, role => {
    role == RuntimeInput(StepCounts)
  })
  let attention_counts = unique_composite_binding(attention_step, role => {
    role == RuntimeInput(StepCounts)
  })
  let query_positions = unique_composite_binding(attention_step, role => {
    role == RuntimeInput(QueryPositions)
  })
  let rope_positions = unique_composite_binding(rope_step, role => {
    role == RuntimeInput(QueryPositions)
  })
  if !same_composite_region(qkv_counts, attention_counts) ||
    !same_composite_region(rope_counts, attention_counts) ||
    !same_composite_region(rope_positions, query_positions) {
    raise InvalidExecutor(ResourceState)
  }
  let runtime = [
    attention_counts,
    query_positions,
    unique_composite_binding(attention_step, role => {
      role == RuntimeInput(QueryRowOffsets)
    }),
    unique_composite_binding(attention_step, role => {
      role == RuntimeInput(SequenceLengths)
    }),
    unique_composite_binding(attention_step, role => {
      role == RuntimeInput(PageTableOffsets)
    }),
    unique_composite_binding(attention_step, role => {
      role == RuntimeInput(PageTablePageIndices)
    }),
  ]
  let qkv_input = unique_composite_binding(qkv_step, role => {
    role is ActivationInput(_)
  })
  let weights : Array[PagedKvOperandBinding] = []
  for binding in qkv_step.operands() {
    if binding.role() is WeightInput(_) {
      weights.push(binding)
    }
  }
  if weights.length() != 3 {
    raise InvalidExecutor(ResourceState)
  }
  let rotated = unique_composite_binding(rope_step, role => {
    role is ActivationOutput(_)
  })
  let readonly_input = unique_composite_binding(attention_step, role => {
    role is ActivationInput(_)
  })
  if !same_composite_region(rotated, readonly_input) {
    raise InvalidExecutor(ResourceState)
  }
  let output = unique_composite_binding(attention_step, role => {
    role is ActivationOutput(_)
  })
  let key_cache = unique_composite_binding(attention_step, role => {
    role is PersistentKvComponent(_, Key)
  })
  let value_cache = unique_composite_binding(attention_step, role => {
    role is PersistentKvComponent(_, Value)
  })
  let qkv_bindings : Array[PagedKvOperandBinding] = []
  for binding in runtime {
    qkv_bindings.push(binding)
  }
  qkv_bindings.push(qkv_input)
  for binding in weights {
    qkv_bindings.push(binding)
  }
  qkv_bindings.push(rotated)
  qkv_bindings.push(key_cache)
  qkv_bindings.push(value_cache)
  let readonly_bindings : Array[PagedKvOperandBinding] = []
  for binding in runtime {
    readonly_bindings.push(binding)
  }
  for binding in [readonly_input, output, key_cache, value_cache] {
    readonly_bindings.push(binding)
  }
  if qkv_bindings.length() != fused_qkv_production_argument_count() ||
    readonly_bindings.length() != readonly_attention_production_argument_count() {
    raise InvalidExecutor(ResourceState)
  }
  (
    prepare_composite_launch(
      qkv_step.operation_id(),
      qkv_function,
      span.qkv_launch(),
      qkv_bindings,
      descriptor,
      weight,
      activation,
      kv,
    ),
    prepare_composite_launch(
      attention_step.operation_id(),
      readonly_function,
      span.readonly_launch(),
      readonly_bindings,
      descriptor,
      weight,
      activation,
      kv,
    ),
  )
}

///|
fn same_composite_region(
  left : PagedKvOperandBinding,
  right : PagedKvOperandBinding,
) -> Bool {
  left.space() == right.space() &&
  left.offset_bytes() == right.offset_bytes() &&
  left.byte_count() == right.byte_count() &&
  left.alignment() == right.alignment()
}

///|
fn[Step] replace_fused_qkv_readonly_span(
  baseline : ReadOnlyArray[Step],
  fused_qkv : Step,
  readonly_attention : Step,
  qkv_index : Int,
  rope_index : Int,
  attention_index : Int,
) -> ReadOnlyArray[Step] {
  let output : Array[Step] = []
  for index, step in baseline {
    if index == qkv_index {
      output.push(fused_qkv)
      output.push(readonly_attention)
    } else if index != rope_index && index != attention_index {
      output.push(step)
    }
  }
  ReadOnlyArray::from_array(output)
}
