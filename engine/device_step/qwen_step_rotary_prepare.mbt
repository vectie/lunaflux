///|
/// A preparation effect has a distinct identity after the semantic graph.
/// It is not a model operator and must not participate in projection selection.
fn qwen_step_rotary_operation(
  blueprint : PagedKvExecutionBlueprint,
) -> @plan.OperationId raise DeviceStepError {
  let mut greatest = -1
  for step in blueprint.steps() {
    greatest = Int::max(greatest, step.operation_id().as_int())
  }
  if greatest == 2147483647 {
    raise InvalidExecutor(ResourceState)
  }
  @plan.OperationId::new(greatest + 1) catch {
    _ => raise InvalidExecutor(ResourceState)
  }
}

///|
/// Preserve the original rotated-QKV region; the immutable pair region is an
/// aligned non-overlapping append in the same deterministic resource owner.
fn qwen_step_rotary_sidecar_shape(
  rotated : (Int64, Int64),
  cache : @elementwise_physical_ir.StepRotaryCachePlan?,
) -> (Int64, Int64) raise DeviceStepError {
  guard cache is Some(plan) else { return rotated }
  let bytes = plan.byte_count()
  if rotated.0 <= 0L ||
    rotated.0 % 8L != 0L ||
    rotated.0 > 9223372036854775807L - bytes {
    raise InvalidExecutor(AllocationSize)
  }
  (rotated.0 + bytes, Int64::max(rotated.1, 8L))
}

///|
fn qwen_step_rotary_argument(
  sidecar : @device.Allocation,
  cache : @elementwise_physical_ir.StepRotaryCachePlan,
) -> @device.KernelArgument raise DeviceStepError {
  @device.KernelArgument::device_region(
    sidecar,
    offset=sidecar.byte_count() - cache.byte_count(),
    byte_count=cache.byte_count(),
    alignment=8L,
  ) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
}

///|
fn qwen_step_rotary_bytes_for_operation(
  blueprint : PagedKvExecutionBlueprint,
  operation : @plan.OperationId,
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> Int64 {
  for span in spans {
    if blueprint.steps()[span.attention_index].operation_id() == operation {
      return match span.step_rotary_cache {
        Some(cache) => cache.byte_count()
        None => 0L
      }
    }
  }
  0L
}

///|
fn qwen_step_rotary_launch(
  cache : @elementwise_physical_ir.StepRotaryCachePlan,
) -> @launch_contract.AotLaunchDimensions raise DeviceStepError {
  let (tokens, pairs) = cache.geometry()
  @launch_contract.AotLaunchDimensions::new(
    grid_x=((tokens.to_int64() * pairs.to_int64() + 127L) / 128L).to_int(),
    grid_y=1,
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise InvalidExecutor(GraphBucket)
  }
}

///|
fn prepare_qwen_step_rotary(
  blueprint : PagedKvExecutionBlueprint,
  span : ApprovedQwenFusedAttentionSpan,
  functions : PreparedQwenFusedAttentionResources,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
) -> PreparedPagedGraphStep? raise DeviceStepError {
  guard span.step_rotary_cache is Some(cache) else { return None }
  guard functions.rotary_prepare_function is Some(function_index) else {
    raise InvalidExecutor(ResourceState)
  }
  let attention = blueprint.steps()[span.attention_index]
  let arguments : Array[@device.KernelArgument] = []
  let roles : Array[@launch_contract.AotOperandRole] = [
    RuntimeInput(StepCounts),
    RuntimeInput(QueryPositions),
  ]
  for role in roles {
    let binding = unique_composite_binding(attention, candidate => {
      candidate == role
    })
    arguments.push(
      composite_binding_argument(binding, descriptor, weight, activation, kv),
    )
  }
  arguments.push(qwen_step_rotary_argument(functions.sidecar, cache))
  let operation = qwen_step_rotary_operation(blueprint)
  let packed = @device.KernelArguments::new(arguments) catch {
    error => raise ExecutorDevice(KernelArguments, error)
  }
  Some({
    operation_id: operation,
    function_index,
    dimensions: composite_launch_dimensions(
      qwen_step_rotary_launch(cache),
      operation,
    ),
    arguments: packed,
  })
}

///|
fn[T] prepend_qwen_step_rotary(
  steps : ReadOnlyArray[T],
  preparation : T?,
) -> ReadOnlyArray[T] {
  guard preparation is Some(prepared) else { return steps }
  let result : Array[T] = [prepared]
  for step in steps {
    result.push(step)
  }
  ReadOnlyArray::from_array(result)
}

///|
fn prepare_qwen_rotary_bucket_launches(
  blueprint : PagedKvExecutionBlueprint,
  steps : ReadOnlyArray[PreparedPagedGraphStep],
  spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
  launches : FixedArray[PagedBucketStepLaunch],
) -> Unit raise DeviceStepError {
  guard spans.get(0) is Some(span) && span.step_rotary_cache is Some(cache) else {
    return
  }
  let (_, pairs) = cache.geometry()
  if pairs <= 0 {
    raise InvalidExecutor(GraphBucket)
  }
  let index = qwen_bucket_step_index(
    steps,
    qwen_step_rotary_operation(blueprint),
  )
  if index != 0 {
    raise InvalidExecutor(GraphBucket)
  }
  launches[index] = QueryCellCappedGridX(qwen_step_rotary_launch(cache), pairs)
}
