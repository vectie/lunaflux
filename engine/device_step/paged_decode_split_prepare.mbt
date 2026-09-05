///|
const PAGED_DECODE_SPLIT_MAXIMUM_WORKSPACE_BYTES : Int64 = 268435456L

///|
const PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD : Int = 2147483647

///|
/// Initial backend-neutral split-K crossover: each partial owns at least 32
/// complete KV pages. This keeps launch/merge overhead away from short decode
/// while the startup autotune table is still being generalized.
const PAGED_DECODE_SPLIT_MINIMUM_PAGES_PER_PARTITION : Int = 32

///|
priv struct PreparedPagedDecodeSplit {
  operation_id : @plan.OperationId
  partial : PreparedPagedGraphStep
  merge : PreparedPagedGraphStep
}

///|
priv struct PagedDecodeSplitLoadedModule {
  digest : @catalog.AotArtifactDigest
  module_index : Int
}

///|
priv struct PagedDecodeSplitLoadedFunction {
  module_index : Int
  symbol : String
  function_index : Int
}

///|
priv enum PagedAttentionPhaseRoute {
  BaselineAttentionPhase
  DecodeSplitAttentionPhase
} derive(Eq)

///|
fn paged_attention_phase_route(
  phase : @luna_execution_graph_strategy.ExecutionGraphPhase,
  split_available : Bool,
) -> PagedAttentionPhaseRoute {
  if split_available && phase == DecodeGraph {
    DecodeSplitAttentionPhase
  } else {
    BaselineAttentionPhase
  }
}

///|
fn paged_decode_split_workspace_envelope(
  bindings : ReadOnlyArray[
    @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
  ],
  maximum_workspace_bytes : Int64,
) -> (Int64, Int64) raise DeviceStepError {
  let shapes : Array[(Int64, Int64)] = []
  for binding in bindings {
    if !binding.manifest_bindable() ||
      binding.contract().workspace().scalar() != F32 {
      raise InvalidExecutor(ResourceState)
    }
    let workspace = binding.contract().workspace()
    shapes.push((workspace.total_bytes(), workspace.alignment()))
  }
  paged_decode_split_workspace_shape(
    ReadOnlyArray::from_array(shapes),
    maximum_workspace_bytes,
  )
}

///|
/// Select split-K only after every partial has a useful live page interval.
fn paged_decode_split_minimum_context_tokens(
  bindings : ReadOnlyArray[
    @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
  ],
) -> Int raise DeviceStepError {
  if bindings.is_empty() {
    return PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD
  }
  let mut selected = -1
  for binding in bindings {
    let split_count = binding.contract().workspace().split_count()
    let tokens_per_page = binding.device_kv_layout().tokens_per_page()
    if split_count <= 1 || tokens_per_page <= 0 {
      raise executor_operation_error(binding.operation_id(), ResourceState)
    }
    if split_count >
      PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD /
      PAGED_DECODE_SPLIT_MINIMUM_PAGES_PER_PARTITION {
      raise executor_operation_error(binding.operation_id(), ResourceState)
    }
    let minimum_pages = split_count *
      PAGED_DECODE_SPLIT_MINIMUM_PAGES_PER_PARTITION
    if minimum_pages >
      PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD / tokens_per_page {
      raise executor_operation_error(binding.operation_id(), ResourceState)
    }
    // Buckets use an upper context bound. Selecting the 256 bucket starts
    // after 128 live tokens, where shared-KV split begins paying for merge.
    let threshold = if binding.is_compiler_readonly() {
      256
    } else {
      minimum_pages * tokens_per_page
    }
    if selected < 0 {
      selected = threshold
    } else if selected != threshold {
      raise executor_operation_error(binding.operation_id(), ResourceState)
    }
  }
  selected
}

///|
fn paged_decode_split_maximum_batch_rows(
  bindings : ReadOnlyArray[
    @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
  ],
) -> Int {
  if bindings.any(binding => binding.is_compiler_readonly()) {
    4
  } else {
    2147483647
  }
}

///|
fn paged_decode_split_workspace_shape(
  shapes : ReadOnlyArray[(Int64, Int64)],
  maximum_workspace_bytes : Int64,
) -> (Int64, Int64) raise DeviceStepError {
  if maximum_workspace_bytes <= 0L ||
    maximum_workspace_bytes > PAGED_DECODE_SPLIT_MAXIMUM_WORKSPACE_BYTES {
    raise InvalidExecutor(ResourceState)
  }
  let mut bytes = 0L
  let mut alignment = 1L
  for shape in shapes {
    if shape.0 <= 0L ||
      shape.0 > maximum_workspace_bytes ||
      shape.1 <= 0L ||
      (shape.1 & (shape.1 - 1L)) != 0L {
      raise InvalidExecutor(ResourceState)
    }
    bytes = Int64::max(bytes, shape.0)
    alignment = Int64::max(alignment, shape.1)
  }
  (bytes, alignment)
}

///|
fn[T] replace_paged_decode_split_steps(
  baseline : ReadOnlyArray[T],
  replacements : ReadOnlyArray[(Int, T, T)],
  operation : (T) -> Int,
) -> ReadOnlyArray[T] raise DeviceStepError {
  let result : Array[T] = []
  let used = FixedArray::make(replacements.length(), false)
  for step in baseline {
    let mut replacement = -1
    for index, item in replacements {
      if item.0 == operation(step) {
        if replacement >= 0 {
          raise InvalidExecutor(ResourceState)
        }
        replacement = index
      }
    }
    if replacement < 0 {
      result.push(step)
    } else {
      result.push(replacements[replacement].1)
      result.push(replacements[replacement].2)
      used[replacement] = true
    }
  }
  if used.any(value => !value) {
    raise InvalidExecutor(ResourceState)
  }
  ReadOnlyArray::from_array(result)
}

///|
fn paged_decode_split_step(
  blueprint : PagedKvExecutionBlueprint,
  operation : @plan.OperationId,
) -> PagedKvExecutionStep raise DeviceStepError {
  let mut found : PagedKvExecutionStep? = None
  for step in blueprint.steps() {
    if step.operation_id() == operation {
      if found is Some(_) || step.operation_kind() != CausalAttention {
        raise executor_operation_error(operation, ResourceState)
      }
      found = Some(step)
    }
  }
  guard found is Some(step) else {
    raise executor_operation_error(operation, ResourceState)
  }
  step
}

///|
/// Returns the Qwen fused ingress sidecar shape when split decode replaces the
/// attention operation owned by that fused span. The model graph and span
/// decide the route; the CUDA launch preparation only binds the selected
/// allocation.
fn paged_decode_split_uses_qwen_sidecar(
  blueprint : PagedKvExecutionBlueprint,
  operation : @plan.OperationId,
  span : ApprovedQwenFusedAttentionSpan,
) -> Bool raise DeviceStepError {
  let (_, _, _, attention_index) = span.operation_indices()
  guard blueprint.steps().get(attention_index) is Some(attention) else {
    raise InvalidExecutor(ResourceState)
  }
  attention.operation_id() == operation
}

///|
fn paged_decode_split_qwen_sidecar_shape(
  blueprint : PagedKvExecutionBlueprint,
  operation : @plan.OperationId,
  qwen_spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> (Int64, Int64)? raise DeviceStepError {
  let mut selected : (Int64, Int64)? = None
  for span in qwen_spans {
    if paged_decode_split_uses_qwen_sidecar(blueprint, operation, span) {
      if selected is Some(_) {
        raise executor_operation_error(operation, ResourceState)
      }
      selected = Some(qwen_fused_attention_sidecar_shape(blueprint, span))
    }
  }
  selected
}

///|
fn paged_decode_split_rejects_composite_overlap(
  blueprint : PagedKvExecutionBlueprint,
  operation : @plan.OperationId,
  fused_qkv_readonly_span : ApprovedFusedQkvReadOnlyAttentionSpan?,
  _qwen_spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> Unit raise DeviceStepError {
  match fused_qkv_readonly_span {
    Some(span) => {
      let (_, _, attention_index) = span.operation_indices()
      guard blueprint.steps().get(attention_index) is Some(attention) else {
        raise InvalidExecutor(ResourceState)
      }
      if attention.operation_id() == operation {
        raise executor_operation_error(operation, ResourceState)
      }
    }
    None => ()
  }
  // Qwen's fused ingress and readonly attention deliberately share the
  // attention operation. Decode split-K replaces only the readonly decode
  // launch; prefill retains that launch and ingress keeps ownership of KV
  // writes. The phase-specific executor prevents simultaneous execution.
}

///|
fn paged_decode_split_loaded_module(
  context : @device.Context,
  binding : @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
  loaded : Array[PagedDecodeSplitLoadedModule],
  resources : PagedGraphResources,
) -> Int raise DeviceStepError {
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
  resources.modules.push({ resource: module_, open: true })
  loaded.push({ digest, module_index })
  module_index
}

///|
fn paged_decode_split_loaded_function(
  module_index : Int,
  symbol : String,
  loaded : Array[PagedDecodeSplitLoadedFunction],
  resources : PagedGraphResources,
) -> Int raise DeviceStepError {
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
  resources.functions.push({ resource: function, module_index, open: true })
  loaded.push({ module_index, symbol, function_index })
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
  let admitted_operations : Array[Int] = []
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
  function_index : Int,
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

///|
fn prepare_paged_decode_split_steps(
  context : @device.Context,
  blueprint : PagedKvExecutionBlueprint,
  baseline : ReadOnlyArray[PreparedPagedGraphStep],
  bindings : ReadOnlyArray[
    @luna_cuda_paged_attention_aot.PagedAttentionDecodeSplitRuntimeCudaBinding,
  ],
  maximum_workspace_bytes : Int64,
  descriptor : DeviceStepResources,
  weight : @device.Allocation,
  activation : @device.Allocation,
  kv : @device.Allocation,
  resources : PagedGraphResources,
  fused_qkv_readonly_span : ApprovedFusedQkvReadOnlyAttentionSpan?,
  qwen_spans : ReadOnlyArray[ApprovedQwenFusedAttentionSpan],
) -> ReadOnlyArray[PreparedPagedGraphStep]? raise DeviceStepError {
  if bindings.is_empty() {
    return None
  }
  let (workspace_bytes, workspace_alignment) = paged_decode_split_workspace_envelope(
    bindings, maximum_workspace_bytes,
  )
  validate_paged_decode_split_bindings(
    blueprint, bindings, workspace_bytes, workspace_alignment, fused_qkv_readonly_span,
    qwen_spans,
  )
  if resources.decode_split_workspace is Some(_) {
    raise InvalidExecutor(ResourceState)
  }
  let workspace = context.allocate(workspace_bytes) catch {
    error => raise ExecutorDevice(DecodeSplitWorkspace, error)
  }
  resources.decode_split_workspace = Some({ resource: workspace, open: true })
  if workspace.byte_count() != workspace_bytes {
    raise InvalidExecutor(AllocationSize)
  }
  context.validate_allocation_region(
    workspace,
    offset=0L,
    byte_count=workspace_bytes,
    alignment=workspace_alignment,
  ) catch {
    error => raise ExecutorDevice(DecodeSplitWorkspace, error)
  }
  let loaded_modules : Array[PagedDecodeSplitLoadedModule] = []
  let loaded_functions : Array[PagedDecodeSplitLoadedFunction] = []
  let prepared : Array[PreparedPagedDecodeSplit] = []
  for binding in bindings {
    let operation = binding.operation_id()
    if binding.target() != blueprint.target() ||
      binding.profile() != blueprint.profile() ||
      binding.device_kv_layout() != blueprint.layout() {
      raise executor_operation_error(operation, ResourceState)
    }
    if prepared.any(item => item.operation_id == operation) {
      raise executor_operation_error(operation, ResourceState)
    }
    paged_decode_split_rejects_composite_overlap(
      blueprint, operation, fused_qkv_readonly_span, qwen_spans,
    )
    let step = paged_decode_split_step(blueprint, operation)
    let qwen_sidecar_shape = paged_decode_split_qwen_sidecar_shape(
      blueprint, operation, qwen_spans,
    )
    let qwen_sidecar = match qwen_sidecar_shape {
      Some(shape) => {
        guard resources.qwen_fused_attention_sidecar is Some(sidecar) &&
          sidecar.open &&
          sidecar.resource.byte_count() == shape.0 else {
          raise executor_operation_error(operation, ResourceState)
        }
        Some(sidecar.resource)
      }
      None => None
    }
    let module_index = paged_decode_split_loaded_module(
      context, binding, loaded_modules, resources,
    )
    let launches = binding.contract().launches()
    if launches.length() != 2 ||
      launches[0].stage() != Partial ||
      launches[1].stage() != Merge {
      raise executor_operation_error(operation, ResourceState)
    }
    let partial_function = paged_decode_split_loaded_function(
      module_index,
      launches[0].function_symbol(),
      loaded_functions,
      resources,
    )
    let merge_function = paged_decode_split_loaded_function(
      module_index,
      launches[1].function_symbol(),
      loaded_functions,
      resources,
    )
    prepared.push({
      operation_id: operation,
      partial: prepare_paged_decode_split_launch(
        operation,
        step,
        partial_function,
        launches[0],
        workspace,
        qwen_sidecar,
        descriptor,
        weight,
        activation,
        kv,
      ),
      merge: prepare_paged_decode_split_launch(
        operation,
        step,
        merge_function,
        launches[1],
        workspace,
        qwen_sidecar,
        descriptor,
        weight,
        activation,
        kv,
      ),
    })
  }
  let replacements : Array[
    (Int, PreparedPagedGraphStep, PreparedPagedGraphStep),
  ] = []
  for item in prepared {
    replacements.push((item.operation_id.as_int(), item.partial, item.merge))
  }
  Some(
    replace_paged_decode_split_steps(
      baseline,
      ReadOnlyArray::from_array(replacements),
      step => step.operation_id.as_int(),
    ),
  )
}

///|
fn append_paged_sampling_suffix(
  model_steps : ReadOnlyArray[PreparedPagedGraphStep],
  complete_steps : ReadOnlyArray[PreparedPagedGraphStep],
  decode_model_steps : ReadOnlyArray[PreparedPagedGraphStep],
) -> ReadOnlyArray[PreparedPagedGraphStep] raise DeviceStepError {
  if complete_steps.length() < model_steps.length() {
    raise InvalidExecutor(ResourceState)
  }
  let result : Array[PreparedPagedGraphStep] = []
  for step in decode_model_steps {
    result.push(step)
  }
  for index in model_steps.length()..<complete_steps.length() {
    result.push(complete_steps[index])
  }
  ReadOnlyArray::from_array(result)
}
