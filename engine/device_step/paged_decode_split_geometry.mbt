///|
const PAGED_DECODE_SPLIT_MAXIMUM_WORKSPACE_BYTES : Int64 = 268435456L

///|
const PAGED_DECODE_SPLIT_DISABLED_CONTEXT_THRESHOLD : Int32 = 2147483647

///|
/// Initial backend-neutral split-K crossover: each partial owns at least 32
/// complete KV pages. This keeps launch/merge overhead away from short decode
/// while the startup autotune table is still being generalized.
const PAGED_DECODE_SPLIT_MINIMUM_PAGES_PER_PARTITION : Int32 = 32

///|
priv struct PreparedPagedDecodeSplit {
  operation_id : @plan.OperationId
  partial : PreparedPagedGraphStep
  merge : PreparedPagedGraphStep
  partial_dimensions : @launch_contract.AotLaunchDimensions
  merge_dimensions : @launch_contract.AotLaunchDimensions
}

///|
priv struct PagedDecodeSplitLoadedModule {
  digest : @catalog.AotArtifactDigest
  module_index : Int32
}

///|
priv struct PagedDecodeSplitLoadedFunction {
  module_index : Int32
  symbol : String
  function_index : Int32
}

///|
priv enum PagedAttentionPhaseRoute {
  BaselineAttentionPhase
  DecodeSplitAttentionPhase
} derive(Eq)

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
) -> Int32 raise DeviceStepError {
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
    let threshold = if binding.is_blockwise_readonly() {
      // Full-chain measurements may override this startup fallback. The
      // blockwise merge is useful only after a substantial history exists.
      4096
    } else if binding.is_compiler_readonly() {
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
) -> Int32 {
  if bindings.any(binding => binding.is_blockwise_readonly()) {
    // Current paired measurements justify only the one-request long-history
    // route; compiling other row buckets does not assert they are faster.
    1
  } else if bindings.any(binding => binding.is_compiler_readonly()) {
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
