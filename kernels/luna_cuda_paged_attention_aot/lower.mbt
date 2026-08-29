///|
/// Lowers one non-bindable paged-attention candidate to deterministic
/// reference-grade CUDA source and a closed offline recipe.
///
/// The one kernel covers mixed scheduler batches: prefill rows precede decode
/// rows exactly as `StepCounts` declares. It writes current K/V into the
/// canonical split cache, while reading current-chunk K/V directly from the
/// immutable QKV activation. Consequently correctness does not rely on a
/// grid-wide synchronization or cross-block cache visibility.
pub fn lower_paged_attention_cuda_aot_candidate(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  limits : PagedAttentionAotLimits,
) -> PagedAttentionCudaLowering raise PagedAttentionAotError {
  let (query_heads, kv_heads, head_dimension, page_stride, page_capacity) = paged_attention_validate_candidate(
    operation, profile, layout, operands, target, compiler,
  )
  let function_symbol = paged_attention_symbol(entry_point_id)
  let dimensions = paged_attention_expected_dimensions(profile, query_heads)
  let source = paged_attention_cuda_source(
    function_symbol,
    query_heads,
    kv_heads,
    head_dimension,
    layout.tokens_per_page(),
    page_stride,
    page_capacity,
    profile.max_query_rows(),
    profile.max_query_tokens(),
    profile.max_page_table_entries(),
  )
  if source.length() > limits.max_source_bytes {
    raise LimitExceeded(Source)
  }
  let source_digest = PagedAttentionCudaSourceDigest(
    paged_attention_sha256(source),
  )
  let recipe = paged_attention_recipe(
    operation,
    profile,
    entry_point_id,
    layout.tokens_per_page(),
    operands,
    dimensions,
    target,
    compiler,
    source_digest,
    function_symbol,
    query_heads,
    kv_heads,
    head_dimension,
    page_stride,
    page_capacity,
  )
  if recipe.length() > limits.max_recipe_bytes {
    raise LimitExceeded(Recipe)
  }
  {
    target,
    operation_id: operation.id(),
    entry_point_id,
    profile,
    query_heads,
    kv_heads,
    head_dimension,
    tokens_per_page: layout.tokens_per_page(),
    page_stride_bytes: page_stride,
    physical_page_capacity: page_capacity,
    operands: ReadOnlyArray::from_array(operands),
    dimensions,
    compiler,
    function_symbol,
    source,
    source_digest,
    recipe,
    recipe_digest: PagedAttentionCudaRecipeDigest(
      paged_attention_sha256(recipe),
    ),
  }
}

///|
/// Emits the release producer's operation-neutral one-entry attention module.
/// Layer identity remains in the candidate recipe, operands, catalog
/// semantics, and strict binding, but does not create duplicate device code.
pub fn lower_reusable_paged_attention_cuda_aot_candidate(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  limits : PagedAttentionAotLimits,
) -> PagedAttentionCudaLowering raise PagedAttentionAotError {
  let entry_point_id = @catalog.AotKernelEntryPointId::from_stable_int(1) catch {
    _ => raise Invalid(Operation)
  }
  lower_paged_attention_cuda_aot_candidate(
    operation, profile, layout, entry_point_id, target, compiler, operands, limits,
  )
}
