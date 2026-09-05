///|
/// Lowers an independent two-stage decode split-K candidate.
///
/// The returned bundle is complete deterministic CUDA AOT input, but remains
/// non-bindable until a versioned launch contract explicitly carries its two
/// launches and F32 workspace operand.
pub fn lower_paged_attention_decode_split_cuda_aot_candidate(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  partial_entry_point_id : @catalog.AotKernelEntryPointId,
  merge_entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  maximum_key_value_splits : Int,
  maximum_workspace_bytes : Int64,
  limits : PagedAttentionAotLimits,
  compiler_readonly? : Bool = false,
) -> PagedAttentionDecodeSplitCudaBundle raise PagedAttentionAotError {
  if partial_entry_point_id == merge_entry_point_id ||
    maximum_key_value_splits <= 0 ||
    maximum_workspace_bytes <= 0L {
    raise Invalid(Operation)
  }
  let (query_heads, kv_heads, head_dimension, page_stride, page_capacity) = paged_attention_validate_candidate(
    operation, profile, layout, operands, target, compiler,
  )
  if profile.max_page_table_entries() > 2147483647 / layout.tokens_per_page() {
    raise SizeOverflow(Profile)
  }
  let maximum_context_tokens = profile.max_page_table_entries() *
    layout.tokens_per_page()
  let shape = @luna_attention_strategy.AttentionShape::new(
    query_heads~,
    kv_heads~,
    head_dimension~,
    tokens_per_page=layout.tokens_per_page(),
  ) catch {
    _ => raise Invalid(Shape)
  }
  let dispatch_plan = @luna_attention_strategy.plan_attention_dispatch(
    shape,
    maximum_context_tokens,
    profile.max_query_tokens(),
    maximum_key_value_splits,
  ) catch {
    _ => raise Invalid(Profile)
  }
  let workspace_plan = @luna_attention_strategy.plan_decode_attention_workspace(
    shape,
    dispatch_plan.decode(),
    profile.max_query_rows(),
    maximum_context_tokens,
    maximum_workspace_bytes,
  ) catch {
    _ => raise Unsupported(Profile)
  }
  let version = if compiler_readonly { "v2" } else { "v1" }
  let partial_function_symbol = "lunaflux_paged_attention_bf16_decode_split_partial_" +
    version +
    "_ep_" +
    partial_entry_point_id.as_int().to_string()
  let merge_function_symbol = "lunaflux_paged_attention_bf16_decode_split_merge_" +
    version +
    "_ep_" +
    merge_entry_point_id.as_int().to_string()
  let (_, partial_heads, threads, shared) = if compiler_readonly {
    compiler_decode_launch_geometry(
      shape,
      maximum_context_tokens,
      workspace_plan.split_count(),
    )
  } else {
    (1, query_heads, 32, 0L)
  }
  let partial_dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=profile.max_query_rows(),
    grid_y=partial_heads,
    grid_z=workspace_plan.split_count(),
    block_x=threads,
    block_y=1,
    block_z=1,
    shared_memory_bytes=shared.to_int(),
  ) catch {
    _ => raise Invalid(Dimensions)
  }
  let merge_dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=profile.max_query_rows(),
    grid_y=query_heads,
    grid_z=1,
    block_x=threads,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise Invalid(Dimensions)
  }
  let compiled = if compiler_readonly {
    Some(
      compile_readonly_decode_split(
        shape,
        profile,
        target,
        page_stride,
        page_capacity,
        maximum_context_tokens,
        workspace_plan,
        partial_function_symbol,
        merge_function_symbol,
        limits.max_source_bytes,
      ),
    )
  } else {
    None
  }
  let source = match compiled {
    Some(compiled) => compiled.source().bytes()
    None =>
      paged_attention_decode_split_cuda_source(
        partial_function_symbol,
        merge_function_symbol,
        query_heads,
        kv_heads,
        head_dimension,
        layout.tokens_per_page(),
        page_stride,
        page_capacity,
        profile.max_query_rows(),
        profile.max_query_tokens(),
        profile.max_page_table_entries(),
        workspace_plan,
      )
  }
  if source.length() > limits.max_source_bytes {
    raise LimitExceeded(Source)
  }
  let source_digest = PagedAttentionCudaSourceDigest(
    paged_attention_sha256(source),
  )
  let base_recipe = paged_attention_decode_split_recipe(
    operation,
    profile,
    target,
    compiler,
    partial_entry_point_id,
    merge_entry_point_id,
    partial_function_symbol,
    merge_function_symbol,
    partial_dimensions,
    merge_dimensions,
    source_digest,
    workspace_plan,
    compiler_readonly~,
  )
  let recipe = match compiled {
    None => base_recipe
    Some(compiled) =>
      @utf8.encode(
        (@utf8.decode(base_recipe) catch { _ => raise Invalid(Recipe) }) +
        "compiler_pipeline_sha256=" +
        compiled.digest().as_hex() +
        "\n" +
        "kv_effect=readonly\nworkspace_index=sequence-row\n",
      )
  }
  if recipe.length() > limits.max_recipe_bytes {
    raise LimitExceeded(Recipe)
  }
  {
    compiler_readonly,
    dispatch_plan,
    workspace_plan,
    target,
    operation_id: operation.id(),
    profile,
    layout,
    query_heads,
    kv_heads,
    head_dimension,
    tokens_per_page: layout.tokens_per_page(),
    page_stride_bytes: page_stride,
    physical_page_capacity: page_capacity,
    operands: ReadOnlyArray::from_array(operands),
    partial_entry_point_id,
    merge_entry_point_id,
    compiler,
    partial_function_symbol,
    merge_function_symbol,
    partial_dimensions,
    merge_dimensions,
    source,
    source_digest,
    recipe,
    recipe_digest: PagedAttentionCudaRecipeDigest(
      paged_attention_sha256(recipe),
    ),
  }
}
