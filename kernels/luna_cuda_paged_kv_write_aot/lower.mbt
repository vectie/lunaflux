///|
pub fn lower_positioned_paged_kv_write_cuda_aot_candidate(
  plan : @plan.ModelPlan,
  attention : @plan.PlanOperation,
  input : PositionedQkvActivation,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  fallback : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  limits : PagedKvWriteAotLimits,
) -> PagedKvWriteCudaCandidate raise PagedKvWriteAotError {
  let (query_heads, kv_heads, head_dimension, layer) = paged_kv_write_validate(
    plan, attention, input, profile, layout, fallback, target, compiler, operands,
  )
  let kv_width = kv_heads * head_dimension
  let blocks = profile.max_query_tokens().to_int64()
  if blocks > 2147483647L {
    raise SizeOverflow(Shape)
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=blocks.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise Unsupported(Shape)
  }
  let symbol = "lunaflux_positioned_paged_kv_write_bf16_op" +
    attention.id().as_int().to_string() +
    "_qualification_v1"
  let dispatch_canary : UInt = 22321U
  if profile.max_query_tokens().to_int64() * dispatch_canary.to_int64() >
    4294967295L {
    raise SizeOverflow(Shape)
  }
  let source = paged_kv_write_source(
    symbol,
    query_heads,
    kv_heads,
    head_dimension,
    kv_width,
    layout.tokens_per_page(),
    layout.total_page_count(),
    layout.page_stride_bytes(),
    profile.max_query_rows(),
    profile.max_query_tokens(),
    profile.max_page_table_entries(),
    dispatch_canary,
  )
  if source.length() > limits.max_source_bytes {
    raise LimitExceeded(Source)
  }
  let source_digest = PagedKvWriteDigest(paged_kv_write_sha256(source))
  let recipe = paged_kv_write_recipe(
    plan, attention, input, layer, profile, layout, target, compiler, operands, dimensions,
    symbol, dispatch_canary, source_digest, fallback,
  )
  if recipe.length() > limits.max_recipe_bytes {
    raise LimitExceeded(Recipe)
  }
  let recipe_digest = PagedKvWriteDigest(paged_kv_write_sha256(recipe))
  let candidate_digest = PagedKvWriteDigest(
    paged_kv_write_sha256(
      @utf8.encode(
        "schema=lunaflux-positioned-paged-kv-write-identity.v1\n" +
        "source_sha256=" +
        source_digest.as_hex() +
        "\nrecipe_sha256=" +
        recipe_digest.as_hex() +
        "\n",
      ),
    ),
  )
  {
    model_content: plan.identity().content().as_hex(),
    model_plan: plan.identity().plan().as_hex(),
    attention_operation: attention.id(),
    positioned_qkv: input.qkv,
    kv_layer: layer,
    target,
    profile,
    compiler,
    tokens_per_page: layout.tokens_per_page(),
    total_page_count: layout.total_page_count(),
    page_stride_bytes: layout.page_stride_bytes(),
    query_heads,
    kv_heads,
    head_dimension,
    kv_width,
    operands: ReadOnlyArray::from_array(operands),
    dimensions,
    symbol,
    abi: PositionedQkvPagedWriteBf16V1,
    dispatch_canary,
    fallback_source: fallback.source_digest().as_hex(),
    fallback_recipe: fallback.recipe_digest().as_hex(),
    source,
    source_digest,
    recipe,
    recipe_digest,
    candidate_digest,
  }
}
