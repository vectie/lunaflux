///|
pub fn lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_candidate(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  qkv_fallback : @luna_cuda_projection_aot.ProjectionCudaLowering,
  qk_norm_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  rope_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  kv_write_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  limits : FusedParallelAotLimits,
  fold_choices~ : ArrayView[@luna_projection_tile_compiler.ProjectionFoldChoice] = [],
  ingress_policy? : @resource_policy.Policy,
  maximum_shared_bytes~ : Int64 = 49152L,
) -> FusedQwenQkvQkNormRopeKvWriteCudaCandidate raise FusedParallelAotError {
  let profile = qkv_fallback.profile()
  fused_validate_common(profile, target, compiler)
  if qkv_fallback.family() != QkvProjection ||
    qk_norm_fallback.family() != QkRmsNorm ||
    rope_fallback.family() != PositionedRotary ||
    kv_write_fallback.family() != KvCacheWrite ||
    qkv_fallback.target() != target ||
    qk_norm_fallback.target() != target ||
    rope_fallback.target() != target ||
    kv_write_fallback.target() != target ||
    qkv_fallback.compiler_policy() != compiler ||
    qk_norm_fallback.profile() != profile ||
    rope_fallback.profile() != profile ||
    kv_write_fallback.profile() != profile {
    raise FusedParallelMismatch(ReferenceFallback)
  }
  let semantic = @luna_fusion_plan.match_qkv_qknorm_positioned_rope_paged_kv_write(
    plan,
    qkv_fallback.operation_id(),
  ) catch {
    _ => raise FusedParallelMismatch(Operation)
  }
  ignore(semantic.cut(FullyFused)) catch {
    _ => raise FusedParallelMismatch(Operation)
  }
  let (query_heads, kv_heads, head_dimension) = semantic.attention_shape()
  if head_dimension % 16 != 0 {
    raise FusedParallelUnsupported(StaticResources)
  }
  if head_dimension > FUSED_PARALLEL_THREADS ||
    !fused_is_power_of_two(head_dimension) ||
    !layout.is_canonical_for(plan.identity(), plan.kv_cache_geometry()) ||
    layout.geometry().dtype() != BFloat16 ||
    layout.geometry().layout() != LayerMajorSplitKeyValueV1 ||
    layout.geometry().head_dimension() != head_dimension ||
    layout.tokens_per_page() < 8 ||
    layout.tokens_per_page() > 256 {
    raise FusedParallelMismatch(KvLayout)
  }
  qwen_fused_validate_fallbacks(
    plan, layout, semantic, qkv_fallback, qk_norm_fallback, rope_fallback, kv_write_fallback,
  )
  fused_validate_operand_alignments(qkv_fallback.operands(), 16L)
  fused_validate_operand_alignments(qk_norm_fallback.operands(), 16L)
  fused_validate_operand_alignments(rope_fallback.operands(), 16L)
  fused_validate_operand_alignments(
    kv_write_fallback.operands(),
    layout.segment_alignment_bytes(),
  )
  let tile_epilogue = @luna_projection_tile_compiler.ProjectionAttentionIngressEpilogue::new(
    query_heads~,
    kv_heads~,
    head_dimension~,
    norm_epsilon=semantic.norm_epsilon(),
    rope_theta=semantic.rope_theta(),
  ) catch {
    _ => raise FusedParallelMismatch(Operation)
  }
  let request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::with_attention_ingress_epilogue(
    dispatch=qkv_fallback.dispatch_plan(),
    maximum_rows=profile.max_query_rows(),
    maximum_tokens=profile.max_query_tokens(),
    epilogue=tile_epilogue,
    single_row_numerics=AllowDeterministicDotTree,
  ).with_fold_choices(fold_choices) catch {
    _ => raise FusedParallelMismatch(Operation)
  }
  let request = match ingress_policy {
    None => request
    Some(policy) => {
      guard fold_choices.is_empty() else {
        raise FusedParallelInvalid(CompilerPolicy)
      }
      request
      .with_ingress_row_tiles(policy.row_tiles())
      .with_ingress_head_tiles(policy.head_tiles())
      .with_fold_pipeline(
        MatrixFold,
        stages=policy.operand_stages(),
        transfer_tiles=policy.transfer_tiles(),
        fragment_stages=policy.fragment_stages(),
      ) catch {
        _ => raise FusedParallelUnsupported(StaticResources)
      }
    }
  }
  let tile_compilation = @luna_projection_tile_compiler.compile_projection_tile(
    request,
  )
  if !tile_compilation.schedule().fuses_attention_ingress_epilogue() ||
    tile_compilation.schedule().single_row_reduction() != StridedPairwiseDot(32) {
    raise FusedParallelMismatch(Operation)
  }
  // An explicit offline device budget admits larger shared arenas. Above the
  // portable static limit the recipe requires startup's CUDA dynamic opt-in.
  guard tile_compilation.schedule().matrix_pipeline() is Some(pipeline) else {
    raise FusedParallelUnsupported(StaticResources)
  }
  let ingress_policy = match ingress_policy {
    Some(policy) => policy
    None => qwen_default_ingress_policy(tile_compilation)
  }
  let columns = qwen_ingress_column_plan(
    tile_compilation,
    policy=ingress_policy,
  )
  let shared_bytes = ingress_scratch_plan(
    pipeline.row_width(),
    pipeline.output_tiles() * pipeline.tile_width(),
    pipeline.transfer_width(),
    pipeline.stage_count(),
    columns.geometry().3,
  ).allocation_bytes()
  if shared_bytes > maximum_shared_bytes || maximum_shared_bytes > 98304L {
    raise FusedParallelUnsupported(StaticResources)
  }
  guard qwen_ingress_policies(tile_compilation, maximum_shared_bytes~).contains(
    ingress_policy,
  ) else {
    raise FusedParallelUnsupported(StaticResources)
  }
  let physical_program = qwen_refine_physical(tile_compilation, ingress_policy)
  let tile_rows = pipeline.row_width()
  if tile_rows <= 0 {
    raise FusedParallelUnsupported(StaticResources)
  }
  let tiled_grid_x = (profile.max_query_tokens() + tile_rows - 1) / tile_rows
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=tiled_grid_x,
    grid_y=physical_program.packing.geometry().0,
    grid_z=1,
    block_x=FUSED_PARALLEL_THREADS,
    block_y=1,
    block_z=1,
    shared_memory_bytes=if shared_bytes > 49152L {
      shared_bytes.to_int()
    } else {
      0
    },
  ) catch {
    _ => raise FusedParallelUnsupported(StaticResources)
  }
  let function_symbol = "lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_strict_v1"
  let dispatch_canary = 269U
  let source = qwen_fused_source(
    function_symbol,
    semantic,
    layout,
    profile,
    tile_compilation,
    physical_program,
    Some(dispatch_canary),
  )
  if source.length() > limits.maximum_source_bytes {
    raise FusedParallelLimitExceeded(Source)
  }
  let source_digest = fused_parallel_digest(source)
  let recipe = qwen_fused_recipe(
    plan, semantic, layout, qkv_fallback, qk_norm_fallback, rope_fallback, kv_write_fallback,
    target, profile, compiler, function_symbol, dispatch_canary, tiled_grid_x, tile_compilation,
    physical_program, source_digest,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise FusedParallelLimitExceeded(Recipe)
  }
  let recipe_digest = fused_parallel_digest(recipe)
  {
    model_identity: plan.identity(),
    semantic,
    tile_compilation,
    ingress_policy,
    physical_program,
    qkv_fallback,
    qk_norm_fallback,
    rope_fallback,
    kv_write_fallback,
    target,
    profile,
    compiler,
    layout,
    dimensions,
    numerical_policy: QkvQkNormRopeKvWriteDeterministicDotTreeToleranceV1,
    function_symbol,
    dispatch_canary,
    source,
    source_digest,
    recipe,
    recipe_digest,
    candidate_digest: fused_candidate_digest(
      "qkv-qknorm-positioned-rope-paged-kvwrite", source_digest, recipe_digest,
    ),
  }
}

///|
pub fn lower_fused_qwen_qkv_qknorm_rope_paged_kv_write_production_candidate(
  qualification : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
  limits : FusedParallelAotLimits,
  step_rotary_cache~ : Bool = false,
) -> FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate raise FusedParallelAotError {
  let function_symbol = if step_rotary_cache {
    "lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_cached_rotary_v3"
  } else {
    "lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_v2"
  }
  let source = qwen_fused_source(
    function_symbol,
    qualification.semantic,
    qualification.layout,
    qualification.profile,
    qualification.tile_compilation,
    qualification.physical_program,
    None,
    step_rotary_cache~,
  )
  if source.length() > limits.maximum_source_bytes {
    raise FusedParallelLimitExceeded(Source)
  }
  let source_digest = fused_parallel_digest(source)
  let recipe = qwen_fused_production_recipe(
    source_digest,
    qualification,
    function_symbol,
    step_rotary_cache~,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise FusedParallelLimitExceeded(Recipe)
  }
  let recipe_digest = fused_parallel_digest(recipe)
  {
    qualification,
    step_rotary_cache,
    function_symbol,
    source,
    source_digest,
    recipe,
    recipe_digest,
    candidate_digest: fused_candidate_digest(
      "qkv-qknorm-positioned-rope-paged-kvwrite-production", source_digest, recipe_digest,
    ),
  }
}
