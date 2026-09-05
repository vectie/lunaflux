///|
fn qwen_fused_validate_fallbacks(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering,
  qk_norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  kv_write : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
) -> Unit raise FusedParallelAotError {
  let (qkv_index, qk_norm_index, rope_index, attention_index) = semantic.operation_indices()
  let qkv_operation = fused_find_operation(
    plan,
    @plan.OperationId::new(qkv_index) catch {
      _ => raise FusedParallelMismatch(Operation)
    },
  )
  let qk_norm_operation = fused_find_operation(
    plan,
    @plan.OperationId::new(qk_norm_index) catch {
      _ => raise FusedParallelMismatch(Operation)
    },
  )
  let rope_operation = fused_find_operation(
    plan,
    @plan.OperationId::new(rope_index) catch {
      _ => raise FusedParallelMismatch(Operation)
    },
  )
  let attention = fused_find_operation(
    plan,
    @plan.OperationId::new(attention_index) catch {
      _ => raise FusedParallelMismatch(Operation)
    },
  )
  if qkv.operation_id() != qkv_operation.id() ||
    qk_norm.operation_id() != qk_norm_operation.id() ||
    rope.operation_id() != rope_operation.id() ||
    kv_write.operation_id() != attention.id() ||
    !fused_exact_roles(qkv.operands(), fused_operation_roles(qkv_operation)) ||
    !fused_exact_roles(
      qk_norm.operands(),
      fused_operation_roles(qk_norm_operation),
    ) ||
    !fused_exact_roles(rope.operands(), fused_operation_roles(rope_operation)) {
    raise FusedParallelMismatch(OperandOrder)
  }
  let kv_roles : Array[@launch_contract.AotOperandRole] = []
  for runtime in attention.runtime_inputs() {
    kv_roles.push(RuntimeInput(runtime))
  }
  kv_roles.push(ActivationInput(semantic.positioned_qkv()))
  kv_roles.push(PersistentKvComponent(semantic.kv_layer(), Key))
  kv_roles.push(PersistentKvComponent(semantic.kv_layer(), Value))
  if !fused_exact_roles(kv_write.operands(), kv_roles) {
    raise FusedParallelMismatch(OperandOrder)
  }
  for operand in kv_write.operands() {
    match operand.role() {
      PersistentKvComponent(_, _) =>
        if operand.byte_count() != layout.component_stride_bytes() {
          raise FusedParallelMismatch(OperandSize)
        }
      _ => ()
    }
  }
}

///|
fn qwen_fused_recipe(
  plan : @plan.ModelPlan,
  semantic : @luna_fusion_plan.QkvQkNormPositionedRopePagedKvWritePlan,
  layout : @device_layout.DeviceKvLayout,
  qkv : @luna_cuda_projection_aot.ProjectionCudaLowering,
  qk_norm : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  rope : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  kv_write : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  target : @catalog.DeviceTarget,
  profile : @launch_contract.PagedKernelProfile,
  compiler : @luna_specialization.LunaCompilerPolicy,
  symbol : String,
  dispatch_canary : UInt,
  grid_x : Int,
  tile_compilation : @luna_projection_tile_compiler.ScheduledProjectionTileCompilation,
  source_digest : FusedParallelDigest,
) -> Bytes {
  let (qkv_index, qk_norm_index, rope_index, attention_index) = semantic.operation_indices()
  let (input_width, output_width) = semantic.projection_shape()
  let (query_heads, kv_heads, head_dimension) = semantic.attention_shape()
  @utf8.encode(
    fused_common_recipe(
      "qkv-qknorm-positioned-rope-paged-kvwrite-head-tiled",
      target,
      profile,
      compiler,
      source_digest,
      symbol,
      dispatch_canary,
      QkvQkNormRopeKvWriteDeterministicDotTreeToleranceV1,
      grid_x,
      query_heads + 2 * kv_heads,
      0,
    ) +
    "semantic_family=qkv-qknorm-positioned-rope-paged-kvwrite\n" +
    "tile_program_sha256=" +
    tile_compilation.program().digest().as_hex() +
    "\ntile_optimization_sha256=" +
    tile_compilation.optimization().digest().as_hex() +
    "\ntile_schedule_sha256=" +
    tile_compilation.schedule().digest().as_hex() +
    "\ntile_compilation_sha256=" +
    tile_compilation.digest().as_hex() +
    "\n" +
    "model_content_sha256=" +
    plan.identity().content().as_hex() +
    "\nmodel_plan_sha256=" +
    plan.identity().plan().as_hex() +
    "\nqkv_operation_id=" +
    qkv_index.to_string() +
    "\nqk_norm_operation_id=" +
    qk_norm_index.to_string() +
    "\nrope_operation_id=" +
    rope_index.to_string() +
    "\nattention_operation_id=" +
    attention_index.to_string() +
    "\nshape=" +
    input_width.to_string() +
    "," +
    output_width.to_string() +
    "," +
    query_heads.to_string() +
    "," +
    kv_heads.to_string() +
    "," +
    head_dimension.to_string() +
    "\nnorm_epsilon=" +
    semantic.norm_epsilon().to_string() +
    "\nrope_theta=" +
    semantic.rope_theta().to_string() +
    "\ntokens_per_page=" +
    layout.tokens_per_page().to_string() +
    "\ntotal_page_count=" +
    layout.total_page_count().to_string() +
    "\nqkv_fallback_source_sha256=" +
    qkv.source_digest().as_hex() +
    "\nqk_norm_fallback_source_sha256=" +
    qk_norm.source_digest().as_hex() +
    "\nrope_fallback_source_sha256=" +
    rope.source_digest().as_hex() +
    "\nkv_write_fallback_source_sha256=" +
    kv_write.source_digest().as_hex() +
    "\nabi=step_counts,query_positions,query_row_offsets,sequence_lengths," +
    "page_table_offsets,page_table_page_indices,input,q_weight,k_weight," +
    "v_weight,q_norm_weight,k_norm_weight,rotated_qkv,key_cache,value_cache," +
    "dispatch_canary\nreference_fallback=four-distinct-correctness-kernels\n" +
    "aot_only=true\nruntime_jit=false\nmanifest_bindable=false\n",
  )
}

///|
fn qwen_fused_production_recipe(
  source_digest : FusedParallelDigest,
  qualification : FusedQwenQkvQkNormRopeKvWriteCudaCandidate,
  symbol : String,
) -> Bytes {
  let semantic = qualification.semantic
  let (qkv_index, qk_norm_index, rope_index, attention_index) = semantic.operation_indices()
  let (input_width, output_width) = semantic.projection_shape()
  let (query_heads, kv_heads, head_dimension) = semantic.attention_shape()
  let profile = qualification.profile
  @utf8.encode(
    "schema=lunaflux-fused-parallel-cuda-aot-candidate.v2\n" +
    "family=qkv-qknorm-positioned-rope-paged-kvwrite-production-head-tiled\n" +
    "execution_policy=production-fast-path\n" +
    "qualification_candidate_sha256=" +
    qualification.candidate_digest.as_hex() +
    "\nsource_sha256=" +
    source_digest.as_hex() +
    "\nfunction_symbol=" +
    symbol +
    "\ntarget=sm_" +
    qualification.target.compute_major().to_string() +
    qualification.target.compute_minor().to_string() +
    "\nprofile_id=" +
    profile.id().as_int().to_string() +
    "\nmax_query_rows=" +
    profile.max_query_rows().to_string() +
    "\nmax_query_tokens=" +
    profile.max_query_tokens().to_string() +
    "\nmax_page_table_entries=" +
    profile.max_page_table_entries().to_string() +
    "\nblock=128,1,1\ngrid=" +
    qualification.dimensions.grid_x().to_string() +
    "," +
    (query_heads + 2 * kv_heads).to_string() +
    ",1\nshared_memory_bytes=0\ndiagnostic_canary=absent\n" +
    fused_compiler_record(qualification.compiler) +
    "tile_program_sha256=" +
    qualification.tile_compilation.program().digest().as_hex() +
    "\ntile_optimization_sha256=" +
    qualification.tile_compilation.optimization().digest().as_hex() +
    "\ntile_schedule_sha256=" +
    qualification.tile_compilation.schedule().digest().as_hex() +
    "\ntile_compilation_sha256=" +
    qualification.tile_compilation.digest().as_hex() +
    "\nmodel_content_sha256=" +
    qualification.model_identity.content().as_hex() +
    "\nmodel_plan_sha256=" +
    qualification.model_identity.plan().as_hex() +
    "\n" +
    "qkv_operation_id=" +
    qkv_index.to_string() +
    "\nqk_norm_operation_id=" +
    qk_norm_index.to_string() +
    "\nrope_operation_id=" +
    rope_index.to_string() +
    "\nattention_operation_id=" +
    attention_index.to_string() +
    "\nshape=" +
    input_width.to_string() +
    "," +
    output_width.to_string() +
    "," +
    query_heads.to_string() +
    "," +
    kv_heads.to_string() +
    "," +
    head_dimension.to_string() +
    "\nnorm_epsilon=" +
    semantic.norm_epsilon().to_string() +
    "\nrope_theta=" +
    semantic.rope_theta().to_string() +
    "\ntokens_per_page=" +
    qualification.layout.tokens_per_page().to_string() +
    "\ntotal_page_count=" +
    qualification.layout.total_page_count().to_string() +
    "\nqkv_fallback_source_sha256=" +
    qualification.qkv_fallback.source_digest().as_hex() +
    "\nqk_norm_fallback_source_sha256=" +
    qualification.qk_norm_fallback.source_digest().as_hex() +
    "\nrope_fallback_source_sha256=" +
    qualification.rope_fallback.source_digest().as_hex() +
    "\nkv_write_fallback_source_sha256=" +
    qualification.kv_write_fallback.source_digest().as_hex() +
    "\nabi=step_counts,query_positions,query_row_offsets,sequence_lengths," +
    "page_table_offsets,page_table_page_indices,input,q_weight,k_weight," +
    "v_weight,q_norm_weight,k_norm_weight,rotated_qkv,key_cache,value_cache\n" +
    "reference_fallback=four-distinct-correctness-kernels\n" +
    "aot_only=true\nruntime_jit=false\n" +
    "release_binding=external-approval-and-production-physical-gate-required\n" +
    "manifest_bindable=false\n",
  )
}

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
  let (query_heads, kv_heads, head_dimension) = semantic.attention_shape()
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
  let tile_compilation = @luna_projection_tile_compiler.compile_projection_tile(
    @luna_projection_tile_compiler.ProjectionTileCompilerRequest::with_attention_ingress_epilogue(
      dispatch=qkv_fallback.dispatch_plan(),
      maximum_rows=profile.max_query_rows(),
      maximum_tokens=profile.max_query_tokens(),
      epilogue=tile_epilogue,
      single_row_numerics=AllowDeterministicDotTree,
    ) catch {
      _ => raise FusedParallelMismatch(Operation)
    },
  )
  if !tile_compilation.schedule().fuses_attention_ingress_epilogue() ||
    tile_compilation.schedule().single_row_reduction() != StridedPairwiseDot(32) {
    raise FusedParallelMismatch(Operation)
  }
  let tile_rows = qkv_fallback.dispatch_plan().prefill().tile().dimensions().0
  if tile_rows <= 0 {
    raise FusedParallelUnsupported(StaticResources)
  }
  let tiled_grid_x = (profile.max_query_tokens() + tile_rows - 1) / tile_rows
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=tiled_grid_x,
    grid_y=query_heads + 2 * kv_heads,
    grid_z=1,
    block_x=FUSED_PARALLEL_THREADS,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
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
    Some(dispatch_canary),
  )
  if source.length() > limits.maximum_source_bytes {
    raise FusedParallelLimitExceeded(Source)
  }
  let source_digest = fused_parallel_digest(source)
  let recipe = qwen_fused_recipe(
    plan, semantic, layout, qkv_fallback, qk_norm_fallback, rope_fallback, kv_write_fallback,
    target, profile, compiler, function_symbol, dispatch_canary, tiled_grid_x, tile_compilation,
    source_digest,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise FusedParallelLimitExceeded(Recipe)
  }
  let recipe_digest = fused_parallel_digest(recipe)
  {
    model_identity: plan.identity(),
    semantic,
    tile_compilation,
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
) -> FusedQwenQkvQkNormRopeKvWriteProductionCudaCandidate raise FusedParallelAotError {
  let function_symbol = "lunaflux_fused_qwen_qkv_qknorm_rope_kvwrite_bf16_head128_production_v2"
  let source = qwen_fused_source(
    function_symbol,
    qualification.semantic,
    qualification.layout,
    qualification.profile,
    qualification.tile_compilation,
    None,
  )
  if source.length() > limits.maximum_source_bytes {
    raise FusedParallelLimitExceeded(Source)
  }
  let source_digest = fused_parallel_digest(source)
  let recipe = qwen_fused_production_recipe(
    source_digest, qualification, function_symbol,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise FusedParallelLimitExceeded(Recipe)
  }
  let recipe_digest = fused_parallel_digest(recipe)
  {
    qualification,
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
