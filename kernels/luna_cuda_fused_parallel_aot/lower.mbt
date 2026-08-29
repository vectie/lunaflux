///|
fn fused_candidate_digest(
  family : String,
  source : FusedParallelDigest,
  recipe : FusedParallelDigest,
) -> FusedParallelDigest {
  fused_parallel_digest(
    @utf8.encode(
      "lunaflux.fused-parallel-candidate.v1\n" +
      family +
      "\n" +
      source.as_hex() +
      "\n" +
      recipe.as_hex() +
      "\n",
    ),
  )
}

///|
pub fn lower_fused_qkv_rope_paged_kv_write_candidate(
  plan : @plan.ModelPlan,
  layout : @device_layout.DeviceKvLayout,
  qkv_fallback : @luna_cuda_projection_aot.ProjectionCudaLowering,
  rope_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  kv_write_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  limits : FusedParallelAotLimits,
) -> FusedQkvRopeKvWriteCudaCandidate raise FusedParallelAotError {
  let profile = qkv_fallback.profile()
  fused_validate_common(profile, target, compiler)
  if qkv_fallback.family() != QkvProjection ||
    rope_fallback.family() != PositionedRotary ||
    kv_write_fallback.family() != KvCacheWrite ||
    qkv_fallback.target() != target ||
    rope_fallback.target() != target ||
    kv_write_fallback.target() != target ||
    qkv_fallback.compiler_policy() != compiler ||
    rope_fallback.profile() != profile ||
    kv_write_fallback.profile() != profile {
    raise FusedParallelMismatch(ReferenceFallback)
  }
  let parameters = fused_qkv_parameters(
    plan,
    layout,
    qkv_fallback.operation_id(),
    rope_fallback.operation_id(),
    kv_write_fallback.operation_id(),
  )
  fused_validate_qkv_fallback_abis(
    plan, layout, parameters, qkv_fallback, rope_fallback, kv_write_fallback,
  )
  fused_validate_operand_alignments(qkv_fallback.operands(), 16L)
  fused_validate_operand_alignments(rope_fallback.operands(), 16L)
  fused_validate_operand_alignments(
    kv_write_fallback.operands(),
    layout.segment_alignment_bytes(),
  )
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=profile.max_query_tokens(),
    grid_y=1,
    grid_z=1,
    block_x=FUSED_PARALLEL_THREADS,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise FusedParallelUnsupported(StaticResources)
  }
  let function_symbol = "lunaflux_fused_qkv_positioned_rope_paged_kvwrite_bf16_block128_strict_v1"
  let dispatch_canary : UInt = 257U
  if profile.max_query_tokens().to_int64() * dispatch_canary.to_int64() >
    4294967295L {
    raise FusedParallelUnsupported(Profile)
  }
  let numerical_policy = QkvRopeKvWriteOrderedF32TranscendentalToleranceV1
  let source = fused_qkv_rope_kv_source(
    function_symbol,
    parameters,
    profile,
    Some(dispatch_canary),
  )
  if source.length() > limits.maximum_source_bytes {
    raise FusedParallelLimitExceeded(Source)
  }
  let source_digest = fused_parallel_digest(source)
  let recipe = fused_qkv_recipe(
    source_digest, parameters, qkv_fallback, rope_fallback, kv_write_fallback, target,
    profile, compiler, function_symbol, dispatch_canary, numerical_policy,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise FusedParallelLimitExceeded(Recipe)
  }
  let recipe_digest = fused_parallel_digest(recipe)
  {
    qkv_fallback,
    rope_fallback,
    kv_write_fallback,
    target,
    profile,
    compiler,
    parameters,
    dimensions,
    numerical_policy,
    function_symbol,
    dispatch_canary,
    source,
    source_digest,
    recipe,
    recipe_digest,
    candidate_digest: fused_candidate_digest(
      "qkv-positioned-rope-paged-kvwrite", source_digest, recipe_digest,
    ),
  }
}

///|
/// Derives the canary-free production source from an exact qualification
/// candidate. Physical compilation and deployment approval remain external.
pub fn lower_fused_qkv_rope_paged_kv_write_production_candidate(
  qualification : FusedQkvRopeKvWriteCudaCandidate,
  limits : FusedParallelAotLimits,
) -> FusedQkvRopeKvWriteProductionCudaCandidate raise FusedParallelAotError {
  let function_symbol = "lunaflux_fused_qkv_positioned_rope_paged_kvwrite_bf16_block128_production_v2"
  let source = fused_qkv_rope_kv_production_source(
    function_symbol,
    qualification.parameters,
    qualification.profile,
  )
  if source.length() > limits.maximum_source_bytes {
    raise FusedParallelLimitExceeded(Source)
  }
  let source_digest = fused_parallel_digest(source)
  let recipe = fused_qkv_production_recipe(
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
      "qkv-positioned-rope-paged-kvwrite-production", source_digest, recipe_digest,
    ),
  }
}

///|
pub fn lower_fused_residual_rmsnorm_candidate(
  plan : @plan.ModelPlan,
  residual_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  norm_fallback : @luna_cuda_pointwise_aot.PointwiseCudaLowering,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  limits : FusedParallelAotLimits,
) -> FusedResidualRmsNormCudaCandidate raise FusedParallelAotError {
  let profile = residual_fallback.profile()
  fused_validate_common(profile, target, compiler)
  if residual_fallback.family() != ResidualAdd ||
    norm_fallback.family() != RmsNorm ||
    residual_fallback.target() != target ||
    norm_fallback.target() != target ||
    residual_fallback.profile() != profile ||
    norm_fallback.profile() != profile {
    raise FusedParallelMismatch(ReferenceFallback)
  }
  let parameters = fused_residual_norm_parameters(
    plan,
    residual_fallback.operation_id(),
    norm_fallback.operation_id(),
  )
  fused_validate_residual_norm_fallback_abis(
    plan, parameters, residual_fallback, norm_fallback,
  )
  fused_validate_operand_alignments(residual_fallback.operands(), 16L)
  fused_validate_operand_alignments(norm_fallback.operands(), 16L)
  let shared_memory_bytes = fused_checked_mul(
    FUSED_PARALLEL_THREADS.to_int64(),
    4L,
  ).to_int()
  if shared_memory_bytes > 49152 {
    raise FusedParallelUnsupported(StaticResources)
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=profile.max_query_tokens(),
    grid_y=1,
    grid_z=1,
    block_x=FUSED_PARALLEL_THREADS,
    block_y=1,
    block_z=1,
    shared_memory_bytes~,
  ) catch {
    _ => raise FusedParallelUnsupported(StaticResources)
  }
  let function_symbol = "lunaflux_fused_residual_rmsnorm_bf16_block128_tree_strict_v1"
  let dispatch_canary : UInt = 263U
  if profile.max_query_tokens().to_int64() * dispatch_canary.to_int64() >
    4294967295L {
    raise FusedParallelUnsupported(Profile)
  }
  let numerical_policy = ResidualRmsNormBlock128F32ReductionToleranceV1
  let source = fused_residual_rmsnorm_source(
    function_symbol, parameters, profile, dispatch_canary,
  )
  if source.length() > limits.maximum_source_bytes {
    raise FusedParallelLimitExceeded(Source)
  }
  let source_digest = fused_parallel_digest(source)
  let recipe = fused_residual_norm_recipe(
    source_digest, parameters, residual_fallback, norm_fallback, target, profile,
    compiler, function_symbol, dispatch_canary, numerical_policy,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise FusedParallelLimitExceeded(Recipe)
  }
  let recipe_digest = fused_parallel_digest(recipe)
  {
    residual_fallback,
    norm_fallback,
    target,
    profile,
    compiler,
    parameters,
    dimensions,
    numerical_policy,
    function_symbol,
    dispatch_canary,
    source,
    source_digest,
    recipe,
    recipe_digest,
    candidate_digest: fused_candidate_digest(
      "residual-rmsnorm", source_digest, recipe_digest,
    ),
  }
}

///|
/// Derives a canary-free production companion from one exact qualification
/// candidate. The result has no runtime authority; compilation, production
/// physical validation, and external manifest approval remain separate gates.
pub fn lower_fused_residual_rmsnorm_production_candidate(
  qualification : FusedResidualRmsNormCudaCandidate,
  limits : FusedParallelAotLimits,
) -> FusedResidualRmsNormProductionCudaCandidate raise FusedParallelAotError {
  let function_symbol = "lunaflux_fused_residual_rmsnorm_bf16_block128_tree_production_v2"
  let source = fused_residual_rmsnorm_production_source(
    function_symbol,
    qualification.parameters,
    qualification.profile,
  )
  if source.length() > limits.maximum_source_bytes {
    raise FusedParallelLimitExceeded(Source)
  }
  let source_digest = fused_parallel_digest(source)
  let recipe = fused_residual_norm_production_recipe(
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
      "residual-rmsnorm-production", source_digest, recipe_digest,
    ),
  }
}
