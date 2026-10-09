///|
fn dsa_index_policy(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> (Int, Bool) raise GlmHybridCudaAotError {
  let mut policy : (Int, Bool)? = None
  for requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexPooling(pool_size, true, tail) {
      if policy is Some(_) {
        raise Invalid(Requirement)
      }
      policy = Some((pool_size, tail))
    }
  }
  match policy {
    Some(value) => value
    None => (1, false)
  }
}

///|
fn exact_official_dsa_index_geometry(
  head_count : Int,
  head_dimension : Int,
  top_k : Int,
  rope_interleave : Bool,
) -> Unit raise GlmHybridCudaAotError {
  if head_count != 32 ||
    head_dimension != 128 ||
    top_k != 2048 ||
    !rope_interleave {
    raise Invalid(Shape)
  }
}

///|
fn dsa_index_candidate(
  family : GlmHybridDsaIndexAotFamily,
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirement,
  requirement_ordinal : Int,
  head_count : Int,
  head_dimension : Int,
  top_k : Int,
  pool_size : Int,
  maximum_tokens : Int,
  candidate_capacity : Int,
  output_width : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : Array[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand],
  rendered : @glm_hybrid_dsa_index_cuda_source.GlmHybridDsaIndexCudaSource,
  limits : GlmHybridCudaAotLimits,
) -> GlmHybridDsaIndexCudaCandidate raise GlmHybridCudaAotError {
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = dsa_index_recipe(
    family,
    rendered.numerical_contract(),
    requirements,
    requirement_ordinal,
    head_count,
    head_dimension,
    top_k,
    pool_size,
    maximum_tokens,
    candidate_capacity,
    output_width,
    target,
    compiler,
    dimensions,
    operands,
    source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    head_count,
    head_dimension,
    top_k,
    pool_size,
    maximum_tokens,
    candidate_capacity,
    output_width,
    target,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol: rendered.function_symbol(),
    source,
    source_digest,
    recipe: encoded_recipe,
    recipe_digest: GlmHybridCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(encoded_recipe)),
    ),
  }
}

///|
/// Lowers projected GLM DSA scoring, deterministic top-k, raw-pool expansion,
/// and (for Flash) incomplete-tail selection. Projection and pooling are not
/// part of this candidate.
pub fn lower_dsa_index_selection_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridDsaIndexCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      DsaIndexCompute(head_count, head_dimension, top_k, rope_interleave),
      DsaIndexComputeAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  exact_official_dsa_index_geometry(
    head_count, head_dimension, top_k, rope_interleave,
  )
  let (pool_size, append_tail) = dsa_index_policy(requirements)
  if !((pool_size == 1 && !append_tail) || (pool_size == 4 && append_tail)) ||
    top_k % pool_size != 0 {
    raise Invalid(Shape)
  }
  if maximum_tokens < pool_size ||
    maximum_tokens > limits.maximum_tokens ||
    maximum_tokens > requirements.geometry().context_length() {
    raise Invalid(Limit)
  }
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  let candidate_capacity = maximum_tokens / pool_size
  let output_width = top_k + (if append_tail { pool_size - 1 } else { 0 })
  let query_heads = checked_mul(
    maximum_tokens.to_int64(),
    head_count.to_int64(),
  )
  let query_elements = checked_mul(query_heads, head_dimension.to_int64())
  let candidate_elements = checked_mul(
    candidate_capacity.to_int64(),
    head_dimension.to_int64(),
  )
  let raw_index_elements = checked_mul(
    candidate_capacity.to_int64(),
    pool_size.to_int64(),
  )
  let output_elements = checked_mul(
    maximum_tokens.to_int64(),
    output_width.to_int64(),
  )
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=1,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(
      QueryPositionsInput,
      checked_mul(maximum_tokens.to_int64(), 4L),
      4L,
    ),
    dense_projection_operand(
      DsaIndexQueriesInput,
      checked_mul(query_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaIndexHeadWeightsInput,
      checked_mul(query_heads, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaCandidateKeysInput,
      checked_mul(candidate_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaCandidateRawIndicesInput,
      checked_mul(raw_index_elements, 4L),
      4L,
    ),
    dense_projection_operand(
      DsaCandidateValidityInput,
      checked_mul(candidate_capacity.to_int64(), 4L),
      4L,
    ),
    dense_projection_operand(DsaCandidateCountInput, 4L, 4L),
    dense_projection_operand(
      DsaKeyValidityInput,
      checked_mul(maximum_tokens.to_int64(), 4L),
      4L,
    ),
    dense_projection_operand(
      DsaSelectedIndicesOutput,
      checked_mul(output_elements, 4L),
      4L,
    ),
  ]
  let rendered = @glm_hybrid_dsa_index_cuda_source.projected_index_score_stable_topk_bf16_source(
    head_count~,
    head_dimension~,
    top_k~,
    pool_size~,
    append_incomplete_tail=append_tail,
    maximum_tokens~,
    function_symbol="lunaflux_glm_hybrid_dsa_index_select_bf16_v1",
  ) catch {
    error => raise DsaIndexSource(error)
  }
  dsa_index_candidate(
    if append_tail {
      PooledScoreStableTopKWithTailBf16
    } else {
      RawTokenScoreStableTopKBf16
    },
    requirements,
    requirement,
    requirement_ordinal,
    head_count,
    head_dimension,
    top_k,
    pool_size,
    maximum_tokens,
    candidate_capacity,
    output_width,
    target,
    compiler,
    dimensions,
    operands,
    rendered,
    limits,
  )
}

///|
/// Lowers Full-profile cross-layer reuse as a bit-exact counts-bounded copy.
pub fn lower_dsa_index_reuse_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridDsaIndexCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (DsaIndexReuse, DsaIndexReuseAot) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let (head_dimension, top_k) = dsa_pooling_index_geometry(requirements)
  guard dsa_index_compute_geometry(requirements)
    is Some((head_count, _, _, rope_interleave)) else {
    raise Invalid(Requirement)
  }
  exact_official_dsa_index_geometry(
    head_count, head_dimension, top_k, rope_interleave,
  )
  let (pool_size, append_tail) = dsa_index_policy(requirements)
  if pool_size != 1 || append_tail {
    raise Invalid(Shape)
  }
  if maximum_tokens <= 0 ||
    maximum_tokens > limits.maximum_tokens ||
    maximum_tokens > requirements.geometry().context_length() {
    raise Invalid(Limit)
  }
  let output_width = top_k
  let output_elements = checked_mul(
    maximum_tokens.to_int64(),
    output_width.to_int64(),
  )
  let grid = (output_elements + 255L) / 256L
  if grid <= 0L || grid > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=grid.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let bytes = checked_mul(output_elements, 4L)
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(DsaSelectedIndicesInput, bytes, 4L),
    dense_projection_operand(DsaSelectedIndicesOutput, bytes, 4L),
  ]
  let rendered = @glm_hybrid_dsa_index_cuda_source.reuse_selected_indices_i32_source(
    output_width~,
    maximum_tokens~,
    function_symbol="lunaflux_glm_hybrid_dsa_index_reuse_i32_v1",
  ) catch {
    error => raise DsaIndexSource(error)
  }
  dsa_index_candidate(
    SelectedIndexReuseI32,
    requirements,
    requirement,
    requirement_ordinal,
    head_count,
    head_dimension,
    top_k,
    pool_size,
    maximum_tokens,
    0,
    output_width,
    target,
    compiler,
    dimensions,
    operands,
    rendered,
    limits,
  )
}

///|
fn dsa_index_compute_geometry(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> (Int, Int, Int, Bool)? {
  let mut result = None
  for requirement in requirements.requirements() {
    if requirement.operation() is DsaIndexCompute(a, b, c, d) {
      if result is Some(_) {
        return None
      }
      result = Some((a, b, c, d))
    }
  }
  result
}
