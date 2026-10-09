///|
fn unique_sparse_kv_geometry(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> (Int, Int)? {
  let mut found = None
  for item in requirements.requirements() {
    if item.operation() is KeyValueLowRankProjection(_, heads, _, value) {
      if found is Some(_) {
        return None
      }
      found = Some((heads, value))
    }
  }
  found
}

///|
fn sparse_attention_recipe(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  ordinal : Int,
  query_heads : Int,
  kv_heads : Int,
  qk_dimension : Int,
  value_dimension : Int,
  top_k : Int,
  selected_width : Int,
  maximum_tokens : Int,
  source_digest : GlmHybridCudaSourceDigest,
) -> Bytes {
  @utf8.encode(
    "lunaflux.glm-hybrid-sparse-attention-cuda-aot.v1" +
    "|qualification=serial-correctness-only-not-throughput-qualified" +
    "|scope=projected-qkv-plus-selected-indices-no-projection-no-index-production" +
    "|batch=single-sequence-counts-2-equals-1" +
    "|layout=q-token-query-head-qk:k-token-kv-head-qk:v-token-kv-head-value" +
    "|gqa=query-head-div-kv-head-contiguous-groups" +
    "|selected-width-distinct-from-semantic-top-k" +
    "|sentinel=-1-skipped|invalid-or-duplicate=whole-query-row-zero" +
    "|causal=key-index-lte-query-position|softmax=float32-stable-then-bf16" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|geometry=" +
    query_heads.to_string() +
    ":" +
    kv_heads.to_string() +
    ":" +
    qk_dimension.to_string() +
    ":" +
    value_dimension.to_string() +
    ":" +
    top_k.to_string() +
    ":" +
    selected_width.to_string() +
    ":" +
    maximum_tokens.to_string() +
    "|launch=1:1:1:1:1:1:0" +
    "|operands=counts@20/4,positions@" +
    (maximum_tokens * 4).to_string() +
    "/4,key-count@4/4,key-valid@" +
    (maximum_tokens * 4).to_string() +
    "/4,queries@bf16,keys@bf16,values@bf16,selected@i32,output@bf16" +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
pub fn lower_dsa_sparse_attention_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridSparseAttentionCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      DsaSparseAttention(query_heads, top_k, qk_dimension, value_dimension),
      DsaSparseAttentionAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard unique_sparse_kv_geometry(requirements)
    is Some((kv_heads, kv_value_dimension)) else {
    raise Invalid(Requirement)
  }
  let (pool_size, append_tail) = dsa_index_policy(requirements)
  if query_heads != 64 ||
    kv_heads != 64 ||
    qk_dimension != 256 ||
    value_dimension != 256 ||
    kv_value_dimension != value_dimension ||
    top_k != 2048 ||
    !((pool_size == 1 && !append_tail) || (pool_size == 4 && append_tail)) {
    raise Invalid(Shape)
  }
  if maximum_tokens <= 0 ||
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
  let selected_width = top_k + (if append_tail { pool_size - 1 } else { 0 })
  let query_rows = checked_mul(
    maximum_tokens.to_int64(),
    query_heads.to_int64(),
  )
  let key_rows = checked_mul(maximum_tokens.to_int64(), kv_heads.to_int64())
  let query_elements = checked_mul(query_rows, qk_dimension.to_int64())
  let key_elements = checked_mul(key_rows, qk_dimension.to_int64())
  let value_elements = checked_mul(key_rows, value_dimension.to_int64())
  let selected_elements = checked_mul(
    maximum_tokens.to_int64(),
    selected_width.to_int64(),
  )
  let output_elements = checked_mul(query_rows, value_dimension.to_int64())
  if output_elements > 2147483647L {
    raise SizeOverflow
  }
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
    dense_projection_operand(DsaKeyCountInput, 4L, 4L),
    dense_projection_operand(
      DsaKeyValidityInput,
      checked_mul(maximum_tokens.to_int64(), 4L),
      4L,
    ),
    dense_projection_operand(
      DsaAttentionQueriesInput,
      checked_mul(query_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaAttentionKeysInput,
      checked_mul(key_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaAttentionValuesInput,
      checked_mul(value_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaSelectedIndicesInput,
      checked_mul(selected_elements, 4L),
      4L,
    ),
    dense_projection_operand(
      DsaAttentionOutput,
      checked_mul(output_elements, 2L),
      16L,
    ),
  ]
  let rendered = @glm_hybrid_sparse_attention_cuda_source.selected_index_sparse_attention_bf16_source(
    query_head_count=query_heads,
    key_value_head_count=kv_heads,
    query_key_dimension=qk_dimension,
    value_dimension~,
    selected_width~,
    maximum_tokens~,
    function_symbol="lunaflux_glm_hybrid_dsa_sparse_attention_bf16_v1",
  ) catch {
    error => raise SparseAttentionSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = sparse_attention_recipe(
    requirements, requirement_ordinal, query_heads, kv_heads, qk_dimension, value_dimension,
    top_k, selected_width, maximum_tokens, source_digest,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: SelectedIndexSparseAttentionBf16,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    query_head_count: query_heads,
    key_value_head_count: kv_heads,
    query_key_dimension: qk_dimension,
    value_dimension,
    top_k,
    selected_width,
    maximum_tokens,
    target,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol: rendered.function_symbol(),
    source,
    source_digest,
    recipe,
    recipe_digest: GlmHybridCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}
