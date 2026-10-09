///|
fn unique_full_rotary_geometry(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> (Int, Int, Int)? {
  let mut query = None
  let mut rope = None
  for item in requirements.requirements() {
    match item.operation() {
      DsaQueryBProjection(_, heads, dimension) => {
        if query is Some(_) {
          return None
        }
        query = Some((heads, dimension))
      }
      DsaKeyValueAProjection(_, _, suffix) => {
        if rope is Some(_) {
          return None
        }
        rope = Some(suffix)
      }
      _ => ()
    }
  }
  guard query is Some((heads, dimension)) && rope is Some(rope_dimension) else {
    return None
  }
  Some((heads, dimension, rope_dimension))
}

///|
fn full_rotary_recipe(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  maximum_positions : Int,
  source_digest : GlmHybridCudaSourceDigest,
) -> Bytes {
  @utf8.encode(
    "lunaflux.glm-hybrid-interleaved-rope-cuda-aot.v1" +
    "|stage=query-rope-and-shared-key-rope-only" +
    "|query-input=token-query-head-nope-then-adjacent-pair-rope" +
    "|query-output=token-query-head-nope-bit-exact-then-rotated-pair-component-halves" +
    "|key-input=token-one-shared-adjacent-pair-rope" +
    "|key-output=token-one-shared-rotated-pair-component-halves" +
    "|frequency=float32-default-theta-8000000|trig-storage=bf16" +
    "|pair-arithmetic=bf16-strict-declared-order|amplitude=1" +
    "|excluded=kv-b-key-nope-value-expansion-cache-and-attention" +
    "|downstream=query-output-is-dsa-attention-query-compatible" +
    "|downstream=key-output-awaits-kv-b-key-nope-join-and-head-expansion" +
    "|qualification=source-only-non-bindable" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|geometry=64:1:256:64:" +
    maximum_tokens.to_string() +
    ":" +
    maximum_positions.to_string() +
    "|launch=" +
    maximum_tokens.to_string() +
    ":65:1:256:1:1:0" +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
pub fn lower_dsa_interleaved_rotary_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridRotaryCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (GlmRotaryPosition(Some(theta), true), GlmRotaryPositionAot) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard unique_full_rotary_geometry(requirements)
    is Some((query_heads, query_dimension, rope_dimension)) else {
    raise Invalid(Requirement)
  }
  if requirements.geometry().hidden_size() != 6144 ||
    query_heads != 64 ||
    query_dimension != 256 ||
    rope_dimension != 64 ||
    theta != 8000000.0 {
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
  let query_elements = checked_mul(
    maximum_tokens.to_int64(),
    checked_mul(query_heads.to_int64(), query_dimension.to_int64()),
  )
  let key_elements = checked_mul(
    maximum_tokens.to_int64(),
    rope_dimension.to_int64(),
  )
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
    grid_y=query_heads + 1,
    grid_z=1,
    block_x=256,
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
      DsaQueryHeadsOutput,
      checked_mul(query_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaKeyRopeSuffixOutput,
      checked_mul(key_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaAttentionQueriesInput,
      checked_mul(query_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      DsaAttentionKeyRopeInput,
      checked_mul(key_elements, 2L),
      16L,
    ),
  ]
  let rendered = @decoder_interleaved_rope_cuda_source.interleaved_pair_rope_with_query_prefix_copy_bf16_source(
    query_head_count=query_heads,
    key_head_count=1,
    query_head_dimension=query_dimension,
    rope_dimension~,
    maximum_rows=maximum_tokens,
    maximum_positions=requirements.geometry().context_length(),
    theta~,
    function_symbol="lunaflux_glm_hybrid_dsa_interleaved_rope_bf16_v1",
  ) catch {
    error => raise InterleavedRopeSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = full_rotary_recipe(
    requirements,
    requirement_ordinal,
    maximum_tokens,
    requirements.geometry().context_length(),
    source_digest,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: DsaInterleavedQuerySharedKeyRopeBf16,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    query_head_count: query_heads,
    query_head_dimension: query_dimension,
    rope_dimension,
    maximum_tokens,
    maximum_positions: requirements.geometry().context_length(),
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
