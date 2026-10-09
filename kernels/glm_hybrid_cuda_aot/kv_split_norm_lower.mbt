///|
fn unique_kv_a_split_geometry(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
) -> (Int, Int)? {
  let mut found = None
  for item in requirements.requirements() {
    if item.operation() is DsaKeyValueAProjection(_, prefix, suffix) {
      if found is Some(_) {
        return None
      }
      found = Some((prefix, suffix))
    }
  }
  found
}

///|
fn kv_split_norm_recipe(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  ordinal : Int,
  prefix_width : Int,
  suffix_width : Int,
  maximum_tokens : Int,
  source_digest : GlmHybridCudaSourceDigest,
) -> Bytes {
  @utf8.encode(
    "lunaflux.glm-hybrid-kv-split-norm-cuda-aot.v1" +
    "|numerical=ordered-f32-prefix-reduction-bf16-round-and-bit-exact-suffix-copy-v1" +
    "|stage=kv-a-split-and-prefix-rmsnorm" +
    "|input=token-compressed-prefix-then-rope-suffix" +
    "|normalized-output=token-compressed-kv|downstream=kv-b-normalized-input-compatible" +
    "|suffix-output=token-rope-key-bit-exact|suffix-operand=present-iff-positive-width" +
    "|qualification=source-only-non-bindable" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|geometry=" +
    prefix_width.to_string() +
    ":" +
    suffix_width.to_string() +
    ":" +
    maximum_tokens.to_string() +
    "|launch=" +
    maximum_tokens.to_string() +
    ":1:1:256:1:1:1024" +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
pub fn lower_dsa_key_value_split_norm_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridKvSplitNormCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (QueryKeyRmsNorm(epsilon), Advanced(QueryKeyRmsNormAot)) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard unique_kv_a_split_geometry(requirements)
    is Some((prefix_width, suffix_width)) else {
    raise Invalid(Requirement)
  }
  if prefix_width != 512 ||
    !((requirements.geometry().hidden_size() == 6144 && suffix_width == 64) ||
    (requirements.geometry().hidden_size() == 4096 && suffix_width == 0)) ||
    epsilon != 0.00001 {
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
  if compiler.reassociate() {
    raise Invalid(Compiler)
  }
  let input_elements = checked_mul(
    maximum_tokens.to_int64(),
    (prefix_width + suffix_width).to_int64(),
  )
  let normalized_elements = checked_mul(
    maximum_tokens.to_int64(),
    prefix_width.to_int64(),
  )
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=1024,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(
      DsaKeyValueCompressedRopeInput,
      checked_mul(input_elements, 2L),
      16L,
    ),
    dense_projection_operand(
      ModelWeights,
      checked_mul(prefix_width.to_int64(), 2L),
      16L,
    ),
    dense_projection_operand(
      DsaKeyValueNormalizedOutput,
      checked_mul(normalized_elements, 2L),
      16L,
    ),
  ]
  if suffix_width > 0 {
    operands.push(
      dense_projection_operand(
        DsaKeyRopeSuffixOutput,
        checked_mul(
          checked_mul(maximum_tokens.to_int64(), suffix_width.to_int64()),
          2L,
        ),
        16L,
      ),
    )
  }
  let rendered = @prefix_rms_norm_cuda_source.prefix_rms_norm_bf16_with_optional_suffix_copy_source(
    prefix_width~,
    suffix_width~,
    maximum_rows=maximum_tokens,
    epsilon~,
    function_symbol="lunaflux_glm_hybrid_dsa_kv_a_split_rms_norm_bf16_v1",
  ) catch {
    error => raise PrefixNormSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = kv_split_norm_recipe(
    requirements, requirement_ordinal, prefix_width, suffix_width, maximum_tokens,
    source_digest,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: DsaKeyValuePrefixRmsNormBf16,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    prefix_width,
    suffix_width,
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
