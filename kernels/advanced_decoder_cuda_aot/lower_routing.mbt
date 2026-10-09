///|
fn expert_score_recipe(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  routed_expert_count : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
  source_digest : AdvancedDecoderCudaSourceDigest,
) -> Bytes {
  let compiler_identity = compiler.identity()
  @utf8.encode(
    "lunaflux.advanced-decoder-routing-cuda-aot.v1" +
    "|family=expert-score-transform-f32" +
    "|numerical=finite-input-stable-sqrt-softplus-f32-v1" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|tokens=" +
    maximum_tokens.to_string() +
    "|routed-experts=" +
    routed_expert_count.to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
    ":" +
    target.supports_cublas_lt().to_string() +
    "|compiler=" +
    compiler_identity.toolchain_digest_sha256() +
    ":" +
    compiler_identity.major().to_string() +
    "." +
    compiler_identity.minor().to_string() +
    "." +
    compiler_identity.patch().to_string() +
    "|opt=" +
    compiler.optimization_level().to_string() +
    "|fma=" +
    compiler.fused_multiply_add().to_string() +
    "|reassociate=" +
    compiler.reassociate().to_string() +
    "|max-registers=" +
    compiler.max_registers().to_string() +
    "|launch=" +
    dimensions.grid_x().to_string() +
    ":" +
    dimensions.grid_y().to_string() +
    ":" +
    dimensions.grid_z().to_string() +
    ":" +
    dimensions.block_x().to_string() +
    ":" +
    dimensions.block_y().to_string() +
    ":" +
    dimensions.block_z().to_string() +
    ":" +
    dimensions.shared_memory_bytes().to_string() +
    "|operands=" +
    routing_operand_text(operands) +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
fn token_hash_recipe(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  hash_layer_count : Int,
  vocabulary_size : Int,
  experts_per_token : Int,
  routed_expert_count : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
  source_digest : AdvancedDecoderCudaSourceDigest,
) -> Bytes {
  let compiler_identity = compiler.identity()
  @utf8.encode(
    "lunaflux.advanced-decoder-routing-cuda-aot.v1" +
    "|family=token-hash-routing-i32" +
    "|numerical=bit-exact-int32-table-lookup-v1" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|tokens=" +
    maximum_tokens.to_string() +
    "|hash-layers=" +
    hash_layer_count.to_string() +
    "|vocabulary=" +
    vocabulary_size.to_string() +
    "|experts-per-token=" +
    experts_per_token.to_string() +
    "|routed-experts=" +
    routed_expert_count.to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
    ":" +
    target.supports_cublas_lt().to_string() +
    "|compiler=" +
    compiler_identity.toolchain_digest_sha256() +
    ":" +
    compiler_identity.major().to_string() +
    "." +
    compiler_identity.minor().to_string() +
    "." +
    compiler_identity.patch().to_string() +
    "|opt=" +
    compiler.optimization_level().to_string() +
    "|fma=" +
    compiler.fused_multiply_add().to_string() +
    "|reassociate=" +
    compiler.reassociate().to_string() +
    "|max-registers=" +
    compiler.max_registers().to_string() +
    "|launch=" +
    dimensions.grid_x().to_string() +
    ":" +
    dimensions.grid_y().to_string() +
    ":" +
    dimensions.grid_z().to_string() +
    ":" +
    dimensions.block_x().to_string() +
    ":" +
    dimensions.block_y().to_string() +
    ":" +
    dimensions.block_z().to_string() +
    ":" +
    dimensions.shared_memory_bytes().to_string() +
    "|operands=" +
    routing_operand_text(operands) +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
fn expert_score_source(
  maximum_tokens : Int,
  routed_expert_count : Int,
  function_symbol : String,
) -> Bytes {
  @utf8.encode(
    "#include <math.h>\n#include <stdint.h>\n" +
    "extern \"C\" __global__ void " +
    function_symbol +
    "(const int32_t *counts, const float *router_logits, float *routing_scores) {\n" +
    "  const int32_t live_tokens = counts[3];\n" +
    "  if (live_tokens < 0 || live_tokens > " +
    maximum_tokens.to_string() +
    ") return;\n" +
    "  const int32_t row = (int32_t)blockIdx.x;\n" +
    "  if (row >= live_tokens) return;\n" +
    "  const unsigned long long base = (unsigned long long)row * " +
    routed_expert_count.to_string() +
    "ULL;\n" +
    "  for (int32_t expert = (int32_t)threadIdx.x; expert < " +
    routed_expert_count.to_string() +
    "; expert += 256) {\n" +
    "    const unsigned long long index = base + (unsigned long long)expert;\n" +
    "    const float value = router_logits[index];\n" +
    "    const float softplus = value > 0.0f ? __fadd_rn(value, log1pf(expf(-value))) : log1pf(expf(value));\n" +
    "    routing_scores[index] = sqrtf(softplus);\n" +
    "  }\n}\n",
  )
}

///|
/// Lowers one exact advanced-decoder I32 token-hash table lookup. The table is
/// row-major `[vocabulary_size, experts_per_token]`; runtime `counts[3]`
/// selects the live token prefix. This candidate remains non-bindable pending
/// offline compilation and correctness qualification.
pub fn lower_token_hash_routing_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderRoutingCudaCandidate raise AdvancedDecoderCudaAotError {
  if maximum_tokens <= 0 || maximum_tokens > limits.maximum_tokens {
    raise Invalid(Limit)
  }
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() {
    raise Invalid(Compiler)
  }
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      TokenHashRouting(
        hash_layer_count,
        vocabulary_size,
        experts_per_token,
        routed_expert_count
      ),
      TokenHashRoutingAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if hash_layer_count <= 0 ||
    vocabulary_size <= 0 ||
    experts_per_token <= 0 ||
    experts_per_token > 256 ||
    routed_expert_count <= 0 ||
    experts_per_token > routed_expert_count {
    raise Invalid(Shape)
  }
  let token_bytes = checked_mul(maximum_tokens.to_int64(), 4L)
  let table_bytes = checked_mul(
    checked_mul(vocabulary_size.to_int64(), experts_per_token.to_int64()),
    4L,
  )
  let output_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), experts_per_token.to_int64()),
    4L,
  )
  if token_bytes > 2147483647L ||
    table_bytes > 2147483647L ||
    output_bytes > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    operand_with_alignment(StepCountsInput, 20L, 4L),
    operand_with_alignment(TokenIdsInput, token_bytes, 4L),
    operand_with_alignment(TokenHashTableInput, table_bytes, 4L),
    operand_with_alignment(RoutingIndicesOutput, output_bytes, 4L),
  ]
  let function_symbol = "lunaflux_advanced_decoder_token_hash_i32_v1"
  let rendered = @decoder_token_hash_cuda_source.token_hash_lookup_i32_source(
    vocabulary_size~,
    experts_per_token~,
    routed_expert_count~,
    maximum_rows=maximum_tokens,
    function_symbol~,
  ) catch {
    InvalidVocabularySize => raise TokenHashSource(InvalidVocabularySize)
    InvalidExpertsPerToken => raise TokenHashSource(InvalidExpertsPerToken)
    InvalidRoutedExpertCount => raise TokenHashSource(InvalidRoutedExpertCount)
    InvalidMaximumRows => raise TokenHashSource(InvalidMaximumRows)
    InvalidFunctionSymbol => raise TokenHashSource(InvalidFunctionSymbol)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = token_hash_recipe(
    requirements, requirement_ordinal, maximum_tokens, hash_layer_count, vocabulary_size,
    experts_per_token, routed_expert_count, target, compiler, dimensions, operands,
    source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: TokenHashRoutingI32,
    numerical_contract: BitExactInt32TableLookupV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    routed_expert_count,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol,
    source,
    source_digest,
    recipe: encoded_recipe,
    recipe_digest: AdvancedDecoderCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(encoded_recipe)),
    ),
  }
}

///|
/// Lowers one exact advanced-decoder sqrt-softplus expert-score transform.
/// Inputs and outputs are finite F32 row-major `[maximum_tokens,
/// routed_expert_count]`; runtime `counts[3]` selects the live row prefix.
/// This candidate is deliberately non-bindable pending offline compilation
/// and independent CUDA numerical qualification.
pub fn lower_expert_score_transform_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderRoutingCudaCandidate raise AdvancedDecoderCudaAotError {
  if maximum_tokens <= 0 || maximum_tokens > limits.maximum_tokens {
    raise Invalid(Limit)
  }
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() {
    raise Invalid(Compiler)
  }
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      ExpertScoreTransform(SqrtSoftplus, routed_expert_count),
      ExpertScoreTransformAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if routed_expert_count <= 0 {
    raise Invalid(Shape)
  }
  let score_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), routed_expert_count.to_int64()),
    4L,
  )
  if score_bytes > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    operand_with_alignment(StepCountsInput, 20L, 4L),
    operand(RoutingScoresInput, score_bytes),
    operand(RoutingScoresOutput, score_bytes),
  ]
  let function_symbol = "lunaflux_advanced_decoder_expert_score_sqrt_softplus_f32_v1"
  let source = expert_score_source(
    maximum_tokens, routed_expert_count, function_symbol,
  )
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = expert_score_recipe(
    requirements, requirement_ordinal, maximum_tokens, routed_expert_count, target,
    compiler, dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: ExpertScoreTransformF32,
    numerical_contract: FiniteInputStableSqrtSoftplusF32V1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    routed_expert_count,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol,
    source,
    source_digest,
    recipe: encoded_recipe,
    recipe_digest: AdvancedDecoderCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(encoded_recipe)),
    ),
  }
}
