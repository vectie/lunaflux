///|
pub fn lower_attention_output_a_projection_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderOutputAProjectionCudaCandidate raise AdvancedDecoderCudaAotError {
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
      AttentionOutputAProjectionV1(rank, groups, heads, head_dimension),
      AttentionOutputAProjectionV1Aot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let geometry = requirements.geometry()
  if rank <= 0 ||
    rank > 4096 ||
    groups <= 0 ||
    groups > 128 ||
    heads <= 0 ||
    heads != geometry.attention_head_count() ||
    heads % groups != 0 ||
    head_dimension <= 0 ||
    head_dimension != geometry.attention_head_dimension() {
    raise Invalid(Shape)
  }
  let group_input = checked_mul(
    (heads / groups).to_int64(),
    head_dimension.to_int64(),
  )
  let input_values = checked_mul(
    checked_mul(maximum_tokens.to_int64(), heads.to_int64()),
    head_dimension.to_int64(),
  )
  let weight_values = checked_mul(
    checked_mul(groups.to_int64(), rank.to_int64()),
    group_input,
  )
  let output_values = checked_mul(
    checked_mul(maximum_tokens.to_int64(), groups.to_int64()),
    rank.to_int64(),
  )
  let input_bytes = checked_mul(input_values, 2L)
  let weight_bytes = checked_mul(weight_values, 2L)
  let output_bytes = checked_mul(output_values, 2L)
  if group_input > 2147483647L ||
    input_bytes > 2147483647L ||
    weight_bytes > 2147483647L ||
    output_bytes > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
    grid_y=groups,
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
    operand(AttentionOutputProjectionInput, input_bytes),
    operand(AttentionOutputAWeightsInput, weight_bytes),
    operand(AttentionOutputRankOutput, output_bytes),
  ]
  let function_symbol = "lunaflux_grouped_attention_output_a_bf16_v1"
  let rendered = @decoder_projection_cuda_source.grouped_projection_bf16_source(
    group_count=groups,
    input_width=group_input.to_int(),
    output_width=rank,
    maximum_rows=maximum_tokens,
    function_symbol~,
  ) catch {
    error => raise ProjectionSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let identity = compiler.identity()
  let recipe = @utf8.encode(
    "lunaflux.advanced-decoder-output-a-cuda-aot.v1" +
    "|numerical=converted-bf16-group-local-ordered-f32-bf16-rne-v1" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|geometry=" +
    maximum_tokens.to_string() +
    ":" +
    groups.to_string() +
    ":" +
    heads.to_string() +
    ":" +
    head_dimension.to_string() +
    ":" +
    rank.to_string() +
    "|compiler=" +
    identity.toolchain_digest_sha256() +
    ":" +
    compiler.optimization_level().to_string() +
    ":" +
    compiler.fused_multiply_add().to_string() +
    ":" +
    compiler.reassociate().to_string() +
    ":" +
    compiler.max_registers().to_string() +
    "|target=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
    "|launch=" +
    maximum_tokens.to_string() +
    ":" +
    groups.to_string() +
    ":1:256:1:1:0" +
    "|operands=step-counts@20/4,attention-output-input@" +
    input_bytes.to_string() +
    "/16,converted-grouped-weight@" +
    weight_bytes.to_string() +
    "/16,rank-output@" +
    output_bytes.to_string() +
    "/16" +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: GroupedOutputAProjectionBf16V1,
    numerical_contract: ConvertedGroupedWeightOrderedF32Bf16RoundV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    group_count: groups,
    input_width_per_group: group_input.to_int(),
    output_rank: rank,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol,
    source,
    source_digest,
    recipe,
    recipe_digest: AdvancedDecoderCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}
