///|
fn inverse_rotary_recipe(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  operation : @advanced_decoder_kernel_requirements.AdvancedDecoderSemanticOperation,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
  source_digest : AdvancedDecoderCudaSourceDigest,
) -> Bytes {
  let parameters = match operation {
    AttentionOutputInverseRotaryV1(theta, original, factor, fast, slow) =>
      theta.to_string() +
      ":" +
      original.to_string() +
      ":" +
      factor.to_string() +
      ":" +
      fast.to_string() +
      ":" +
      slow.to_string()
    _ => "invalid"
  }
  let geometry = requirements.geometry()
  let identity = compiler.identity()
  let operand_text = StringBuilder()
  for index, operand in operands {
    if index > 0 {
      operand_text.write_char(',')
    }
    operand_text.write_string(
      match operand.role() {
        StepCountsInput => "step-counts"
        PositionInput => "positions"
        AttentionOutputRotaryInOut => "attention-output-inout"
        _ => "forbidden-role"
      },
    )
    operand_text.write_char('@')
    operand_text.write_string(operand.byte_count().to_string())
    operand_text.write_char('/')
    operand_text.write_string(operand.alignment().to_string())
  }
  @utf8.encode(
    "lunaflux.advanced-decoder-attention-output-inverse-rotary-cuda-aot.v1" +
    "|family=attention-output-inverse-scaled-yarn-bf16-v1" +
    "|numerical=prefix-bit-exact-adjacent-pair-inverse-f32-yarn-bf16-round-v1" +
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
    "|parameters=" +
    parameters +
    "|geometry=" +
    geometry.attention_head_count().to_string() +
    ":" +
    geometry.attention_head_dimension().to_string() +
    ":" +
    geometry.rope_head_dimension().to_string() +
    ":" +
    geometry.maximum_position_count().to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    ":" +
    target.supports_bf16().to_string() +
    "|compiler=" +
    identity.toolchain_digest_sha256() +
    ":" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
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
    ":1:" +
    dimensions.block_x().to_string() +
    ":1:1:0" +
    "|operands=" +
    operand_text.to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lowers only the official full-head attention-output inverse scaled-YaRN
/// rotation. The non-RoPE prefix is copied bit-exactly; no output projection,
/// cache state, compilation, loader, or execution authority is included.
pub fn lower_attention_output_inverse_rotary_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderInverseRotaryCudaCandidate raise AdvancedDecoderCudaAotError {
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
      AttentionOutputInverseRotaryV1(
        compressed_theta,
        original_positions,
        factor,
        fast,
        slow
      ),
      AttentionOutputInverseRotaryV1Aot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let geometry = requirements.geometry()
  let head_count = geometry.attention_head_count()
  let head_dimension = geometry.attention_head_dimension()
  let rope_dimension = geometry.rope_head_dimension()
  let maximum_positions = geometry.maximum_position_count()
  if head_count <= 0 ||
    head_dimension <= 0 ||
    head_dimension % 2 != 0 ||
    head_dimension > 2048 ||
    rope_dimension != 64 ||
    rope_dimension > head_dimension ||
    original_positions <= 0 ||
    original_positions > maximum_positions {
    raise Invalid(Shape)
  }
  let position_bytes = checked_mul(maximum_tokens.to_int64(), 4L)
  let output_bytes = checked_mul(
    checked_mul(
      checked_mul(maximum_tokens.to_int64(), head_count.to_int64()),
      head_dimension.to_int64(),
    ),
    2L,
  )
  if position_bytes > 2147483647L || output_bytes > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
    grid_y=head_count,
    grid_z=1,
    block_x=head_dimension / 2,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    operand_with_alignment(StepCountsInput, 20L, 4L),
    operand_with_alignment(PositionInput, position_bytes, 4L),
    operand(AttentionOutputRotaryInOut, output_bytes),
  ]
  let function_symbol = "lunaflux_attention_output_inverse_scaled_yarn_bf16_v1"
  let rendered = @decoder_scaled_rope_cuda_source.inverse_scaled_yarn_attention_output_bf16_source(
    head_count~,
    head_dimension~,
    rope_dimension~,
    maximum_rows=maximum_tokens,
    maximum_positions~,
    original_positions~,
    compressed_theta~,
    scaling_factor=factor,
    beta_fast=fast,
    beta_slow=slow,
    function_symbol~,
  ) catch {
    error => raise RotarySource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = inverse_rotary_recipe(
    requirements,
    requirement_ordinal,
    maximum_tokens,
    requirement.operation(),
    target,
    compiler,
    dimensions,
    operands,
    source_digest,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: AttentionOutputInverseScaledYarnBf16V1,
    numerical_contract: PrefixBitExactAdjacentPairInverseF32YarnBf16RoundV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    head_count,
    head_dimension,
    rope_dimension,
    maximum_positions,
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
