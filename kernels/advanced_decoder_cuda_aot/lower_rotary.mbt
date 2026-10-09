///|
fn rotary_operand_text(
  operands : ArrayView[
    @advanced_decoder_artifact_admission.AdvancedDecoderLaunchOperand,
  ],
) -> String {
  let text = StringBuilder()
  for index, item in operands {
    if index > 0 {
      text.write_char(',')
    }
    let role = match item.role() {
      StepCountsInput => "step-counts"
      PositionInput => "positions"
      QueryInput => "query-input"
      QueryOutput => "query-output"
      KeyValueInput => "key-value-input"
      KeyValueOutput => "key-value-output"
      _ => "forbidden-non-rope-role"
    }
    text.write_string(role)
    text.write_char('@')
    text.write_string(item.byte_count().to_string())
    text.write_char('/')
    text.write_string(item.alignment().to_string())
  }
  text.to_string()
}

///|
fn rotary_recipe(
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
  let compiler_identity = compiler.identity()
  let parameters = match operation {
    RotaryPositionEncoding(theta, compressed, original, factor, fast, slow) =>
      theta.to_string() +
      ":" +
      compressed.to_string() +
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
  @utf8.encode(
    "lunaflux.advanced-decoder-rotary-cuda-aot.v1" +
    "|family=compressed-scaled-yarn-rope-bf16" +
    "|numerical=adjacent-pair-f32-yarn-bf16-round-v1" +
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
    geometry.kv_head_count().to_string() +
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
    rotary_operand_text(operands) +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lower only the compressed/scaled YaRN path. Plain-theta sliding-layer RoPE
/// is intentionally outside this candidate and no mode selector is invented.
pub fn lower_scaled_rotary_position_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderRotaryCudaCandidate raise AdvancedDecoderCudaAotError {
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
      RotaryPositionEncoding(
        _,
        compressed_theta,
        original_positions,
        factor,
        fast,
        slow
      ),
      ScaledRotaryPositionAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let geometry = requirements.geometry()
  let query_heads = geometry.attention_head_count()
  let kv_heads = geometry.kv_head_count()
  let rope_dimension = geometry.rope_head_dimension()
  let maximum_positions = geometry.maximum_position_count()
  if rope_dimension != 64 ||
    query_heads <= 0 ||
    kv_heads <= 0 ||
    maximum_positions <= 0 {
    raise Invalid(Shape)
  }
  let query_bytes = checked_mul(
    checked_mul(
      checked_mul(maximum_tokens.to_int64(), query_heads.to_int64()),
      rope_dimension.to_int64(),
    ),
    2L,
  )
  let kv_bytes = checked_mul(
    checked_mul(
      checked_mul(maximum_tokens.to_int64(), kv_heads.to_int64()),
      rope_dimension.to_int64(),
    ),
    2L,
  )
  let position_bytes = checked_mul(maximum_tokens.to_int64(), 4L)
  if query_bytes > 2147483647L ||
    kv_bytes > 2147483647L ||
    position_bytes > 2147483647L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
    grid_y=query_heads + kv_heads,
    grid_z=1,
    block_x=rope_dimension / 2,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let operands = [
    operand_with_alignment(StepCountsInput, 20L, 4L),
    operand_with_alignment(PositionInput, position_bytes, 4L),
    operand(QueryInput, query_bytes),
    operand(QueryOutput, query_bytes),
    operand(KeyValueInput, kv_bytes),
    operand(KeyValueOutput, kv_bytes),
  ]
  let symbol = "lunaflux_advanced_decoder_scaled_yarn_rope_bf16_v1"
  let rendered = @decoder_scaled_rope_cuda_source.scaled_yarn_rope_bf16_source(
    query_head_count=query_heads,
    key_value_head_count=kv_heads,
    rope_dimension~,
    maximum_rows=maximum_tokens,
    maximum_positions~,
    original_positions~,
    compressed_theta~,
    scaling_factor=factor,
    beta_fast=fast,
    beta_slow=slow,
    function_symbol=symbol,
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
  let encoded_recipe = rotary_recipe(
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
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: CompressedScaledYarnRopeBf16,
    numerical_contract: AdjacentPairF32YarnBf16RoundV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    query_head_count: query_heads,
    key_value_head_count: kv_heads,
    rope_dimension,
    maximum_positions,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol: symbol,
    source,
    source_digest,
    recipe: encoded_recipe,
    recipe_digest: AdvancedDecoderCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(encoded_recipe)),
    ),
  }
}
