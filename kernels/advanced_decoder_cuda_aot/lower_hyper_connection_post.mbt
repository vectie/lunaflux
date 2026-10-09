///|
fn hyper_connection_post_operand_text(
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
      HyperConnectionBranchInput => "hyper-branch-input"
      HyperConnectionStateInput => "hyper-state-input"
      HyperConnectionPostInput => "hyper-post-input"
      HyperConnectionCombinationInput => "hyper-combination-input"
      HyperConnectionCombinedOutput => "hyper-combined-output"
      _ => "forbidden-non-post-combination-role"
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
fn hyper_connection_post_recipe(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  ordinal : Int,
  maximum_tokens : Int,
  stream_width : Int,
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
    "lunaflux.advanced-decoder-hyper-connection-post-cuda-aot.v1" +
    "|family=post-combine-bf16-v1" +
    "|numerical=ordered-f32-post-combination-single-bf16-rne-v1" +
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
    "|geometry=" +
    stream_width.to_string() +
    ":" +
    requirements.geometry().hidden_size().to_string() +
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
    hyper_connection_post_operand_text(operands) +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lowers exact V1 post-combination only. Every output rounds once to BF16.
pub fn lower_hyper_connection_post_combine_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderHyperConnectionPostCombineCudaCandidate raise AdvancedDecoderCudaAotError {
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
      HyperConnectionPostCombineV1(stream_width),
      HyperConnectionPostCombineV1Aot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let hidden_width = requirements.geometry().hidden_size()
  let rows = maximum_tokens.to_int64()
  let stream = stream_width.to_int64()
  let hidden = hidden_width.to_int64()
  let branch_bytes = checked_mul(checked_mul(rows, hidden), 2L)
  let state_bytes = checked_mul(
    checked_mul(checked_mul(rows, stream), hidden),
    2L,
  )
  let post_bytes = checked_mul(checked_mul(rows, stream), 4L)
  let combination_bytes = checked_mul(
    checked_mul(checked_mul(rows, stream), stream),
    4L,
  )
  for bytes in [branch_bytes, state_bytes, post_bytes, combination_bytes] {
    if bytes > 2147483647L {
      raise SizeOverflow
    }
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
    operand(HyperConnectionBranchInput, branch_bytes),
    operand(HyperConnectionStateInput, state_bytes),
    operand(HyperConnectionPostInput, post_bytes),
    operand(HyperConnectionCombinationInput, combination_bytes),
    operand(HyperConnectionCombinedOutput, state_bytes),
  ]
  let symbol = "lunaflux_advanced_decoder_hc_post_combine_v1"
  let rendered = @decoder_hyper_connection_cuda_source.post_combine_v1_bf16_source(
    stream_width~,
    hidden_width~,
    maximum_rows=maximum_tokens,
    function_symbol=symbol,
  ) catch {
    error => raise HyperConnectionSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let encoded_recipe = hyper_connection_post_recipe(
    requirements, requirement_ordinal, maximum_tokens, stream_width, target, compiler,
    dimensions, operands, source_digest,
  )
  if encoded_recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: PostCombineBf16V1,
    numerical_contract: OrderedF32PostCombinationBf16RoundV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    stream_width,
    hidden_width,
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
