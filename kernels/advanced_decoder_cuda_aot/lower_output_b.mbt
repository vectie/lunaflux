///|
pub fn lower_attention_output_b_projection_cuda_aot_candidate(
  requirements : @advanced_decoder_kernel_requirements.AdvancedDecoderKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : AdvancedDecoderCudaAotLimits,
) -> AdvancedDecoderOutputBProjectionCudaCandidate raise AdvancedDecoderCudaAotError {
  if maximum_tokens <= 0 || maximum_tokens > limits.maximum_tokens {
    raise Invalid(Limit)
  }
  if !target.supports_bf16() || target.compute_major() < 9 {
    raise Invalid(Target)
  }
  let compiler_identity = compiler.identity()
  if compiler.reassociate() ||
    compiler.fused_multiply_add() ||
    compiler_identity.major() < 12 ||
    (compiler_identity.major() == 12 && compiler_identity.minor() < 8) {
    raise Invalid(Compiler)
  }
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      AttentionOutputBProjectionV1(rank, groups, hidden),
      AttentionOutputBProjectionV1Aot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let geometry = requirements.geometry()
  if rank <= 0 ||
    groups <= 0 ||
    hidden != geometry.hidden_size() ||
    rank > 65536 / groups {
    raise Invalid(Shape)
  }
  let input_width = rank * groups
  if input_width % 128 != 0 || hidden % 128 != 0 {
    raise Invalid(Shape)
  }
  let input_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), input_width.to_int64()),
    2L,
  )
  let weight_bytes = checked_mul(hidden.to_int64(), input_width.to_int64())
  let scale_bytes = checked_mul(
    (hidden / 128).to_int64(),
    (input_width / 128).to_int64(),
  )
  let output_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), hidden.to_int64()),
    2L,
  )
  for count in [input_bytes, weight_bytes, scale_bytes, output_bytes] {
    if count > 2147483647L {
      raise SizeOverflow
    }
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
    grid_y=(hidden + 255) / 256,
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
    operand(AttentionOutputRankInput, input_bytes),
    operand(AttentionOutputBWeightCodesInput, weight_bytes),
    operand(AttentionOutputBWeightScalesInput, scale_bytes),
    operand(HiddenOutput, output_bytes),
  ]
  let function_symbol = "lunaflux_attention_output_b_block_fp8_ue8m0_v1"
  let rendered = @block_fp8_ue8m0_projection_cuda_source.block_fp8_ue8m0_projection_cuda_source(
    input_width~,
    output_width=hidden,
    maximum_rows=maximum_tokens,
    function_symbol~,
  ) catch {
    error => raise BlockFp8Ue8M0Source(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = AdvancedDecoderCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = @utf8.encode(
    "lunaflux.advanced-decoder-output-b-block-fp8-cuda-aot.v1" +
    "|numerical=bf16-input-dynamic-row-block128-e4m3-ue8m0-ordered-f32-bf16-rne-v1" +
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
    input_width.to_string() +
    ":" +
    hidden.to_string() +
    ":128" +
    "|tensor-parallel-world-size=1|collective=external-unowned" +
    "|compiler=" +
    compiler_identity.toolchain_digest_sha256() +
    ":" +
    compiler_identity.major().to_string() +
    "." +
    compiler_identity.minor().to_string() +
    "." +
    compiler_identity.patch().to_string() +
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
    ((hidden + 255) / 256).to_string() +
    ":1:256:1:1:0" +
    "|operands=counts@20/4,rank-input@" +
    input_bytes.to_string() +
    "/16,e4m3-weight@" +
    weight_bytes.to_string() +
    "/16,ue8m0-scale@" +
    scale_bytes.to_string() +
    "/16,hidden-output@" +
    output_bytes.to_string() +
    "/16" +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: BlockFp8E4M3Ue8M0ProjectionV1,
    numerical_contract: PerRowBlock128Ue8M0OrderedF32Bf16RoundV1,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    target,
    maximum_tokens,
    input_width,
    output_width: hidden,
    block_width: 128,
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
