///|
fn kda_short_convolution_recipe(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  ordinal : Int,
  channel_count : Int,
  kernel_size : Int,
  maximum_tokens : Int,
  maximum_sequences : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : ArrayView[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand],
  source_digest : GlmHybridCudaSourceDigest,
) -> Bytes {
  let compiler_identity = compiler.identity()
  let operand_text = StringBuilder()
  for index, item in operands {
    if index > 0 {
      operand_text.write_char(',')
    }
    operand_text.write_string(dense_projection_operand_tag(item.role()))
    operand_text.write_char('@')
    operand_text.write_string(item.byte_count().to_string())
    operand_text.write_char('/')
    operand_text.write_string(item.alignment().to_string())
  }
  @utf8.encode(
    "lunaflux.glm-hybrid-kda-short-convolution-cuda-aot.v1" +
    "|family=reusable-depthwise-causal-silu-bf16" +
    "|invocations=independent-q-k-v" +
    "|layout=input-output-token-major-channel-minor" +
    "|weights=channel-kernel-oldest-to-current-bf16-no-bias" +
    "|checkpoint-weight=channel-singleton-kernel-zero-copy-squeeze" +
    "|state=sequence-channel-history-oldest-first-bf16-explicit-in-out" +
    "|sequences=counts2-csr-offsets|live-rows=counts3" +
    "|zero-history=all-zero-initial-state" +
    "|numerical=ordered-f32-products-and-sum-f32-silu-one-bf16-output-round" +
    "|state-copy=bit-exact-bf16" +
    "|excluded=projection-recurrent-update-residual-norm-cache-ownership" +
    "|qualification=correctness-source-only-non-bindable" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|geometry=" +
    channel_count.to_string() +
    ":" +
    kernel_size.to_string() +
    ":" +
    maximum_tokens.to_string() +
    ":" +
    maximum_sequences.to_string() +
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
    operand_text.to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
/// Lowers the official GLM-5.3-Flash short convolution as one reusable Q/K/V
/// stream invocation. Projection, packing, recurrence, and cache ownership are
/// deliberately outside this artifact.
pub fn lower_kda_short_convolution_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  maximum_sequences~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridKdaShortConvolutionCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (KdaShortConvolution(channel_count, kernel_size), KdaShortConvolutionAot) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if channel_count != 8192 || kernel_size != 4 {
    raise Invalid(Shape)
  }
  if maximum_tokens <= 0 ||
    maximum_tokens > limits.maximum_tokens ||
    maximum_tokens > requirements.geometry().context_length() ||
    maximum_sequences <= 0 ||
    maximum_sequences > maximum_tokens {
    raise Invalid(Limit)
  }
  if !target.supports_bf16() {
    raise Invalid(Target)
  }
  if compiler.reassociate() || compiler.fused_multiply_add() {
    raise Invalid(Compiler)
  }
  let history = kernel_size - 1
  let output_cells = checked_mul(
    maximum_tokens.to_int64(),
    channel_count.to_int64(),
  )
  let state_cells = checked_mul(
    checked_mul(maximum_sequences.to_int64(), channel_count.to_int64()),
    history.to_int64(),
  )
  let work = if output_cells > state_cells { output_cells } else { state_cells }
  if work > 9223372036854775552L || (work + 255L) / 256L > 2147483647L {
    raise SizeOverflow
  }
  let input_bytes = checked_mul(output_cells, 2L)
  let weight_bytes = checked_mul(
    checked_mul(channel_count.to_int64(), kernel_size.to_int64()),
    2L,
  )
  let state_bytes = checked_mul(state_cells, 2L)
  let offset_bytes = checked_mul(maximum_sequences.to_int64() + 1L, 4L)
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(KdaSequenceOffsetsInput, offset_bytes, 4L),
    dense_projection_operand(KdaProjectedInput, input_bytes, 16L),
    dense_projection_operand(KdaShortConvolutionWeights, weight_bytes, 16L),
    dense_projection_operand(KdaConvolutionInitialStateInput, state_bytes, 16L),
    dense_projection_operand(KdaConvolvedOutput, input_bytes, 16L),
    dense_projection_operand(KdaConvolutionFinalStateOutput, state_bytes, 16L),
  ]
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=((work + 255L) / 256L).to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let rendered = @causal_short_convolution_cuda_source.depthwise_causal_silu_bf16_source(
    channel_count~,
    kernel_size~,
    maximum_rows=maximum_tokens,
    maximum_sequences~,
    function_symbol="lunaflux_glm_hybrid_kda_short_convolution_bf16_v1",
  ) catch {
    error => raise CausalShortConvolutionSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = kda_short_convolution_recipe(
    requirements, requirement_ordinal, channel_count, kernel_size, maximum_tokens,
    maximum_sequences, target, compiler, dimensions, operands, source_digest,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: ReusableDepthwiseCausalSiluBf16,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    channel_count,
    kernel_size,
    maximum_tokens,
    maximum_sequences,
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
