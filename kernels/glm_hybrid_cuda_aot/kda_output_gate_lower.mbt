///|
pub fn lower_kda_output_gate_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridKdaOutputGateCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      KdaOutputGateProjection(hidden_width, head_count, head_dimension),
      KdaOutputGateProjectionAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if hidden_width != 4096 ||
    head_count != 64 ||
    head_dimension != 128 ||
    requirements.geometry().hidden_size() != hidden_width {
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
  let output_width = checked_mul(
    head_count.to_int64(),
    head_dimension.to_int64(),
  )
  let hidden_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), hidden_width.to_int64()),
    2L,
  )
  let low_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), head_dimension.to_int64()),
    2L,
  )
  let output_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), output_width),
    2L,
  )
  let first_weight_bytes = checked_mul(
    checked_mul(head_dimension.to_int64(), hidden_width.to_int64()),
    2L,
  )
  let second_weight_bytes = checked_mul(
    checked_mul(output_width, head_dimension.to_int64()),
    2L,
  )
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(HiddenInput, hidden_bytes, 16L),
    dense_projection_operand(
      KdaOutputGateLowRankWeights,
      first_weight_bytes,
      16L,
    ),
    dense_projection_operand(
      KdaOutputGateExpansionWeights,
      second_weight_bytes,
      16L,
    ),
    dense_projection_operand(KdaOutputGateLowRankOutput, low_bytes, 16L),
    dense_projection_operand(KdaOutputGateOutput, output_bytes, 16L),
  ]
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
  let rendered = @two_stage_projection_cuda_source.bias_free_bf16_low_rank_projection_source(
    input_width=hidden_width,
    intermediate_width=head_dimension,
    output_width=output_width.to_int(),
    maximum_rows=maximum_tokens,
    function_symbol="lunaflux_glm_hybrid_kda_output_gate_bf16_v1",
  ) catch {
    error => raise TwoStageProjectionSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let compiler_identity = compiler.identity()
  let recipe = @utf8.encode(
    "lunaflux.glm-hybrid-kda-output-gate-cuda-aot.v1" +
    "|family=serial-bias-free-two-stage-bf16" +
    "|layout=hidden-to-head-dimension-intermediate-to-head-channel-gate" +
    "|weights=output-major-bf16-no-bias" +
    "|numerical=ordered-rn-f32-sums-bf16-intermediate-round-bf16-output-round" +
    "|qualification=serial-correctness-source-only-non-bindable" +
    "|excluded=gated-rmsnorm-final-dense-residual-cache" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|geometry=" +
    hidden_width.to_string() +
    ":" +
    head_count.to_string() +
    ":" +
    head_dimension.to_string() +
    ":" +
    maximum_tokens.to_string() +
    "|sm=" +
    target.compute_major().to_string() +
    "." +
    target.compute_minor().to_string() +
    "|compiler=" +
    compiler_identity.toolchain_digest_sha256() +
    "|fma=" +
    compiler.fused_multiply_add().to_string() +
    "|reassociate=" +
    compiler.reassociate().to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: SerialBiasFreeTwoStageBf16,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    hidden_width,
    head_count,
    head_dimension,
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
