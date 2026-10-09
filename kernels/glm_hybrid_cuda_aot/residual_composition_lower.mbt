///|
pub fn lower_moe_decoder_residual_add_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridMoeFeedForwardCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (MoeDecoderResidualAdd(hidden_width), MoeDecoderResidualAddAot) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if hidden_width != 6144 ||
    hidden_width != requirements.geometry().hidden_size() {
    raise Invalid(Shape)
  }
  validate_moe_feed_forward_environment(
    requirements, maximum_tokens, target, compiler, limits,
  )
  let row_cells = checked_mul(
    maximum_tokens.to_int64(),
    hidden_width.to_int64(),
  )
  let row_bytes = checked_mul(row_cells, 2L)
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(DecoderResidualInput, row_bytes, 16L),
    dense_projection_operand(MoeCombinedOutput, row_bytes, 16L),
    dense_projection_operand(DecoderResidualOutput, row_bytes, 16L),
  ]
  let rendered = @bf16_binary_cuda_source.bf16_add_source(
    width=hidden_width,
    maximum_rows=maximum_tokens,
    function_symbol="lunaflux_glm_hybrid_moe_decoder_residual_add_bf16_v1",
  ) catch {
    error => raise Bf16BinarySource(error)
  }
  moe_feed_forward_candidate(
    FullDecoderResidualAdd,
    DecoderResidualF32AddBf16Round,
    requirements,
    requirement,
    requirement_ordinal,
    maximum_tokens,
    target,
    moe_feed_forward_dimensions(maximum_tokens, hidden_width),
    operands,
    compiler,
    rendered.function_symbol(),
    rendered.source_bytes(),
    "moe-decoder-outer-residual-add",
    hidden_width.to_string() + ":" + maximum_tokens.to_string(),
    "f32-add-one-bf16-round",
    "routing-routed-shared-combine-norm-next-layer-residual-aliasing",
    limits,
  )
}

///|
pub fn lower_hyper_connection_mix_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridMoeFeedForwardCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (HyperConnectionMix(stream_count), Advanced(HyperConnectionMixAot)) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let hidden_width = requirements.geometry().hidden_size()
  if hidden_width != 4096 || stream_count != 4 {
    raise Invalid(Shape)
  }
  validate_moe_feed_forward_environment(
    requirements, maximum_tokens, target, compiler, limits,
  )
  let rows = maximum_tokens.to_int64()
  let hidden = hidden_width.to_int64()
  let streams = stream_count.to_int64()
  let branch_bytes = checked_mul(checked_mul(rows, hidden), 2L)
  let stream_bytes = checked_mul(branch_bytes, streams)
  let post_bytes = checked_mul(checked_mul(rows, streams), 4L)
  let combine_bytes = checked_mul(
    checked_mul(checked_mul(rows, streams), streams),
    4L,
  )
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(HyperPostControlInput, post_bytes, 16L),
    dense_projection_operand(HyperCombineControlInput, combine_bytes, 16L),
    dense_projection_operand(HyperResidualStreamsInput, stream_bytes, 16L),
    dense_projection_operand(HyperSublayerOutputInput, branch_bytes, 16L),
    dense_projection_operand(HyperMixedStreamsOutput, stream_bytes, 16L),
  ]
  let rendered = @hyper_connection_mix_cuda_source.hyper_connection_mix_bf16_source(
    hidden_width~,
    stream_count~,
    maximum_rows=maximum_tokens,
    function_symbol="lunaflux_glm_hybrid_hyper_connection_mix_bf16_v1",
  ) catch {
    error => raise HyperConnectionMixSource(error)
  }
  let dimensions = moe_feed_forward_dimensions(
    maximum_tokens,
    hidden_width * stream_count,
  )
  moe_feed_forward_candidate(
    FlashHyperConnectionMix4,
    HyperControlBf16OrderedResidualMix,
    requirements,
    requirement,
    requirement_ordinal,
    maximum_tokens,
    target,
    dimensions,
    operands,
    compiler,
    rendered.function_symbol(),
    rendered.source_bytes(),
    "hyper-connection-outer-residual-mix",
    hidden_width.to_string() +
    ":" +
    stream_count.to_string() +
    ":" +
    maximum_tokens.to_string(),
    "f32-controls-cast-bf16-products-ordered-f32-residual-sum-bf16-stage-rounds",
    "hyper-function-sinkhorn-branch-compute-next-layer-collapse-in-place-aliasing",
    limits,
  )
}
