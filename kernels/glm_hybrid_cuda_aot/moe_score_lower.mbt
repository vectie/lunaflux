///|
pub fn lower_moe_sigmoid_correction_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridMoeScoreCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      MoeSigmoidAndCorrectionBias(expert_count),
      MoeSigmoidAndCorrectionBiasAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let family = match (requirements.geometry().hidden_size(), expert_count) {
    (6144, 256) => FullSigmoidCorrection256
    (4096, 288) => FlashSigmoidCorrection288
    _ => raise Invalid(Shape)
  }
  if maximum_tokens <= 0 ||
    maximum_tokens > limits.maximum_tokens ||
    maximum_tokens > requirements.geometry().context_length() {
    raise Invalid(Limit)
  }
  if compiler.reassociate() {
    raise Invalid(Compiler)
  }
  let score_cells = checked_mul(
    maximum_tokens.to_int64(),
    expert_count.to_int64(),
  )
  let score_bytes = checked_mul(score_cells, 4L)
  let bias_bytes = checked_mul(expert_count.to_int64(), 4L)
  let grid = (score_cells + 255L) / 256L
  if grid <= 0L || grid > 2147483647L {
    raise SizeOverflow
  }
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(MoeRouterFloat32LogitsOutput, score_bytes, 16L),
    dense_projection_operand(MoeCorrectionBiasInput, bias_bytes, 16L),
    dense_projection_operand(MoeUncorrectedScoresOutput, score_bytes, 16L),
    dense_projection_operand(MoeChoiceScoresOutput, score_bytes, 16L),
  ]
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=grid.to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let rendered = @sigmoid_correction_cuda_source.sigmoid_and_correction_f32_source(
    expert_count~,
    maximum_rows=maximum_tokens,
    function_symbol="lunaflux_glm_hybrid_moe_sigmoid_correction_f32_v1",
  ) catch {
    error => raise SigmoidCorrectionSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let family_tag = match family {
    FullSigmoidCorrection256 => "full-experts256"
    FlashSigmoidCorrection288 => "flash-experts288"
  }
  let compiler_identity = compiler.identity()
  let recipe = @utf8.encode(
    "lunaflux.glm-hybrid-moe-sigmoid-correction-cuda-aot.v1" +
    "|family=" +
    family_tag +
    "|input=row-major-f32-raw-router-logits" +
    "|bias=f32-expert-correction-vector" +
    "|outputs=separate-row-major-f32-uncorrected-and-choice-scores" +
    "|aliasing=uncorrected-and-choice-outputs-forbidden" +
    "|numerical=f32-sigmoid-then-f32-correction-add" +
    "|qualification=correctness-source-only-expf-parity-unqualified-non-bindable" +
    "|excluded=group-scoring-topk-masking-normalization-scaling-expert-ffn" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|geometry=" +
    expert_count.to_string() +
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
    family,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    expert_count,
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
