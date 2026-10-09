///|
pub fn lower_kda_gated_rms_norm_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridKdaGatedNormCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (KdaGatedRmsNorm(head_count, head_dimension, epsilon), KdaGatedRmsNormAot
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if head_count != 64 || head_dimension != 128 || epsilon != 0.00001 {
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
  let vector_width = checked_mul(
    head_count.to_int64(),
    head_dimension.to_int64(),
  )
  let vector_bytes = checked_mul(
    checked_mul(maximum_tokens.to_int64(), vector_width),
    2L,
  )
  let weight_bytes = checked_mul(head_dimension.to_int64(), 2L)
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(KdaGatedNormRecurrentInput, vector_bytes, 16L),
    dense_projection_operand(KdaGatedNormGateInput, vector_bytes, 16L),
    dense_projection_operand(KdaGatedNormWeights, weight_bytes, 16L),
    dense_projection_operand(KdaGatedNormOutput, vector_bytes, 16L),
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
  let rendered = @gated_rms_norm_cuda_source.sigmoid_gated_rms_norm_bf16_source(
    head_count~,
    head_dimension~,
    maximum_rows=maximum_tokens,
    epsilon~,
    function_symbol="lunaflux_glm_hybrid_kda_gated_rms_norm_bf16_v1",
  ) catch {
    error => raise GatedRmsNormSource(error)
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
    "lunaflux.glm-hybrid-kda-gated-rms-norm-cuda-aot.v1" +
    "|family=serial-per-head-sigmoid-gated-rmsnorm-bf16" +
    "|layout=row-head-dimension-contiguous" +
    "|order=ordered-f32-rms-reduction-weight-then-sigmoid-gate" +
    "|round=bf16-input-f32-compute-single-bf16-output" +
    "|qualification=serial-correctness-source-only-non-bindable" +
    "|excluded=output-projection-residual-cache" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|geometry=" +
    head_count.to_string() +
    ":" +
    head_dimension.to_string() +
    ":" +
    epsilon.to_string() +
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
    family: SerialPerHeadSigmoidGatedRmsNormBf16,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    head_count,
    head_dimension,
    epsilon,
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
