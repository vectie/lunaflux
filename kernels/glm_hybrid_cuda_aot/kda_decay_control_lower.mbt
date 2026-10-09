///|
pub fn lower_kda_decay_control_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridKdaDecayControlCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      KdaDecayControl(head_count, head_dimension, lower_bound),
      KdaDecayControlAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if head_count != 64 || head_dimension != 128 || lower_bound != -5.0 {
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
  let row_width = checked_mul(head_count.to_int64(), head_dimension.to_int64())
  let cells = checked_mul(maximum_tokens.to_int64(), row_width)
  let grid = (cells + 255L) / 256L
  if grid <= 0L || grid > 2147483647L {
    raise SizeOverflow
  }
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(
      KdaDecayRawForgetInput,
      checked_mul(cells, 2L),
      16L,
    ),
    dense_projection_operand(KdaDecayBiasInput, checked_mul(row_width, 4L), 16L),
    dense_projection_operand(
      KdaDecayLogRateInput,
      head_count.to_int64() * 4L,
      16L,
    ),
    dense_projection_operand(KdaDecayOutput, checked_mul(cells, 4L), 16L),
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
  let rendered = @decay_control_cuda_source.safe_lower_bound_sigmoid_decay_f32_source(
    head_count~,
    head_dimension~,
    lower_bound~,
    maximum_rows=maximum_tokens,
    function_symbol="lunaflux_glm_hybrid_kda_decay_control_f32_v1",
  ) catch {
    error => raise DecayControlSource(error)
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
    "lunaflux.glm-hybrid-kda-decay-control-cuda-aot.v1" +
    "|family=pointwise-safe-lower-bound-sigmoid-f32" +
    "|layout=token-head-component" +
    "|order=bf16-raw-plus-f32-dt-bias-exp-f32-a-log-rate-multiply-sigmoid-lower-bound" +
    "|output=f32-log-decay" +
    "|qualification=correctness-source-only-non-bindable" +
    "|excluded=qkv-beta-projection-recurrence-cache" +
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
    lower_bound.to_string() +
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
    family: PointwiseSafeLowerBoundSigmoidF32,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    head_count,
    head_dimension,
    lower_bound,
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
