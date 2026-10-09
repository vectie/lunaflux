///|
fn hyper_control_common_checks(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  maximum_tokens : Int,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  limits : GlmHybridCudaAotLimits,
) -> Unit raise GlmHybridCudaAotError {
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
}

///|
fn hyper_control_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal : Int,
  family : GlmHybridHyperControlAotFamily,
  numerical_contract : @decoder_hyper_connection_cuda_source.DecoderHyperConnectionNumericalContract,
  stream_count : Int,
  hidden_width : Int,
  sinkhorn_iterations : Int,
  control_epsilon : Double,
  normalization_epsilon : Double,
  maximum_tokens : Int,
  target : @catalog.DeviceTarget,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : Array[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand],
  compiler : @luna_specialization.LunaCompilerPolicy,
  function_symbol : String,
  source : Bytes,
  limits : GlmHybridCudaAotLimits,
) -> GlmHybridHyperControlCudaCandidate raise GlmHybridCudaAotError {
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let (family_tag, stage) = match family {
    FlashFunctionAndCollapse4x4096 =>
      (
        "flash-function-collapse-4x4096", "function=unweighted-rmsnorm-f32-bf16-weight-linear-pre-post-row-softmax-collapse-bf16",
      )
    FlashPositiveSinkhorn4x20 =>
      (
        "flash-positive-sinkhorn-4x20", "sinkhorn=initial-column-then-nineteen-row-column-positive-f32",
      )
  }
  let recipe = @utf8.encode(
    "lunaflux.glm-hybrid-hyper-control-cuda-aot.v1" +
    "|family=" +
    family_tag +
    "|" +
    stage +
    "|layout=token-stream-hidden-and-destination-source" +
    "|alias=all-input-and-output-buffers-distinct" +
    "|rounding=ordered-f32-controls-one-bf16-collapse-round" +
    "|qualification=serial-correctness-source-only-non-bindable" +
    "|excluded=sublayer-residual-mix-materialization-upload-compile-launch" +
    "|geometry=" +
    stream_count.to_string() +
    ":" +
    hidden_width.to_string() +
    ":" +
    sinkhorn_iterations.to_string() +
    ":" +
    control_epsilon.to_string() +
    ":" +
    normalization_epsilon.to_string() +
    ":" +
    maximum_tokens.to_string() +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  let requirement = requirements.requirements()[requirement_ordinal]
  {
    family,
    numerical_contract,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    stream_count,
    hidden_width,
    sinkhorn_iterations,
    control_epsilon,
    normalization_epsilon,
    maximum_tokens,
    target,
    dimensions,
    operands: ReadOnlyArray::from_array(operands),
    compiler,
    function_symbol,
    source,
    source_digest,
    recipe,
    recipe_digest: GlmHybridCudaRecipeDigest(
      @crypto.bytes_to_hex_string(@crypto.sha256(recipe)),
    ),
  }
}

///|
pub fn lower_hyper_connection_function_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridHyperControlCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      HyperConnectionFunction(
        stream_count,
        hidden_width,
        control_epsilon,
        normalization_epsilon
      ),
      HyperConnectionFunctionAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if stream_count != 4 ||
    hidden_width != 4096 ||
    hidden_width != requirements.geometry().hidden_size() ||
    control_epsilon != 0.000001 ||
    normalization_epsilon != 0.00001 {
    raise Invalid(Shape)
  }
  hyper_control_common_checks(
    requirements, maximum_tokens, target, compiler, limits,
  )
  let rows = maximum_tokens.to_int64()
  let flattened = checked_mul(stream_count.to_int64(), hidden_width.to_int64())
  let mix_width = checked_mul(
    (2 + stream_count).to_int64(),
    stream_count.to_int64(),
  )
  let stream_bytes = checked_mul(checked_mul(rows, flattened), 2L)
  let function_bytes = checked_mul(checked_mul(mix_width, flattened), 2L)
  let post_bytes = checked_mul(checked_mul(rows, stream_count.to_int64()), 4L)
  let matrix_bytes = checked_mul(
    checked_mul(
      checked_mul(rows, stream_count.to_int64()),
      stream_count.to_int64(),
    ),
    4L,
  )
  let collapsed_bytes = checked_mul(
    checked_mul(rows, hidden_width.to_int64()),
    2L,
  )
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(HyperResidualStreamsInput, stream_bytes, 16L),
    dense_projection_operand(HyperFunctionWeightsInput, function_bytes, 16L),
    dense_projection_operand(HyperBaseInput, mix_width * 4L, 16L),
    dense_projection_operand(HyperScaleInput, 12L, 16L),
    dense_projection_operand(HyperPostControlInput, post_bytes, 16L),
    dense_projection_operand(HyperInitialCombinationOutput, matrix_bytes, 16L),
    dense_projection_operand(HyperCollapsedOutput, collapsed_bytes, 16L),
  ]
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let rendered = @decoder_hyper_connection_cuda_source.staged_function_and_collapse_bf16_f32_source(
    stream_width=stream_count,
    hidden_width~,
    maximum_rows=maximum_tokens,
    control_epsilon~,
    normalization_epsilon~,
    function_symbol="lunaflux_glm_hybrid_hc_function_bf16_f32_v1",
  ) catch {
    error => raise HyperConnectionControlSource(error)
  }
  hyper_control_candidate(
    requirements,
    requirement_ordinal,
    FlashFunctionAndCollapse4x4096,
    rendered.numerical_contract(),
    stream_count,
    hidden_width,
    0,
    control_epsilon,
    normalization_epsilon,
    maximum_tokens,
    target,
    dimensions,
    operands,
    compiler,
    rendered.function_symbol(),
    rendered.source_bytes(),
    limits,
  )
}

///|
pub fn lower_hyper_connection_sinkhorn_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridHyperControlCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      HyperConnectionSinkhorn(stream_count, iterations, control_epsilon),
      Advanced(HyperConnectionSinkhornAot),
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if stream_count != 4 || iterations != 20 || control_epsilon != 0.000001 {
    raise Invalid(Shape)
  }
  hyper_control_common_checks(
    requirements, maximum_tokens, target, compiler, limits,
  )
  let matrix_bytes = checked_mul(
    checked_mul(
      checked_mul(maximum_tokens.to_int64(), stream_count.to_int64()),
      stream_count.to_int64(),
    ),
    4L,
  )
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(HyperInitialCombinationOutput, matrix_bytes, 16L),
    dense_projection_operand(HyperCombineControlInput, matrix_bytes, 16L),
  ]
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=maximum_tokens,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let rendered = @decoder_hyper_connection_cuda_source.staged_positive_sinkhorn_f32_source(
    stream_width=stream_count,
    maximum_rows=maximum_tokens,
    sinkhorn_iterations=iterations,
    control_epsilon~,
    function_symbol="lunaflux_glm_hybrid_hc_sinkhorn_f32_v1",
  ) catch {
    error => raise HyperConnectionControlSource(error)
  }
  hyper_control_candidate(
    requirements,
    requirement_ordinal,
    FlashPositiveSinkhorn4x20,
    rendered.numerical_contract(),
    stream_count,
    requirements.geometry().hidden_size(),
    iterations,
    control_epsilon,
    0.0,
    maximum_tokens,
    target,
    dimensions,
    operands,
    compiler,
    rendered.function_symbol(),
    rendered.source_bytes(),
    limits,
  )
}
