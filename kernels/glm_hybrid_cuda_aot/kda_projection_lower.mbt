///|
fn kda_projection_recipe(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  ordinal : Int,
  hidden_width : Int,
  head_count : Int,
  head_dimension : Int,
  maximum_tokens : Int,
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
    "lunaflux.glm-hybrid-kda-projection-cuda-aot.v1" +
    "|family=serial-qkv-forget-and-input-gate-bf16" +
    "|hidden=token-hidden-bf16" +
    "|qkv=three-independent-output-major-bf16-weights" +
    "|forget=hidden-to-control-rank-bf16-round-then-control-rank-to-head-channel-bf16-round" +
    "|input-gate=hidden-to-head-bf16-logit-round-f32-sigmoid-bf16-round" +
    "|outputs=separate-q-k-v-forget-low-rank-raw-forget-beta" +
    "|numerical=ordered-rn-f32-dense-accumulation-expf-one-round-per-declared-bf16-stage" +
    "|qualification=serial-correctness-source-only-non-bindable" +
    "|excluded=f32-decay-control-short-convolution-recurrent-update-output-projection-cache-ownership" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
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
/// Lowers the official Flash BF16 Q/K/V, two-stage forget, and sigmoid beta
/// projections. The adjacent F32 decay-control requirement remains separate.
pub fn lower_kda_projection_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridKdaProjectionCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (KdaProjection(head_count, head_dimension), KdaProjectionAot) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let hidden_width = requirements.geometry().hidden_size()
  let control_rank = head_dimension
  if hidden_width != 4096 || head_count != 64 || head_dimension != 128 {
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
  let projected_width = checked_mul(
    head_count.to_int64(),
    head_dimension.to_int64(),
  )
  let hidden_cells = checked_mul(
    maximum_tokens.to_int64(),
    hidden_width.to_int64(),
  )
  let projected_cells = checked_mul(maximum_tokens.to_int64(), projected_width)
  let low_cells = checked_mul(
    maximum_tokens.to_int64(),
    control_rank.to_int64(),
  )
  let gate_cells = checked_mul(maximum_tokens.to_int64(), head_count.to_int64())
  let qkv_weight_cells = checked_mul(projected_width, hidden_width.to_int64())
  let forget_a_cells = checked_mul(
    control_rank.to_int64(),
    hidden_width.to_int64(),
  )
  let forget_b_cells = checked_mul(projected_width, control_rank.to_int64())
  let gate_weight_cells = checked_mul(
    head_count.to_int64(),
    hidden_width.to_int64(),
  )
  let hidden_bytes = checked_mul(hidden_cells, 2L)
  let projected_bytes = checked_mul(projected_cells, 2L)
  let low_bytes = checked_mul(low_cells, 2L)
  let beta_bytes = checked_mul(gate_cells, 2L)
  let qkv_weight_bytes = checked_mul(qkv_weight_cells, 2L)
  let forget_a_bytes = checked_mul(forget_a_cells, 2L)
  let forget_b_bytes = checked_mul(forget_b_cells, 2L)
  let gate_weight_bytes = checked_mul(gate_weight_cells, 2L)
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(HiddenInput, hidden_bytes, 16L),
    dense_projection_operand(KdaQueryProjectionWeights, qkv_weight_bytes, 16L),
    dense_projection_operand(KdaKeyProjectionWeights, qkv_weight_bytes, 16L),
    dense_projection_operand(KdaValueProjectionWeights, qkv_weight_bytes, 16L),
    dense_projection_operand(KdaForgetLowRankWeights, forget_a_bytes, 16L),
    dense_projection_operand(KdaForgetExpansionWeights, forget_b_bytes, 16L),
    dense_projection_operand(KdaInputGateWeights, gate_weight_bytes, 16L),
    dense_projection_operand(KdaProjectedQueryOutput, projected_bytes, 16L),
    dense_projection_operand(KdaProjectedKeyOutput, projected_bytes, 16L),
    dense_projection_operand(KdaProjectedValueOutput, projected_bytes, 16L),
    dense_projection_operand(KdaForgetLowRankOutput, low_bytes, 16L),
    dense_projection_operand(KdaRawForgetOutput, projected_bytes, 16L),
    dense_projection_operand(KdaBetaOutput, beta_bytes, 16L),
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
  let rendered = @recurrent_attention_projection_cuda_source.qkv_forget_input_gate_bf16_source(
    hidden_width~,
    head_count~,
    head_dimension~,
    control_rank~,
    maximum_rows=maximum_tokens,
    function_symbol="lunaflux_glm_hybrid_kda_projection_bf16_v1",
  ) catch {
    error => raise RecurrentAttentionProjectionSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = kda_projection_recipe(
    requirements, requirement_ordinal, hidden_width, head_count, head_dimension,
    maximum_tokens, target, compiler, dimensions, operands, source_digest,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: SerialQkvForgetAndInputGateBf16,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    hidden_width,
    head_count,
    head_dimension,
    control_rank,
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
