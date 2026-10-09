///|
fn kda_recurrent_recipe(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  ordinal : Int,
  head_count : Int,
  head_dimension : Int,
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
    "lunaflux.glm-hybrid-kda-recurrent-cuda-aot.v1" +
    "|family=serial-normalized-recurrent-delta-bf16-f32" +
    "|layout=q-k-token-head-key-bf16:value-output-token-head-value-bf16" +
    "|decay=token-head-key-f32-log-domain-range-minus5-to-zero" +
    "|beta=token-head-bf16-range-zero-to-one" +
    "|state=sequence-head-key-value-f32-explicit-in-out" +
    "|sequences=counts2-csr-offsets|live-rows=counts3" +
    "|qk-normalization=f32-sqrt-sum-squares-plus-1e-6" +
    "|query-scale=one-over-sqrt-key-dimension" +
    "|transition=decay-state-read-delta-rank1-update-state-read-output" +
    "|numerical=ordered-rn-f32-exp-sqrt-one-bf16-output-round" +
    "|runtime-validation=finite-input-nonpositive-decay-unit-beta-before-write" +
    "|qualification=serial-correctness-source-only-non-bindable" +
    "|excluded=projection-decay-control-residual-norm-output-projection-cache-ownership" +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    ordinal.to_string() +
    "|geometry=" +
    head_count.to_string() +
    ":" +
    head_dimension.to_string() +
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
/// Lowers the official GLM-5.3-Flash recurrent KDA transition. The emitted
/// source is intentionally serial correctness evidence; it is not a throughput
/// kernel and owns neither gate projection nor persistent cache state.
pub fn lower_kda_recurrent_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  maximum_sequences~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridKdaRecurrentCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      KdaRecurrentUpdate(head_count, head_dimension, gate_lower_bound),
      KdaRecurrentUpdateAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if head_count != 64 || head_dimension != 128 || gate_lower_bound != -5.0 {
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
  let row_width = checked_mul(head_count.to_int64(), head_dimension.to_int64())
  let vector_cells = checked_mul(maximum_tokens.to_int64(), row_width)
  let gate_cells = checked_mul(maximum_tokens.to_int64(), head_count.to_int64())
  let state_width = checked_mul(row_width, head_dimension.to_int64())
  let state_cells = checked_mul(maximum_sequences.to_int64(), state_width)
  let vector_bytes = checked_mul(vector_cells, 2L)
  let decay_bytes = checked_mul(vector_cells, 4L)
  let gate_bytes = checked_mul(gate_cells, 2L)
  let state_bytes = checked_mul(state_cells, 4L)
  let offset_bytes = checked_mul(maximum_sequences.to_int64() + 1L, 4L)
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(KdaSequenceOffsetsInput, offset_bytes, 4L),
    dense_projection_operand(KdaRecurrentQueryInput, vector_bytes, 16L),
    dense_projection_operand(KdaRecurrentKeyInput, vector_bytes, 16L),
    dense_projection_operand(KdaRecurrentValueInput, vector_bytes, 16L),
    dense_projection_operand(KdaLogDecayInput, decay_bytes, 16L),
    dense_projection_operand(KdaInputGateInput, gate_bytes, 16L),
    dense_projection_operand(KdaRecurrentInitialStateInput, state_bytes, 16L),
    dense_projection_operand(KdaRecurrentOutput, vector_bytes, 16L),
    dense_projection_operand(KdaRecurrentFinalStateOutput, state_bytes, 16L),
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
  let normalization_epsilon = 0.000001
  let rendered = @recurrent_delta_cuda_source.causal_normalized_recurrent_delta_bf16_f32_source(
    head_count~,
    key_dimension=head_dimension,
    value_dimension=head_dimension,
    normalization_epsilon~,
    log_decay_lower_bound=gate_lower_bound,
    maximum_rows=maximum_tokens,
    maximum_sequences~,
    function_symbol="lunaflux_glm_hybrid_kda_recurrent_delta_bf16_f32_v1",
  ) catch {
    error => raise RecurrentDeltaSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = kda_recurrent_recipe(
    requirements, requirement_ordinal, head_count, head_dimension, maximum_tokens,
    maximum_sequences, target, compiler, dimensions, operands, source_digest,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: SerialNormalizedRecurrentDeltaBf16F32,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    head_count,
    head_dimension,
    normalization_epsilon,
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
