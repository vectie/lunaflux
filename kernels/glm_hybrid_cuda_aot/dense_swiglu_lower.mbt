///|
fn dense_swiglu_recipe(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  ordinal : Int,
  hidden_size : Int,
  intermediate_size : Int,
  maximum_tokens : Int,
  source_digest : GlmHybridCudaSourceDigest,
) -> Bytes {
  @utf8.encode(
    "lunaflux.glm-hybrid-dense-swiglu-cuda-aot.v1" +
    "|stage=bias-free-gate-silu-times-up-then-down" +
    "|weights=separate-row-major-gate-up-down" +
    "|gate-up=ordered-f32-accumulation-then-bf16-round" +
    "|silu=bf16-input-expf-then-bf16-round" +
    "|product=bf16-round|down=ordered-f32-accumulation-then-bf16-round" +
    "|excluded=residual-norm-routing-and-shared-experts" +
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
    hidden_size.to_string() +
    ":" +
    intermediate_size.to_string() +
    ":" +
    maximum_tokens.to_string() +
    "|source=" +
    source_digest.as_hex(),
  )
}

///|
pub fn lower_dense_swiglu_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridDenseSwiGluCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (DenseSwiGlu(hidden_size, intermediate_size), DenseSwiGluAot) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  if hidden_size != requirements.geometry().hidden_size() ||
    intermediate_size != 12288 ||
    !(hidden_size == 6144 || hidden_size == 4096) {
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
  let row_elements = checked_mul(
    maximum_tokens.to_int64(),
    hidden_size.to_int64(),
  )
  let weight_elements = checked_mul(
    hidden_size.to_int64(),
    intermediate_size.to_int64(),
  )
  if row_elements > 9223372036854775552L {
    raise SizeOverflow
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=((row_elements + 255L) / 256L).to_int(),
    grid_y=1,
    grid_z=1,
    block_x=256,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    error => raise Launch(error)
  }
  let row_bytes = checked_mul(row_elements, 2L)
  let weight_bytes = checked_mul(weight_elements, 2L)
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(HiddenInput, row_bytes, 16L),
    dense_projection_operand(DenseGateWeights, weight_bytes, 16L),
    dense_projection_operand(DenseUpWeights, weight_bytes, 16L),
    dense_projection_operand(DenseDownWeights, weight_bytes, 16L),
    dense_projection_operand(HiddenOutput, row_bytes, 16L),
  ]
  let rendered = @gated_mlp_cuda_source.separate_gate_up_swiglu_dense_bf16_source(
    hidden_size~,
    intermediate_size~,
    maximum_rows=maximum_tokens,
    function_symbol="lunaflux_glm_hybrid_dense_swiglu_bf16_v1",
  ) catch {
    error => raise GatedMlpSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let recipe = dense_swiglu_recipe(
    requirements, requirement_ordinal, hidden_size, intermediate_size, maximum_tokens,
    source_digest,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise Invalid(Encoding)
  }
  {
    family: SeparateGateUpDownWeightsBf16,
    numerical_contract: rendered.numerical_contract(),
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
    hidden_size,
    intermediate_size,
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
