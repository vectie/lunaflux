///|
pub fn lower_moe_router_projection_cuda_aot_candidate(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement_ordinal~ : Int,
  maximum_tokens~ : Int,
  target~ : @catalog.DeviceTarget,
  compiler~ : @luna_specialization.LunaCompilerPolicy,
  limits~ : GlmHybridCudaAotLimits,
) -> GlmHybridMoeRouterCudaCandidate raise GlmHybridCudaAotError {
  guard requirements.requirements().get(requirement_ordinal)
    is Some(requirement) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  guard (requirement.operation(), requirement.capability())
    is (
      MoeRouterProjection(hidden_width, expert_count, true),
      MoeRouterProjectionAot,
    ) else {
    raise UnsupportedRequirement(requirement_ordinal)
  }
  let family = match (hidden_width, expert_count) {
    (6144, 256) => FullFloat32Router256
    (4096, 288) => FlashFloat32Router288
    _ => raise Invalid(Shape)
  }
  if hidden_width != requirements.geometry().hidden_size() {
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
  let input_cells = checked_mul(
    maximum_tokens.to_int64(),
    hidden_width.to_int64(),
  )
  let output_cells = checked_mul(
    maximum_tokens.to_int64(),
    expert_count.to_int64(),
  )
  let weight_cells = checked_mul(
    hidden_width.to_int64(),
    expert_count.to_int64(),
  )
  let grid = (output_cells + 255L) / 256L
  if grid <= 0L || grid > 2147483647L {
    raise SizeOverflow
  }
  let operands = [
    dense_projection_operand(StepCountsInput, 20L, 4L),
    dense_projection_operand(
      MoeRouterFloat32Input,
      checked_mul(input_cells, 4L),
      16L,
    ),
    dense_projection_operand(
      MoeRouterBf16Weights,
      checked_mul(weight_cells, 2L),
      16L,
    ),
    dense_projection_operand(
      MoeRouterFloat32LogitsOutput,
      checked_mul(output_cells, 4L),
      16L,
    ),
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
  let rendered = @router_projection_cuda_source.f32_input_bf16_weight_projection_source(
    input_width=hidden_width,
    output_width=expert_count,
    maximum_rows=maximum_tokens,
    function_symbol="lunaflux_glm_hybrid_moe_router_f32_bf16_v1",
  ) catch {
    error => raise RouterProjectionSource(error)
  }
  let source = rendered.source_bytes()
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let compiler_identity = compiler.identity()
  let family_tag = match family {
    FullFloat32Router256 => "full-hidden6144-experts256"
    FlashFloat32Router288 => "flash-hidden4096-experts288"
  }
  let recipe = @utf8.encode(
    "lunaflux.glm-hybrid-moe-router-projection-cuda-aot.v1" +
    "|family=" +
    family_tag +
    "|input=f32-activation-view|weights=row-major-bf16-expert-hidden-no-bias" +
    "|numerical=ordered-f32-accumulation-f32-output" +
    "|qualification=correctness-source-only-non-bindable" +
    "|excluded=sigmoid-correction-bias-group-topk-expert-ffn-residual" +
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
    hidden_width,
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
