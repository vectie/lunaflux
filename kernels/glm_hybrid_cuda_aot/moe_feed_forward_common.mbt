///|
fn moe_feed_forward_selected_width(
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  expert_count : Int,
) -> Int? {
  for requirement in requirements.requirements() {
    if requirement.operation()
      is MoeGroupedNoAuxiliaryLossTopK(
        routed_experts,
        _,
        _,
        _,
        experts_per_token,
        _,
        _
      ) &&
      routed_experts == expert_count {
      return Some(experts_per_token)
    }
  }
  None
}

///|
fn validate_moe_feed_forward_environment(
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
fn moe_feed_forward_dimensions(
  maximum_tokens : Int,
  hidden_size : Int,
) -> @launch_contract.AotLaunchDimensions raise GlmHybridCudaAotError {
  let cells = checked_mul(maximum_tokens.to_int64(), hidden_size.to_int64())
  let grid = (cells + 255L) / 256L
  if grid <= 0L || grid > 2147483647L {
    raise SizeOverflow
  }
  @launch_contract.AotLaunchDimensions::new(
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
}

///|
fn moe_feed_forward_candidate(
  family : GlmHybridMoeFeedForwardAotFamily,
  numerical_contract : GlmHybridMoeFeedForwardNumericalContract,
  requirements : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirements,
  requirement : @glm_hybrid_kernel_requirements.GlmHybridKernelRequirement,
  requirement_ordinal : Int,
  maximum_tokens : Int,
  target : @catalog.DeviceTarget,
  dimensions : @launch_contract.AotLaunchDimensions,
  operands : Array[@glm_hybrid_artifact_admission.GlmHybridLaunchOperand],
  compiler : @luna_specialization.LunaCompilerPolicy,
  function_symbol : String,
  source : Bytes,
  stage : String,
  geometry : String,
  rounding : String,
  excluded : String,
  limits : GlmHybridCudaAotLimits,
) -> GlmHybridMoeFeedForwardCudaCandidate raise GlmHybridCudaAotError {
  if source.length() > limits.maximum_source_bytes {
    raise Invalid(Encoding)
  }
  let source_digest = GlmHybridCudaSourceDigest(
    @crypto.bytes_to_hex_string(@crypto.sha256(source)),
  )
  let compiler_identity = compiler.identity()
  let recipe = @utf8.encode(
    "lunaflux.glm-hybrid-moe-feed-forward-cuda-aot.v1" +
    "|stage=" +
    stage +
    "|geometry=" +
    geometry +
    "|rounding=" +
    rounding +
    "|qualification=serial-correctness-source-only-non-bindable" +
    "|excluded=" +
    excluded +
    "|content=" +
    requirements.source_identity().content().as_hex() +
    "|plan=" +
    requirements.source_identity().plan().as_hex() +
    "|requirements=" +
    requirements.digest().as_hex() +
    "|ordinal=" +
    requirement_ordinal.to_string() +
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
    numerical_contract,
    source_identity: requirements.source_identity(),
    requirements_digest: requirements.digest(),
    requirement_ordinal,
    operation: requirement.operation(),
    capability: requirement.capability(),
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
