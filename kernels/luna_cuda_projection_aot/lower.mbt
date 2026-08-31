///|
fn projection_validate_policy(
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
) -> Unit raise ProjectionAotError {
  if !target.supports_bf16() {
    raise Unsupported(DeviceTarget)
  }
  if compiler.reassociate() {
    raise Unsupported(CompilerPolicy)
  }
}

///|
fn projection_finish_candidate(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  symbol : String,
  embed_greedy_sampling : Bool,
  limits : ProjectionAotLimits,
) -> ProjectionCudaLowering raise ProjectionAotError {
  projection_validate_policy(target, compiler)
  let parameters = projection_bind_qkv_widths(
    operation,
    projection_parameters(operation),
    operands,
  )
  validate_projection_operands(operation, parameters, profile, operands)
  let dispatch_plan = projection_dispatch_plan(
    parameters,
    profile.max_query_tokens(),
  )
  let dimensions = projection_dimensions(profile, parameters)
  let source = projection_source(
    symbol,
    parameters,
    profile.max_query_tokens(),
    profile.max_query_rows(),
    dispatch_plan,
    embed_greedy_sampling,
    dimensions.shared_memory_bytes() > 0,
  )
  if source.length() > limits.max_source_bytes {
    raise LimitExceeded(Source)
  }
  let source_digest = ProjectionCudaSourceDigest(projection_sha256(source))
  let numerical_contract = projection_numerical_contract(
    parameters, dispatch_plan,
  )
  let recipe = projection_recipe(
    parameters,
    operation.id(),
    entry_point_id,
    target,
    compiler,
    profile,
    operands,
    dimensions,
    numerical_contract,
    dispatch_plan,
    symbol,
    source_digest,
    embed_greedy_sampling,
  )
  if recipe.length() > limits.max_recipe_bytes {
    raise LimitExceeded(Recipe)
  }
  {
    family: parameters.family,
    operation_id: operation.id(),
    entry_point_id,
    target,
    profile,
    input_width: parameters.input_width,
    output_width: parameters.output_width,
    intermediate_width: parameters.intermediate_width,
    dispatch_plan,
    operands: ReadOnlyArray::from_array(operands),
    dimensions,
    numerical_contract,
    compiler,
    function_symbol: symbol,
    source,
    source_digest,
    recipe,
    recipe_digest: ProjectionCudaRecipeDigest(projection_sha256(recipe)),
  }
}

///|
/// Lowers one non-bindable BF16 QKV, dense output, gated-MLP, or final
/// language-model-head candidate into deterministic reference CUDA source.
///
/// The ABI is the exact canonical paged-graph operand order. No source
/// fragment, compiler flag, path, or symbol text is caller controlled.
/// Compilation remains an isolated offline release operation.
pub fn lower_projection_cuda_aot_candidate(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  limits : ProjectionAotLimits,
) -> ProjectionCudaLowering raise ProjectionAotError {
  let parameters = projection_parameters(operation)
  projection_finish_candidate(
    operation,
    profile,
    entry_point_id,
    target,
    compiler,
    operands,
    projection_symbol(parameters.family, operation.id(), entry_point_id),
    false,
    limits,
  )
}

///|
/// Emits an operation-neutral one-entry release candidate. Same-shape decoder
/// operations compile to one reusable module while recipes and final binders
/// continue to authenticate their distinct operation/operand identities.
pub fn lower_reusable_projection_cuda_aot_candidate(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  limits : ProjectionAotLimits,
) -> ProjectionCudaLowering raise ProjectionAotError {
  let parameters = projection_parameters(operation)
  let entry_point_id = @catalog.AotKernelEntryPointId::from_stable_int(1) catch {
    _ => raise Invalid(Operation)
  }
  let symbol = "lunaflux_luna_" +
    projection_family_tag(parameters.family) +
    "_bf16_release_v1"
  projection_finish_candidate(
    operation,
    profile,
    entry_point_id,
    target,
    compiler,
    operands,
    symbol,
    parameters.family == LanguageModelHead,
    limits,
  )
}
