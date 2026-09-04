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
  runtime_dispatch : @luna_projection_strategy.ProjectionRuntimeDispatch,
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
  let dispatch_plan = projection_validate_runtime_dispatch(
    runtime_dispatch,
    parameters,
    profile.max_query_tokens(),
    target,
  )
  let demand : @luna_projection_tile_compiler.ProjectionResultDemand = if parameters.family ==
    LanguageModelHead {
    QueryRowEnds
  } else {
    AllTokenRows
  }
  let tile_request = @luna_projection_tile_compiler.ProjectionTileCompilerRequest::new(
    dispatch=dispatch_plan,
    maximum_rows=profile.max_query_rows(),
    maximum_tokens=profile.max_query_tokens(),
    demand~,
  ) catch {
    _ => raise Invalid(CompilerPolicy)
  }
  let tile_compilation = @luna_projection_tile_compiler.compile_projection_tile(
    tile_request,
  )
  validate_projection_operands(
    operation, parameters, profile, operands, dispatch_plan,
  )
  let dimensions = projection_dimensions(profile, parameters, dispatch_plan)
  let source = projection_source(
    symbol,
    parameters,
    profile.max_query_tokens(),
    profile.max_query_rows(),
    dispatch_plan,
    tile_compilation,
    MixedPrefillDecodeProjection,
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
    tile_compilation,
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
    tile_compilation,
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
/// Builds CUDA's startup dispatch table from backend-neutral offline records.
/// Device-specific capability facts are adapted here; record validation,
/// fallback policy, and deterministic selection remain in the strategy layer.
pub fn prepare_projection_cuda_runtime_dispatch(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  operands : ArrayView[@launch_contract.AotOperand],
  records : ArrayView[@luna_projection_strategy.ProjectionAutotuneRecord],
) -> @luna_projection_strategy.ProjectionRuntimeDispatch raise ProjectionAotError {
  if !target.supports_bf16() {
    raise Unsupported(DeviceTarget)
  }
  let parameters = projection_bind_qkv_widths(
    operation,
    projection_parameters(operation),
    operands,
  )
  projection_build_runtime_dispatch(
    parameters,
    profile.max_query_tokens(),
    target,
    records,
  )
}

///|
/// Lowers a candidate from an already materialized startup strategy table.
pub fn lower_projection_cuda_aot_candidate_with_dispatch(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  runtime_dispatch : @luna_projection_strategy.ProjectionRuntimeDispatch,
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
    runtime_dispatch,
    projection_symbol(parameters.family, operation.id(), entry_point_id),
    false,
    limits,
  )
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
  let records : Array[@luna_projection_strategy.ProjectionAutotuneRecord] = []
  let runtime_dispatch = prepare_projection_cuda_runtime_dispatch(
    operation, profile, target, operands, records,
  )
  lower_projection_cuda_aot_candidate_with_dispatch(
    operation, profile, entry_point_id, target, compiler, operands, runtime_dispatch,
    limits,
  )
}

///|
/// Reusable lowering variant consuming an already materialized strategy table.
pub fn lower_reusable_projection_cuda_aot_candidate_with_dispatch(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  runtime_dispatch : @luna_projection_strategy.ProjectionRuntimeDispatch,
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
    runtime_dispatch,
    symbol,
    parameters.family == LanguageModelHead,
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
  let records : Array[@luna_projection_strategy.ProjectionAutotuneRecord] = []
  let runtime_dispatch = prepare_projection_cuda_runtime_dispatch(
    operation, profile, target, operands, records,
  )
  lower_reusable_projection_cuda_aot_candidate_with_dispatch(
    operation, profile, target, compiler, operands, runtime_dispatch, limits,
  )
}
