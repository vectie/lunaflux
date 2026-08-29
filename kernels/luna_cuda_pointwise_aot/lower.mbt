///|
fn validate_offline_policy(
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
) -> Unit raise PointwiseAotError {
  if !target.supports_bf16() {
    raise Unsupported(DeviceTarget)
  }
  if compiler.reassociate() {
    raise Unsupported(CompilerPolicy)
  }
}

///|
fn finish_lowering(
  family : PointwiseAotFamily,
  operation : @plan.PlanOperation,
  entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  profile : @launch_contract.PagedKernelProfile,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  function_symbol : String,
  source : Bytes,
  limits : PointwiseAotLimits,
) -> PointwiseCudaLowering raise PointwiseAotError {
  if source.length() > limits.max_source_bytes {
    raise LimitExceeded(Source)
  }
  let source_digest = PointwiseSourceDigest(pointwise_sha256(source))
  let numerical_contract = pointwise_contract(family)
  let recipe = pointwise_recipe(
    family,
    operation.id(),
    entry_point_id,
    target,
    compiler,
    operands,
    dimensions,
    numerical_contract,
    false,
    function_symbol,
    source_digest,
  )
  if recipe.length() > limits.max_recipe_bytes {
    raise LimitExceeded(Recipe)
  }
  {
    family,
    operation_id: operation.id(),
    entry_point_id,
    target,
    profile,
    operands: ReadOnlyArray::from_array(operands),
    dimensions,
    numerical_contract,
    manifest_bindable: false,
    function_symbol,
    source,
    source_digest,
    recipe,
    recipe_digest: PointwiseRecipeDigest(pointwise_sha256(recipe)),
  }
}

///|
/// Lowers one non-bindable BF16 embedding, RMSNorm, positioned-RoPE, or
/// residual-add candidate. The supplied operands must match the catalog-v3
/// positional role and byte-count vocabulary; `bind_manifest_pointwise_cuda_aot`
/// later authenticates them against the exact admitted contract and module.
pub fn lower_pointwise_cuda_aot_candidate(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  limits : PointwiseAotLimits,
) -> PointwiseCudaLowering raise PointwiseAotError {
  validate_offline_policy(target, compiler)
  let parameters = pointwise_parameters(operation)
  validate_operands(operation, parameters, profile, operands)
  let work_items = match parameters.family {
    RmsNorm => profile.max_query_tokens().to_int64()
    QkRmsNorm =>
      checked_mul(
        profile.max_query_tokens().to_int64(),
        (parameters.query_heads + 2 * parameters.kv_heads).to_int64(),
      )
    _ =>
      checked_mul(
        profile.max_query_tokens().to_int64(),
        parameters.width.to_int64(),
      )
  }
  let dimensions = pointwise_dimensions(parameters.family, work_items)
  let symbol = pointwise_symbol(
    parameters.family,
    operation.id(),
    entry_point_id,
  )
  let source = pointwise_source(symbol, parameters)
  finish_lowering(
    parameters.family,
    operation,
    entry_point_id,
    target,
    profile,
    compiler,
    operands,
    dimensions,
    symbol,
    source,
    limits,
  )
}

///|
/// Emits the release producer's operation-neutral one-entry module candidate.
/// Equivalent operations may therefore share one actual CUBIN/catalog family
/// while their recipes and strict final bindings retain distinct operation and
/// operand identities. The physical-fixture lowering above remains unchanged.
pub fn lower_reusable_pointwise_cuda_aot_candidate(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  limits : PointwiseAotLimits,
) -> PointwiseCudaLowering raise PointwiseAotError {
  validate_offline_policy(target, compiler)
  let parameters = pointwise_parameters(operation)
  if parameters.family == KvCacheWrite {
    raise Unsupported(Operation)
  }
  validate_operands(operation, parameters, profile, operands)
  let work_items = match parameters.family {
    RmsNorm => profile.max_query_tokens().to_int64()
    QkRmsNorm =>
      checked_mul(
        profile.max_query_tokens().to_int64(),
        (parameters.query_heads + 2 * parameters.kv_heads).to_int64(),
      )
    _ =>
      checked_mul(
        profile.max_query_tokens().to_int64(),
        parameters.width.to_int64(),
      )
  }
  let dimensions = pointwise_dimensions(parameters.family, work_items)
  let entry_point_id = @catalog.AotKernelEntryPointId::from_stable_int(1) catch {
    _ => raise Invalid(Operation)
  }
  let symbol = "lunaflux_luna_" +
    family_tag(parameters.family) +
    "_bf16_release_v1"
  let source = pointwise_source(symbol, parameters)
  finish_lowering(
    parameters.family,
    operation,
    entry_point_id,
    target,
    profile,
    compiler,
    operands,
    dimensions,
    symbol,
    source,
    limits,
  )
}

///|
/// Lowers the reference-grade standalone KV write used for differential and
/// dispatch evidence. The current model plan fuses persistent KV ownership
/// into paged attention, so this auxiliary is deliberately not manifest-
/// bindable. Its ABI retains canonical attention runtime-input order, followed
/// by rotated QKV, Key base, and Value base.
pub fn lower_auxiliary_kv_cache_write_cuda_aot(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  layout : @device_layout.DeviceKvLayout,
  entry_point_id : @catalog.AotKernelEntryPointId,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  operands : ArrayView[@launch_contract.AotOperand],
  limits : PointwiseAotLimits,
) -> PointwiseCudaLowering raise PointwiseAotError {
  validate_offline_policy(target, compiler)
  let parameters = kv_write_parameters(operation, layout)
  validate_kv_write_operands(operation, parameters, profile, layout, operands)
  let kv_width = checked_mul(
    parameters.kv_heads.to_int64(),
    parameters.head_dimension.to_int64(),
  )
  let work_items = checked_mul(profile.max_query_tokens().to_int64(), kv_width)
  let dimensions = pointwise_dimensions(KvCacheWrite, work_items)
  let symbol = pointwise_symbol(KvCacheWrite, operation.id(), entry_point_id)
  let source = kv_write_source(symbol, parameters)
  finish_lowering(
    KvCacheWrite,
    operation,
    entry_point_id,
    target,
    profile,
    compiler,
    operands,
    dimensions,
    symbol,
    source,
    limits,
  )
}
