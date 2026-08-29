///|
fn require_exact_fused_catalog_entry(
  catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog,
  module_ : @luna_fused_artifact_admission.ApprovedFusedKernelModule,
) -> Unit raise FusedGroupedOperationError {
  if module_.target() != catalog.target() {
    raise MismatchedFusedGroupedOperation(Target)
  }
  let resolved = catalog.resolve_exact(
    catalog.manifest_digest(),
    catalog.target(),
    module_.family(),
    module_.candidate_digest(),
    module_.source_digest(),
    module_.recipe_digest(),
    module_.toolchain_digest_sha256(),
    module_.driver_identity_sha256(),
    module_.function_symbol(),
    module_.kernel_abi(),
    module_.launch_dimensions(),
    module_.module_digest(),
    module_.compiled_binding_digest(),
  ) catch {
    _ => raise MismatchedFusedGroupedOperation(Catalog)
  }
  if resolved.module_digest() != module_.module_digest() ||
    resolved.compiled_binding_digest() != module_.compiled_binding_digest() {
    raise MismatchedFusedGroupedOperation(Catalog)
  }
}

///|
fn require_fallback_operation(
  bundle : @luna_kernel_bundle.LunaKernelBundleInput,
  fallback : @luna_fused_artifact_admission.FusedStandaloneFallbackIdentity,
  expected_role : @luna_fused_artifact_admission.FusedStandaloneFallbackRole,
  expected_kind : @plan.OperationKind,
  target : @catalog.DeviceTarget,
  plan : @plan.ModelPlan,
) -> Int raise FusedGroupedOperationError {
  if fallback.role() != expected_role {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
  let index = fallback.operation_index()
  guard plan.operations().get(index) is Some(operation) &&
    operation.id().as_int() == index &&
    operation.kind() == expected_kind else {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
  let mut matches = 0
  for compiled in bundle.operations() {
    if compiled.operation_id() == operation.id() {
      matches += 1
      if compiled.device_target() != target ||
        compiled.source_digest_sha256() != fallback.source_digest().as_hex() ||
        compiled.recipe_digest_sha256() != fallback.recipe_digest().as_hex() {
        raise MismatchedFusedGroupedOperation(StandaloneFallback)
      }
    }
  }
  if matches != 1 {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
  index
}

///|
fn require_blueprint_fallback_module(
  bundle : @luna_kernel_bundle.LunaKernelBundleInput,
  blueprint : PagedKvExecutionBlueprint,
  index : Int,
) -> Unit raise FusedGroupedOperationError {
  guard blueprint.steps().get(index) is Some(step) else {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
  let mut matches = 0
  for compiled in bundle.operations() {
    if compiled.operation_id() == step.operation_id() {
      matches += 1
      if compiled.module_digest() != step.entry_point().family().module_digest() {
        raise MismatchedFusedGroupedOperation(StandaloneFallback)
      }
    }
  }
  if matches != 1 {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
}

///|
pub fn admit_approved_fused_residual_rmsnorm_span(
  plan : @plan.ModelPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  blueprint : PagedKvExecutionBlueprint,
  standalone : @luna_kernel_bundle.LunaKernelBundleInput,
  catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog,
  module_ : @luna_fused_artifact_admission.ApprovedFusedKernelModule,
) -> ApprovedFusedResidualRmsNormSpan raise FusedGroupedOperationError {
  require_exact_fused_catalog_entry(catalog, module_)
  if module_.family() != ResidualRmsNorm ||
    (
      module_.kernel_abi() != ResidualRmsNormAbiV1 &&
      module_.kernel_abi() != ResidualRmsNormProductionAbiV2
    ) {
    raise MismatchedFusedGroupedOperation(Family)
  }
  if blueprint.identity() != plan.identity() ||
    blueprint.target() != catalog.target() {
    raise MismatchedFusedGroupedOperation(Target)
  }
  let fallbacks = module_.fallbacks()
  if fallbacks.length() != 2 {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
  let residual = require_fallback_operation(
    standalone,
    fallbacks[0],
    ResidualAddFallback,
    ResidualAdd,
    catalog.target(),
    plan,
  )
  let norm = require_fallback_operation(
    standalone,
    fallbacks[1],
    RmsNormFallback,
    RmsNorm,
    catalog.target(),
    plan,
  )
  if residual + 1 != norm ||
    blueprint.steps().get(residual) is None ||
    blueprint.steps().get(norm) is None {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
  require_blueprint_fallback_module(standalone, blueprint, residual)
  require_blueprint_fallback_module(standalone, blueprint, norm)
  let canary = module_.dispatch_canary_per_token()
  match module_.kernel_abi() {
    ResidualRmsNormAbiV1 =>
      if canary == 0U ||
        blueprint.limits().max_tokens().reinterpret_as_uint() >
        0xffffffffU / canary {
        raise InvalidFusedGroupedOperation(KernelAbi)
      }
    ResidualRmsNormProductionAbiV2 =>
      if canary != 0U {
        raise InvalidFusedGroupedOperation(KernelAbi)
      }
    _ => raise InvalidFusedGroupedOperation(KernelAbi)
  }
  {
    generation,
    manifest: catalog.manifest_digest(),
    catalog: catalog.digest(),
    target: module_.target(),
    module_bytes: module_.module_bytes(),
    function_symbol: module_.function_symbol(),
    launch: module_.launch_dimensions(),
    abi: module_.kernel_abi(),
    dispatch_canary_per_token: canary,
    residual_index: residual,
    norm_index: norm,
  }
}

///|
/// Projects the canary-free production residual/RMSNorm replacement from an
/// already authenticated kernel bundle and its exact catalog. This admission
/// runs before device resources open and deliberately does not accept release
/// evidence or independently supplied module bytes.
pub fn admit_production_fused_residual_rmsnorm_span(
  plan : @plan.ModelPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  blueprint : PagedKvExecutionBlueprint,
  catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog,
  bundle : @luna_fused_artifact_admission.ApprovedFusedKernelBundle,
) -> ApprovedFusedResidualRmsNormSpan raise FusedGroupedOperationError {
  if bundle.manifest_digest() != catalog.manifest_digest() ||
    bundle.target() != catalog.target() {
    raise MismatchedFusedGroupedOperation(Catalog)
  }
  let module_ = bundle.residual_module()
  require_exact_fused_catalog_entry(catalog, module_)
  if module_.family() != ResidualRmsNorm ||
    module_.artifact_purpose() != ProductionFastPath ||
    module_.kernel_abi() != ResidualRmsNormProductionAbiV2 ||
    module_.dispatch_canary_per_token() != 0U {
    raise MismatchedFusedGroupedOperation(Family)
  }
  if blueprint.identity() != plan.identity() ||
    blueprint.target() != catalog.target() {
    raise MismatchedFusedGroupedOperation(Target)
  }
  let fallbacks = module_.fallbacks()
  if fallbacks.length() != 2 ||
    fallbacks[0].role() != ResidualAddFallback ||
    fallbacks[1].role() != RmsNormFallback {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
  let residual = fallbacks[0].operation_index()
  let norm = fallbacks[1].operation_index()
  guard plan.operations().get(residual) is Some(residual_operation) &&
    plan.operations().get(norm) is Some(norm_operation) &&
    blueprint.steps().get(residual) is Some(residual_step) &&
    blueprint.steps().get(norm) is Some(norm_step) &&
    residual_operation.id().as_int() == residual &&
    norm_operation.id().as_int() == norm &&
    residual_operation.kind() == ResidualAdd &&
    norm_operation.kind() == RmsNorm &&
    residual_step.operation_id() == residual_operation.id() &&
    norm_step.operation_id() == norm_operation.id() &&
    residual_step.operation_kind() == ResidualAdd &&
    norm_step.operation_kind() == RmsNorm &&
    residual + 1 == norm else {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
  {
    generation,
    manifest: catalog.manifest_digest(),
    catalog: catalog.digest(),
    target: module_.target(),
    module_bytes: module_.module_bytes(),
    function_symbol: module_.function_symbol(),
    launch: module_.launch_dimensions(),
    abi: module_.kernel_abi(),
    dispatch_canary_per_token: 0U,
    residual_index: residual,
    norm_index: norm,
  }
}

///|
/// Reauthenticates the QKV catalog entry and exact adjacency, but returns no
/// executable authority. `PagedKvExecutionBlueprint` currently has only a
/// monolithic attention launch whose KV write cannot be safely replaced.
pub fn require_fused_qkv_read_only_attention_split(
  plan : @plan.ModelPlan,
  generation : @worker_protocol.ModelPlanGeneration,
  blueprint : PagedKvExecutionBlueprint,
  catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog,
  module_ : @luna_fused_artifact_admission.ApprovedFusedKernelModule,
) -> FusedQkvReadOnlyAttentionSplitRequirement raise FusedGroupedOperationError {
  require_exact_fused_catalog_entry(catalog, module_)
  if module_.family() != QkvPositionedRopePagedKvWrite ||
    module_.kernel_abi() != QkvPositionedRopePagedKvWriteAbiV1 ||
    blueprint.identity() != plan.identity() ||
    blueprint.target() != catalog.target() {
    raise MismatchedFusedGroupedOperation(Family)
  }
  let fallbacks = module_.fallbacks()
  if fallbacks.length() != 3 ||
    fallbacks[0].role() != QkvProjectionFallback ||
    fallbacks[1].role() != PositionedRopeFallback ||
    fallbacks[2].role() != PagedKvWriteFallback {
    raise MismatchedFusedGroupedOperation(StandaloneFallback)
  }
  let qkv = fallbacks[0].operation_index()
  let rope = fallbacks[1].operation_index()
  let attention = fallbacks[2].operation_index()
  guard plan.operations().get(qkv) is Some(qkv_op) &&
    plan.operations().get(rope) is Some(rope_op) &&
    plan.operations().get(attention) is Some(attention_op) &&
    qkv_op.kind() == QkvProjection &&
    rope_op.kind() == RotaryEmbedding &&
    attention_op.kind() == CausalAttention &&
    qkv + 1 == rope &&
    rope + 1 == attention else {
    raise MismatchedFusedGroupedOperation(OperationOrder)
  }
  {
    generation,
    manifest: catalog.manifest_digest(),
    qkv_index: qkv,
    rope_index: rope,
    attention_index: attention,
  }
}
