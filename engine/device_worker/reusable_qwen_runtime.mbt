///|
fn require_qwen_ingress_operation_kinds(
  model_plan : @plan.ModelPlan,
  qkv_operation : Int,
  qk_norm_operation : Int,
  rope_operation : Int,
  attention_operation : Int,
) -> Unit raise DeviceWorkerError {
  let operations = model_plan.operations()
  if qkv_operation < 0 ||
    attention_operation >= operations.length() ||
    qk_norm_operation != qkv_operation + 1 ||
    rope_operation != qkv_operation + 2 ||
    attention_operation != qkv_operation + 3 ||
    operations[qkv_operation].kind() != QkvProjection ||
    operations[qk_norm_operation].kind() != QkRmsNorm ||
    operations[rope_operation].kind() != RotaryEmbedding ||
    operations[attention_operation].kind() != CausalAttention {
    raise Invalid(ResourceState)
  }
}

///|
fn require_optional_reusable_residual_seed(
  model_plan : @plan.ModelPlan,
  bundle : @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle,
) -> Unit raise DeviceWorkerError {
  guard bundle.residual_rmsnorm_operation_indices()
    is Some((residual_operation, norm_operation)) else {
    return
  }
  let operations = model_plan.operations()
  if residual_operation < 0 ||
    norm_operation >= operations.length() ||
    norm_operation != residual_operation + 1 ||
    operations[residual_operation].kind() != ResidualAdd ||
    operations[norm_operation].kind() != RmsNorm {
    raise Invalid(ResourceState)
  }
}

///|
/// Validates the exact Qwen-only reusable bundle before it is retained by a
/// worker plan. This is startup admission: token execution never revisits the
/// model identity, target, operation kinds, or pair completeness checks.
fn validate_reusable_qwen_runtime_bundle(
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  bundle : @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle,
) -> Unit raise DeviceWorkerError {
  let identity = model_plan.identity()
  if bundle.model_content_sha256() != identity.content().as_hex() ||
    bundle.model_plan_sha256() != identity.plan().as_hex() {
    raise Invalid(ModelIdentity)
  }
  if bundle.target() != target {
    raise Invalid(DeviceTarget)
  }
  guard bundle.qwen_qknorm_rope_kv_write_operation_indices()
    is Some((qk_norm_operation, rope_operation, attention_operation)) &&
    bundle.qwen_paged_attention_readonly_operation_index()
    is Some(readonly_operation) else {
    raise Invalid(ResourceState)
  }
  let qkv_operation = qk_norm_operation - 1
  if readonly_operation != attention_operation {
    raise Invalid(ResourceState)
  }
  require_qwen_ingress_operation_kinds(
    model_plan, qkv_operation, qk_norm_operation, rope_operation, attention_operation,
  )
  require_optional_reusable_residual_seed(model_plan, bundle)
}

///|
fn admit_optional_reusable_qwen_runtime_bundle(
  model_plan : @plan.ModelPlan,
  target : @catalog.DeviceTarget,
  reusable_bundle : @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle?,
  legacy_residual : @luna_fused_artifact_admission.ApprovedReusableFusedResidualRuntime?,
  fused_catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog?,
  fused_bundle : @luna_fused_artifact_admission.ApprovedFusedKernelBundle?,
  fused_qkv_readonly_artifacts : @luna_paged_composite_artifact_admission.ApprovedFusedQkvReadOnlyProductionArtifacts?,
  fused_qkv_readonly_source : @luna_paged_composite_artifact_admission.DigestPinnedFusedProductionRuntime?,
) -> @luna_fused_artifact_admission.ApprovedReusableFusedRuntimeBundle? raise DeviceWorkerError {
  guard reusable_bundle is Some(bundle) else { return None }
  if fused_catalog is Some(_) ||
    fused_bundle is Some(_) ||
    fused_qkv_readonly_artifacts is Some(_) ||
    fused_qkv_readonly_source is Some(_) ||
    (legacy_residual is Some(_) && bundle.residual_rmsnorm_module() is Some(_)) {
    raise Invalid(ResourceState)
  }
  validate_reusable_qwen_runtime_bundle(model_plan, target, bundle)
  Some(bundle)
}
