///|
fn require_exact_fused_qkv_catalog_entry(
  catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog,
  module_ : @luna_fused_artifact_admission.ApprovedFusedKernelModule,
) -> Unit raise PagedCompositeArtifactError {
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
    _ => raise MismatchedPagedCompositeArtifact(Approval)
  }
  if resolved.module_digest() != module_.module_digest() ||
    resolved.compiled_binding_digest() != module_.compiled_binding_digest() {
    raise MismatchedPagedCompositeArtifact(Approval)
  }
}

///|
/// Selects the executable production pair. Diagnostic and mixed-ABI inputs
/// deliberately return `None`, preserving the standalone graph. Once both V2
/// bindings are present, every cross-artifact identity is checked exactly.
pub fn select_fused_qkv_readonly_production_artifacts(
  fused_catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog,
  fused_bundle : @luna_fused_artifact_admission.ApprovedFusedKernelBundle,
  readonly_identity : ApprovedPagedAttentionReadOnlyRuntimeIdentity,
  kv_write : QualifiedPagedKvWriteFallbackIdentity,
) -> ApprovedFusedQkvReadOnlyProductionArtifacts? raise PagedCompositeArtifactError {
  let fused_qkv = fused_bundle.qkv_module()
  if fused_qkv.artifact_purpose() != ProductionFastPath ||
    fused_qkv.kernel_abi() != QkvPositionedRopePagedKvWriteProductionAbiV2 ||
    fused_qkv.dispatch_canary_per_token() != 0U ||
    readonly_identity.kernel_abi() !=
    RotatedQueryPagedKvReadOnlyProductionRawPointerV2 ||
    readonly_identity.dispatch_canary_per_token() != 0 ||
    readonly_identity.dispatch_canary_cell_count() != 0 {
    return None
  }
  if fused_bundle.manifest_digest() != fused_catalog.manifest_digest() ||
    fused_bundle.target() != fused_catalog.target() ||
    fused_qkv.family() != QkvPositionedRopePagedKvWrite {
    raise MismatchedPagedCompositeArtifact(Approval)
  }
  require_exact_fused_qkv_catalog_entry(fused_catalog, fused_qkv)
  if fused_qkv.target() != readonly_identity.target() ||
    fused_qkv.target() != kv_write.target() ||
    fused_qkv.toolchain_digest_sha256() !=
    readonly_identity.toolchain_digest_sha256() ||
    fused_qkv.toolchain_digest_sha256() != kv_write.toolchain_digest_sha256() ||
    fused_qkv.driver_identity_sha256() !=
    readonly_identity.driver_identity_sha256() ||
    fused_qkv.driver_identity_sha256() != kv_write.driver_identity_sha256() {
    raise MismatchedPagedCompositeArtifact(Approval)
  }
  let fallbacks = fused_qkv.fallbacks()
  if fallbacks.length() != 3 ||
    fallbacks[0].role() != QkvProjectionFallback ||
    fallbacks[1].role() != PositionedRopeFallback ||
    fallbacks[2].role() != PagedKvWriteFallback ||
    fallbacks[2].source_digest().as_hex() !=
    readonly_identity.fallback_source_digest_sha256() ||
    fallbacks[2].recipe_digest().as_hex() !=
    readonly_identity.fallback_recipe_digest_sha256() ||
    fallbacks[2].source_digest().as_hex() !=
    kv_write.fallback_source_digest_sha256() ||
    fallbacks[2].recipe_digest().as_hex() !=
    kv_write.fallback_recipe_digest_sha256() {
    raise MismatchedPagedCompositeArtifact(Fallback)
  }
  Some({ fused_catalog, fused_bundle, readonly_identity, kv_write })
}
