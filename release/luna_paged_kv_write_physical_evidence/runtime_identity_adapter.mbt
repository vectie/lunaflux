///|
/// Projects only the fallback identity required by composite startup
/// admission. Module bytes and dispatch authority are intentionally omitted.
pub fn adapt_qualified_paged_kv_write_fallback_identity(
  qualified : QualifiedPagedKvWriteArtifact,
) -> @luna_paged_composite_artifact_admission.QualifiedPagedKvWriteFallbackIdentity raise PagedKvWritePhysicalError {
  @luna_paged_composite_artifact_admission.qualified_paged_kv_write_fallback_identity(
    model_content_sha256=qualified.model_content_digest_sha256(),
    model_plan_sha256=qualified.model_plan_digest_sha256(),
    candidate_sha256=qualified.candidate_digest().as_hex(),
    source_sha256=qualified.source_digest().as_hex(),
    recipe_sha256=qualified.recipe_digest().as_hex(),
    compiled_binding_sha256=qualified.compiled_binding_digest().as_hex(),
    module_sha256=qualified.module_digest().as_hex(),
    physical_evidence_sha256=qualified.evidence_digest().as_hex(),
    approved_policy_sha256=qualified.approved_policy_digest().as_hex(),
    fallback_source_sha256=qualified.fallback_source_digest_sha256(),
    fallback_recipe_sha256=qualified.fallback_recipe_digest_sha256(),
    operation=qualified.operation_id(),
    activation=qualified.positioned_qkv_activation(),
    row_width=qualified.positioned_qkv_row_width(),
    target=qualified.target(),
    profile=qualified.profile(),
    tokens_per_page=qualified.tokens_per_page(),
    total_page_count=qualified.total_page_count(),
    page_stride_bytes=qualified.page_stride_bytes(),
    toolchain_sha256=qualified.toolchain_digest_sha256(),
    driver_sha256=qualified.driver_identity_sha256(),
  ) catch {
    _ => raise Mismatch(Candidate)
  }
}
