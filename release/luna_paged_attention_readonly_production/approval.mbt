///|
fn readonly_approval_canonical(identity : ReadOnlyQualifiedIdentity) -> Bytes {
  @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-approval-record.v1\n" +
    "scope=exact-approved-readonly-paged-attention\n" +
    "candidate_sha256=" +
    identity.candidate +
    "\nsource_sha256=" +
    identity.source +
    "\nrecipe_sha256=" +
    identity.recipe +
    "\nfallback_source_sha256=" +
    identity.fallback_source +
    "\nfallback_recipe_sha256=" +
    identity.fallback_recipe +
    "\ncompiled_binding_sha256=" +
    identity.compiled_binding +
    "\ncubin_sha256=" +
    identity.module_digest +
    "\ncompile_receipt_sha256=" +
    identity.compile_receipt +
    "\nphysical_evidence_sha256=" +
    identity.physical_evidence +
    "\nphysical_files_manifest_sha256=" +
    identity.physical_files_manifest +
    "\nphysical_outer_seal_sha256=" +
    identity.physical_outer_seal +
    "\napproved_physical_policy_sha256=" +
    identity.approved_physical_policy +
    "\ntoolchain_sha256=" +
    identity.toolchain +
    "\ndriver_identity_sha256=" +
    identity.driver +
    "\ndevice_uuid=" +
    identity.device_uuid +
    "\ndevice_pci_bus_id=" +
    identity.device_pci_bus_id +
    "\ndevice_name_hex=" +
    identity.device_name_hex +
    "\ndevice_total_memory_bytes=" +
    identity.device_total_memory_bytes +
    "\ntarget=sm_120\noperation_id=" +
    identity.operation.to_string() +
    "\nactivation_ref=" +
    identity.activation.to_string() +
    "\ninput_row_width=" +
    identity.input_row_width.to_string() +
    "\nprofile_id=" +
    identity.profile_id.to_string() +
    "\nprofile=" +
    identity.profile_rows.to_string() +
    "," +
    identity.profile_tokens.to_string() +
    "," +
    identity.profile_page_entries.to_string() +
    "\nkv_layout=layer_major_split_key_value_v1\nkv_layout_stable_version=1" +
    "\ntokens_per_page=" +
    identity.tokens_per_page.to_string() +
    "\npage_stride_bytes=" +
    identity.page_stride_bytes.to_string() +
    "\nphysical_page_capacity=" +
    identity.physical_page_capacity.to_string() +
    "\nfunction_symbol=" +
    identity.symbol +
    "\nraw_pointer_abi=" +
    identity.abi +
    "\ndispatch_canary_per_token=" +
    identity.dispatch_canary.to_string() +
    "\ndispatch_canary_cell_count=" +
    identity.dispatch_canary_cells.to_string() +
    "\nkv_cache_mutation=none\nexternal_signature_required=true" +
    "\nrequested_authority=exact-readonly-production-manifest\n",
  )
}

///|
pub fn prepare_paged_attention_readonly_approval_record(
  qualified : @luna_paged_attention_readonly_physical_evidence.QualifiedPagedAttentionReadOnlyArtifact,
  approved_policy : @luna_paged_attention_readonly_physical_evidence.ApprovedPagedAttentionReadOnlyPhysicalPolicy,
) -> PagedAttentionReadOnlyApprovalRecord raise ReadOnlyProductionError {
  let identity = extract_readonly_qualified_identity(qualified, approved_policy)
  let canonical = readonly_approval_canonical(identity)
  if canonical.length() > 65536 {
    raise InvalidReadOnlyProduction(ExternalApproval)
  }
  {
    qualified,
    policy: approved_policy,
    identity,
    canonical,
    digest: readonly_production_digest(canonical),
  }
}

///|
pub fn admit_external_paged_attention_readonly_approval(
  record : PagedAttentionReadOnlyApprovalRecord,
  envelope : @luna_capability_manifest.LunaDetachedSignatureEnvelope,
  receipt : @luna_capability_manifest.LunaAuthenticatedExternalApproval,
) -> ApprovedPagedAttentionReadOnlyArtifacts raise ReadOnlyProductionError {
  let expected = extract_readonly_qualified_identity(
    record.qualified,
    record.policy,
  )
  let canonical = readonly_approval_canonical(expected)
  if canonical != record.canonical ||
    readonly_production_digest(canonical) != record.digest {
    raise MismatchedReadOnlyProduction(ExternalApproval)
  }
  let approval = @luna_capability_manifest.admit_luna_external_approved_record(
    canonical,
    envelope,
    receipt,
    maximum_canonical_bytes=65536,
  ) catch {
    _ => raise MismatchedReadOnlyProduction(ExternalApproval)
  }
  if approval.record_digest().as_hex() != record.digest.as_hex() {
    raise MismatchedReadOnlyProduction(ExternalApproval)
  }
  { record, approval }
}
