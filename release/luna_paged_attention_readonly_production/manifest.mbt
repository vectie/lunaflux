///|
fn readonly_manifest_canonical(
  approved : ApprovedPagedAttentionReadOnlyArtifacts,
) -> Bytes {
  let identity = approved.record.identity
  @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-production-manifest.v1\n" +
    "approval_record_sha256=" +
    approved.record.digest.as_hex() +
    "\napproved_source_sha256=" +
    approved.approval.approved_source_identity().as_hex() +
    "\napproved_key_sha256=" +
    approved.approval.approved_key_identity().as_hex() +
    "\ndetached_signature_sha256=" +
    approved.approval.detached_signature_digest().as_hex() +
    "\ncandidate_sha256=" +
    identity.candidate +
    "\ncompiled_binding_sha256=" +
    identity.compiled_binding +
    "\ncubin_sha256=" +
    identity.module_digest +
    "\nphysical_evidence_sha256=" +
    identity.physical_evidence +
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
    "\nentry_count=1\nselection=exact-approved-identity-tuple" +
    "\nruntime_dispatch_authority=approved-exact-manifest-only\n",
  )
}

///|
pub fn admit_approved_paged_attention_readonly_manifest(
  approved : ApprovedPagedAttentionReadOnlyArtifacts,
) -> ApprovedPagedAttentionReadOnlyManifest raise ReadOnlyProductionError {
  let identity = extract_readonly_qualified_identity(
    approved.record.qualified,
    approved.record.policy,
  )
  let approval_canonical = readonly_approval_canonical(identity)
  if approval_canonical != approved.record.canonical ||
    readonly_production_digest(approval_canonical) != approved.record.digest ||
    approved.approval.record_digest().as_hex() !=
    approved.record.digest.as_hex() {
    raise MismatchedReadOnlyProduction(ExternalApproval)
  }
  let canonical = readonly_manifest_canonical(approved)
  { approved, canonical, digest: readonly_production_digest(canonical) }
}

///|
pub fn ApprovedPagedAttentionReadOnlyManifest::digest(
  self : ApprovedPagedAttentionReadOnlyManifest,
) -> ReadOnlyProductionDigest {
  self.digest
}

///|
pub fn ApprovedPagedAttentionReadOnlyManifest::canonical_bytes(
  self : ApprovedPagedAttentionReadOnlyManifest,
) -> Bytes {
  self.canonical
}

///|
pub fn ApprovedPagedAttentionReadOnlyManifest::manifest_bindable(
  _self : ApprovedPagedAttentionReadOnlyManifest,
) -> Bool {
  true
}

///|
pub fn ApprovedPagedAttentionReadOnlyManifest::production_authorized(
  _self : ApprovedPagedAttentionReadOnlyManifest,
) -> Bool {
  true
}
