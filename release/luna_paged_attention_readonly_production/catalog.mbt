///|
fn readonly_exact_launch(
  identity : ReadOnlyQualifiedIdentity,
) -> @launch_contract.AotLaunchDimensions raise ReadOnlyProductionError {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=identity.profile_tokens,
    grid_y=1,
    grid_z=1,
    block_x=1,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise InvalidReadOnlyProduction(Launch)
  }
}

///|
fn readonly_catalog_canonical(
  manifest : ApprovedPagedAttentionReadOnlyManifest,
  identity : ReadOnlyQualifiedIdentity,
) -> Bytes {
  @utf8.encode(
    "schema=lunaflux-approved-paged-attention-readonly-catalog.v1\n" +
    "manifest_sha256=" +
    manifest.digest.as_hex() +
    "\ncandidate_sha256=" +
    identity.candidate +
    "\nsource_sha256=" +
    identity.source +
    "\nrecipe_sha256=" +
    identity.recipe +
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
    "\noperation_id=" +
    identity.operation.to_string() +
    "\nactivation_ref=" +
    identity.activation.to_string() +
    "\ninput_row_width=" +
    identity.input_row_width.to_string() +
    "\nprofile_id=" +
    identity.profile_id.to_string() +
    "\nentry_count=1\nselection=exact-manifest-candidate-binding-operation-activation\n",
  )
}

///|
pub fn admit_approved_paged_attention_readonly_catalog(
  manifest : ApprovedPagedAttentionReadOnlyManifest,
) -> ApprovedPagedAttentionReadOnlyCatalog raise ReadOnlyProductionError {
  if !manifest.manifest_bindable() ||
    !manifest.production_authorized() ||
    readonly_production_digest(manifest.canonical) != manifest.digest {
    raise MismatchedReadOnlyProduction(Catalog)
  }
  let approved = manifest.approved
  let identity = extract_readonly_qualified_identity(
    approved.record.qualified,
    approved.record.policy,
  )
  if readonly_manifest_canonical(approved) != manifest.canonical {
    raise MismatchedReadOnlyProduction(Catalog)
  }
  let module_ = {
    manifest: manifest.digest,
    identity,
    module_bytes: approved.record.qualified.compiled().module_bytes(),
    launch: readonly_exact_launch(identity),
  }
  if readonly_production_digest(module_.module_bytes).as_hex() !=
    identity.module_digest {
    raise MismatchedReadOnlyProduction(CompiledArtifact)
  }
  let canonical = readonly_catalog_canonical(manifest, identity)
  {
    manifest,
    module_,
    canonical,
    digest: readonly_production_digest(canonical),
  }
}

///|
pub fn ApprovedPagedAttentionReadOnlyCatalog::resolve_exact(
  self : ApprovedPagedAttentionReadOnlyCatalog,
  manifest_digest : ReadOnlyProductionDigest,
  target : @catalog.DeviceTarget,
  candidate_sha256 : String,
  source_sha256 : String,
  recipe_sha256 : String,
  compiled_binding_sha256 : String,
  module_sha256 : String,
  physical_evidence_sha256 : String,
  approved_physical_policy_sha256 : String,
  operation_id : Int,
  activation_ref : Int,
  input_row_width : Int,
  profile : @launch_contract.PagedKernelProfile,
  symbol : String,
  abi : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyKernelAbi,
  launch : @launch_contract.AotLaunchDimensions,
) -> ApprovedPagedAttentionReadOnlyModule raise ReadOnlyProductionError {
  if readonly_production_digest(self.canonical) != self.digest ||
    readonly_catalog_canonical(self.manifest, self.module_.identity) !=
    self.canonical ||
    manifest_digest != self.manifest.digest ||
    target != self.module_.identity.target {
    raise MismatchedReadOnlyProduction(Catalog)
  }
  let expected = self.module_.identity
  if candidate_sha256 != expected.candidate ||
    source_sha256 != expected.source ||
    recipe_sha256 != expected.recipe ||
    compiled_binding_sha256 != expected.compiled_binding ||
    module_sha256 != expected.module_digest ||
    physical_evidence_sha256 != expected.physical_evidence ||
    approved_physical_policy_sha256 != expected.approved_physical_policy {
    raise MismatchedReadOnlyProduction(Candidate)
  }
  if operation_id != expected.operation ||
    activation_ref != expected.activation ||
    input_row_width != expected.input_row_width {
    raise MismatchedReadOnlyProduction(Operation)
  }
  if profile.id().as_int() != expected.profile_id ||
    profile.max_query_rows() != expected.profile_rows ||
    profile.max_query_tokens() != expected.profile_tokens ||
    profile.max_page_table_entries() != expected.profile_page_entries {
    raise MismatchedReadOnlyProduction(Profile)
  }
  if symbol != expected.symbol || abi.canonical_name() != expected.abi {
    raise MismatchedReadOnlyProduction(KernelAbi)
  }
  if launch != self.module_.launch {
    raise MismatchedReadOnlyProduction(Launch)
  }
  self.module_
}
