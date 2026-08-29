///|
fn paged_composite_is_sha256(value : String) -> Bool {
  if value.length() != 64 {
    return false
  }
  for unit in value {
    if !(unit is ('0'..='9' | 'a'..='f')) {
      return false
    }
  }
  true
}

///|
pub fn admit_paged_attention_readonly_runtime_identity(
  manifest_sha256~ : String,
  catalog_sha256~ : String,
  approval_record_sha256~ : String,
  runtime_device_sha256~ : String,
  candidate_sha256~ : String,
  source_sha256~ : String,
  recipe_sha256~ : String,
  fallback_source_sha256~ : String,
  fallback_recipe_sha256~ : String,
  compiled_binding_sha256~ : String,
  module_sha256~ : String,
  physical_evidence_sha256~ : String,
  approved_policy_sha256~ : String,
  toolchain_sha256~ : String,
  driver_sha256~ : String,
  target~ : @catalog.DeviceTarget,
  operation~ : Int,
  activation~ : Int,
  input_row_width~ : Int,
  profile_id~ : Int,
  profile_rows~ : Int,
  profile_tokens~ : Int,
  profile_page_entries~ : Int,
  tokens_per_page~ : Int,
  page_stride_bytes~ : Int64,
  physical_page_capacity~ : Int64,
  symbol~ : String,
  abi~ : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyKernelAbi,
  launch~ : @launch_contract.AotLaunchDimensions,
  module_bytes~ : Bytes,
  dispatch_canary~ : Int,
  dispatch_canary_cells~ : Int,
  approved_device_uuid~ : String,
  approved_device_pci~ : String,
  approved_device_name_hex~ : String,
  approved_device_memory_bytes~ : Int64,
  manifest_approval~ : @luna_capability_manifest.LunaAdmittedExternalRecord,
  runtime_device_approval~ : @luna_capability_manifest.LunaAdmittedExternalRecord,
  live_device~ : @device.ContextDeviceIdentity,
) -> ApprovedPagedAttentionReadOnlyRuntimeIdentity raise PagedCompositeArtifactError {
  for
    digest in [
      manifest_sha256, catalog_sha256, approval_record_sha256, runtime_device_sha256,
      candidate_sha256, source_sha256, recipe_sha256, fallback_source_sha256, fallback_recipe_sha256,
      compiled_binding_sha256, module_sha256, physical_evidence_sha256, approved_policy_sha256,
      toolchain_sha256, driver_sha256,
    ] {
    if !paged_composite_is_sha256(digest) {
      raise InvalidPagedCompositeArtifact(Digest)
    }
  }
  if manifest_approval.record_digest().as_hex() != approval_record_sha256 ||
    runtime_device_approval.record_digest().as_hex() != runtime_device_sha256 {
    raise MismatchedPagedCompositeArtifact(Approval)
  }
  if module_bytes.is_empty() ||
    @crypto.bytes_to_hex_string(@crypto.sha256(module_bytes)) != module_sha256 {
    raise MismatchedPagedCompositeArtifact(Module)
  }
  match abi {
    RotatedQueryPagedKvReadOnlyRawPointerV1 =>
      if dispatch_canary <= 0 || dispatch_canary_cells != profile_tokens {
        raise MismatchedPagedCompositeArtifact(KernelAbi)
      }
    RotatedQueryPagedKvReadOnlyProductionRawPointerV2 =>
      if dispatch_canary != 0 || dispatch_canary_cells != 0 {
        raise MismatchedPagedCompositeArtifact(KernelAbi)
      }
  }
  if operation < 0 ||
    activation < 0 ||
    input_row_width <= 0 ||
    profile_id < 0 ||
    profile_rows <= 0 ||
    profile_tokens < profile_rows ||
    profile_page_entries <= 0 ||
    tokens_per_page <= 0 ||
    page_stride_bytes <= 0L ||
    physical_page_capacity <= 0L ||
    symbol.is_empty() ||
    approved_device_memory_bytes <= 0L {
    raise InvalidPagedCompositeArtifact(Module)
  }
  let observed = live_device.capability()
  if live_device.uuid() != approved_device_uuid ||
    live_device.pci() != approved_device_pci ||
    @crypto.bytes_to_hex_string(@utf8.encode(live_device.name())) !=
    approved_device_name_hex ||
    live_device.total_memory_bytes() != approved_device_memory_bytes ||
    !target.matches_capability(observed) {
    raise MismatchedPagedCompositeArtifact(RuntimeDevice)
  }
  {
    manifest: manifest_sha256,
    catalog: catalog_sha256,
    candidate: candidate_sha256,
    source: source_sha256,
    recipe: recipe_sha256,
    fallback_source: fallback_source_sha256,
    fallback_recipe: fallback_recipe_sha256,
    compiled_binding: compiled_binding_sha256,
    module_digest: module_sha256,
    physical_evidence: physical_evidence_sha256,
    approved_policy: approved_policy_sha256,
    toolchain: toolchain_sha256,
    driver: driver_sha256,
    target,
    operation,
    activation,
    input_row_width,
    profile_id,
    profile_rows,
    profile_tokens,
    profile_page_entries,
    tokens_per_page,
    page_stride_bytes,
    physical_page_capacity,
    symbol,
    abi,
    launch,
    module_bytes,
    dispatch_canary,
    dispatch_canary_cells,
    device_uuid: approved_device_uuid,
    device_pci: approved_device_pci,
    device_name_hex: approved_device_name_hex,
    device_memory_bytes: approved_device_memory_bytes,
    runtime_device: runtime_device_sha256,
  }
}

///|
pub fn qualified_paged_kv_write_fallback_identity(
  model_content_sha256~ : String,
  model_plan_sha256~ : String,
  candidate_sha256~ : String,
  source_sha256~ : String,
  recipe_sha256~ : String,
  compiled_binding_sha256~ : String,
  module_sha256~ : String,
  physical_evidence_sha256~ : String,
  approved_policy_sha256~ : String,
  fallback_source_sha256~ : String,
  fallback_recipe_sha256~ : String,
  operation~ : @plan.OperationId,
  activation~ : @plan.ActivationRef,
  row_width~ : Int,
  target~ : @catalog.DeviceTarget,
  profile~ : @launch_contract.PagedKernelProfile,
  tokens_per_page~ : Int,
  total_page_count~ : Int,
  page_stride_bytes~ : Int64,
  toolchain_sha256~ : String,
  driver_sha256~ : String,
) -> QualifiedPagedKvWriteFallbackIdentity raise PagedCompositeArtifactError {
  for
    digest in [
      model_content_sha256, model_plan_sha256, candidate_sha256, source_sha256, recipe_sha256,
      compiled_binding_sha256, module_sha256, physical_evidence_sha256, approved_policy_sha256,
      fallback_source_sha256, fallback_recipe_sha256, toolchain_sha256, driver_sha256,
    ] {
    if !paged_composite_is_sha256(digest) {
      raise InvalidPagedCompositeArtifact(Digest)
    }
  }
  if row_width <= 0 ||
    tokens_per_page <= 0 ||
    total_page_count <= 0 ||
    page_stride_bytes <= 0L {
    raise InvalidPagedCompositeArtifact(Fallback)
  }
  {
    model_content: model_content_sha256,
    model_plan: model_plan_sha256,
    candidate: candidate_sha256,
    source: source_sha256,
    recipe: recipe_sha256,
    compiled_binding: compiled_binding_sha256,
    module_digest: module_sha256,
    physical_evidence: physical_evidence_sha256,
    approved_policy: approved_policy_sha256,
    fallback_source: fallback_source_sha256,
    fallback_recipe: fallback_recipe_sha256,
    operation,
    activation,
    row_width,
    target,
    profile,
    tokens_per_page,
    total_page_count,
    page_stride_bytes,
    toolchain: toolchain_sha256,
    driver: driver_sha256,
  }
}
