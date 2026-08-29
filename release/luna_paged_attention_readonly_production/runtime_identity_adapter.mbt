///|
/// Offline/startup adapter from release validation into the kernel-owned
/// runtime identity. No release evidence object crosses this boundary.
pub fn adapt_paged_attention_readonly_runtime_identity(
  catalog : ApprovedPagedAttentionReadOnlyCatalog,
  runtime_device : ApprovedPagedAttentionReadOnlyRuntimeDevice,
  live_device : @device.ContextDeviceIdentity,
) -> @luna_paged_composite_artifact_admission.ApprovedPagedAttentionReadOnlyRuntimeIdentity raise ReadOnlyProductionError {
  let module_ = catalog.module_
  if !runtime_device.matches_module(module_, live_device) {
    raise MismatchedReadOnlyProduction(RuntimeDevice)
  }
  let abi = if module_.kernel_abi_name() ==
    @luna_cuda_paged_attention_readonly_aot.RotatedQueryPagedKvReadOnlyRawPointerV1.canonical_name() {
    @luna_cuda_paged_attention_readonly_aot.RotatedQueryPagedKvReadOnlyRawPointerV1
  } else if module_.kernel_abi_name() ==
    @luna_cuda_paged_attention_readonly_aot.RotatedQueryPagedKvReadOnlyProductionRawPointerV2.canonical_name() {
    RotatedQueryPagedKvReadOnlyProductionRawPointerV2
  } else {
    raise MismatchedReadOnlyProduction(KernelAbi)
  }
  let shape = module_.profile_shape()
  @luna_paged_composite_artifact_admission.admit_paged_attention_readonly_runtime_identity(
    manifest_sha256=module_.manifest_digest().as_hex(),
    catalog_sha256=catalog.digest().as_hex(),
    approval_record_sha256=catalog.manifest.approved.approval
      .record_digest()
      .as_hex(),
    runtime_device_sha256=runtime_device.digest().as_hex(),
    candidate_sha256=module_.candidate_digest_sha256(),
    source_sha256=module_.source_digest_sha256(),
    recipe_sha256=module_.recipe_digest_sha256(),
    fallback_source_sha256=module_.fallback_source_digest_sha256(),
    fallback_recipe_sha256=module_.fallback_recipe_digest_sha256(),
    compiled_binding_sha256=module_.compiled_binding_digest_sha256(),
    module_sha256=module_.module_digest_sha256(),
    physical_evidence_sha256=module_.physical_evidence_digest_sha256(),
    approved_policy_sha256=module_.approved_physical_policy_digest_sha256(),
    toolchain_sha256=module_.toolchain_digest_sha256(),
    driver_sha256=module_.driver_identity_sha256(),
    target=module_.target(),
    operation=module_.operation_id(),
    activation=module_.activation_ref(),
    input_row_width=module_.input_row_width(),
    profile_id=module_.profile_id(),
    profile_rows=shape.0,
    profile_tokens=shape.1,
    profile_page_entries=shape.2,
    tokens_per_page=module_.tokens_per_page(),
    page_stride_bytes=module_.page_stride_bytes(),
    physical_page_capacity=module_.physical_page_capacity(),
    symbol=module_.function_symbol(),
    abi~,
    launch=module_.launch_dimensions(),
    module_bytes=module_.module_bytes(),
    dispatch_canary=module_.dispatch_canary_per_token(),
    dispatch_canary_cells=module_.dispatch_canary_cell_count(),
    approved_device_uuid=module_.approved_device_uuid(),
    approved_device_pci=module_.approved_device_pci_bus_id(),
    approved_device_name_hex=module_.approved_device_name_hex(),
    approved_device_memory_bytes=module_.approved_device_total_memory_bytes(),
    manifest_approval=catalog.manifest.approved.approval,
    runtime_device_approval=runtime_device.approval,
    live_device~,
  ) catch {
    _ => raise MismatchedReadOnlyProduction(Catalog)
  }
}
