///|
fn readonly_runtime_device_canonical(
  module_ : ApprovedPagedAttentionReadOnlyModule,
  live_device : ReadOnlyLiveDeviceIdentity,
) -> Bytes {
  @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-runtime-device-record.v2\n" +
    "manifest_sha256=" +
    module_.manifest_digest().as_hex() +
    "\ncandidate_sha256=" +
    module_.candidate_digest_sha256() +
    "\ncubin_sha256=" +
    module_.module_digest_sha256() +
    "\napproved_physical_policy_sha256=" +
    module_.approved_physical_policy_digest_sha256() +
    "\ntoolchain_sha256=" +
    module_.toolchain_digest_sha256() +
    "\ndriver_identity_sha256=" +
    module_.driver_identity_sha256() +
    "\nlive_device_ordinal=" +
    live_device.ordinal.to_string() +
    "\nlive_device_uuid=" +
    live_device.uuid +
    "\nlive_device_pci_bus_id=" +
    live_device.pci +
    "\nlive_device_name_hex=" +
    live_device.name_hex +
    "\nlive_device_total_memory_bytes=" +
    live_device.total_memory_bytes.to_string() +
    "\nlive_target=sm_" +
    live_device.capability.compute_major().to_string() +
    live_device.capability.compute_minor().to_string() +
    "\nobservation=opened-context-device-identity" +
    "\nexternal_signature_required=true\n",
  )
}

///|
fn readonly_live_device_identity(
  live_device : @device.ContextDeviceIdentity,
) -> ReadOnlyLiveDeviceIdentity {
  {
    ordinal: live_device.ordinal(),
    name_hex: @crypto.bytes_to_hex_string(@utf8.encode(live_device.name())),
    uuid: live_device.uuid(),
    pci: live_device.pci(),
    total_memory_bytes: live_device.total_memory_bytes(),
    capability: live_device.capability(),
  }
}

///|
fn require_readonly_live_device_identity(
  module_ : ApprovedPagedAttentionReadOnlyModule,
  live_device : ReadOnlyLiveDeviceIdentity,
) -> Unit raise ReadOnlyProductionError {
  if live_device.ordinal < 0 ||
    live_device.uuid != module_.approved_device_uuid() ||
    live_device.pci != module_.approved_device_pci_bus_id() ||
    live_device.name_hex != module_.approved_device_name_hex() ||
    live_device.total_memory_bytes !=
    module_.approved_device_total_memory_bytes() ||
    !module_.target().matches_capability(live_device.capability) {
    raise MismatchedReadOnlyProduction(RuntimeDevice)
  }
}

///|
fn prepare_paged_attention_readonly_runtime_device_record_from_live_identity(
  module_ : ApprovedPagedAttentionReadOnlyModule,
  live_device : ReadOnlyLiveDeviceIdentity,
) -> PagedAttentionReadOnlyRuntimeDeviceRecord raise ReadOnlyProductionError {
  require_readonly_live_device_identity(module_, live_device)
  let canonical = readonly_runtime_device_canonical(module_, live_device)
  {
    module_,
    live_device,
    canonical,
    digest: readonly_production_digest(canonical),
  }
}

///|
pub fn prepare_paged_attention_readonly_runtime_device_record(
  module_ : ApprovedPagedAttentionReadOnlyModule,
  live_device : @device.ContextDeviceIdentity,
) -> PagedAttentionReadOnlyRuntimeDeviceRecord raise ReadOnlyProductionError {
  prepare_paged_attention_readonly_runtime_device_record_from_live_identity(
    module_,
    readonly_live_device_identity(live_device),
  )
}

///|
pub fn admit_external_paged_attention_readonly_runtime_device(
  record : PagedAttentionReadOnlyRuntimeDeviceRecord,
  envelope : @luna_capability_manifest.LunaDetachedSignatureEnvelope,
  receipt : @luna_capability_manifest.LunaAuthenticatedExternalApproval,
) -> ApprovedPagedAttentionReadOnlyRuntimeDevice raise ReadOnlyProductionError {
  require_readonly_live_device_identity(record.module_, record.live_device)
  let canonical = readonly_runtime_device_canonical(
    record.module_,
    record.live_device,
  )
  if canonical != record.canonical ||
    readonly_production_digest(canonical) != record.digest {
    raise MismatchedReadOnlyProduction(PhysicalPolicy)
  }
  let approval = @luna_capability_manifest.admit_luna_external_approved_record(
    canonical,
    envelope,
    receipt,
    maximum_canonical_bytes=16384,
  ) catch {
    _ => raise MismatchedReadOnlyProduction(ExternalApproval)
  }
  if approval.record_digest().as_hex() != record.digest.as_hex() {
    raise MismatchedReadOnlyProduction(ExternalApproval)
  }
  { record, approval }
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeDevice::matches_module(
  self : ApprovedPagedAttentionReadOnlyRuntimeDevice,
  module_ : ApprovedPagedAttentionReadOnlyModule,
  live_device : @device.ContextDeviceIdentity,
) -> Bool {
  let observed = readonly_live_device_identity(live_device)
  require_readonly_live_device_identity(module_, observed) catch {
    _ => return false
  }
  self.record.live_device == observed &&
  self.record.module_.manifest_digest() == module_.manifest_digest() &&
  self.record.module_.candidate_digest_sha256() ==
  module_.candidate_digest_sha256() &&
  self.record.module_.module_digest_sha256() == module_.module_digest_sha256() &&
  self.record.module_.approved_physical_policy_digest_sha256() ==
  module_.approved_physical_policy_digest_sha256() &&
  self.record.module_.approved_device_uuid() == module_.approved_device_uuid() &&
  self.record.module_.approved_device_pci_bus_id() ==
  module_.approved_device_pci_bus_id() &&
  self.record.module_.approved_device_name_hex() ==
  module_.approved_device_name_hex() &&
  self.record.module_.approved_device_total_memory_bytes() ==
  module_.approved_device_total_memory_bytes() &&
  self.record.module_.target() == module_.target() &&
  self.approval.record_digest().as_hex() == self.record.digest.as_hex()
}

///|
pub fn ApprovedPagedAttentionReadOnlyRuntimeDevice::digest(
  self : ApprovedPagedAttentionReadOnlyRuntimeDevice,
) -> ReadOnlyProductionDigest {
  self.record.digest
}
