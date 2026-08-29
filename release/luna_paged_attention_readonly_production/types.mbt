///|
priv struct ReadOnlyQualifiedIdentity {
  candidate : String
  source : String
  recipe : String
  fallback_source : String
  fallback_recipe : String
  compiled_binding : String
  module_digest : String
  compile_receipt : String
  physical_evidence : String
  physical_files_manifest : String
  physical_outer_seal : String
  approved_physical_policy : String
  toolchain : String
  driver : String
  device_uuid : String
  device_pci_bus_id : String
  device_name_hex : String
  device_total_memory_bytes : String
  operation : Int
  activation : Int
  input_row_width : Int
  profile_id : Int
  profile_rows : Int
  profile_tokens : Int
  profile_page_entries : Int
  tokens_per_page : Int
  page_stride_bytes : Int64
  physical_page_capacity : Int64
  dispatch_canary : Int
  dispatch_canary_cells : Int
  symbol : String
  abi : String
  target : @catalog.DeviceTarget
}

///|
pub struct PagedAttentionReadOnlyApprovalRecord {
  priv qualified : @luna_paged_attention_readonly_physical_evidence.QualifiedPagedAttentionReadOnlyArtifact
  priv policy : @luna_paged_attention_readonly_physical_evidence.ApprovedPagedAttentionReadOnlyPhysicalPolicy
  priv identity : ReadOnlyQualifiedIdentity
  priv canonical : Bytes
  priv digest : ReadOnlyProductionDigest
}

///|
pub struct ApprovedPagedAttentionReadOnlyArtifacts {
  priv record : PagedAttentionReadOnlyApprovalRecord
  priv approval : @luna_capability_manifest.LunaAdmittedExternalRecord
}

///|
pub struct ApprovedPagedAttentionReadOnlyManifest {
  priv approved : ApprovedPagedAttentionReadOnlyArtifacts
  priv canonical : Bytes
  priv digest : ReadOnlyProductionDigest
}

///|
pub struct ApprovedPagedAttentionReadOnlyModule {
  priv manifest : ReadOnlyProductionDigest
  priv identity : ReadOnlyQualifiedIdentity
  priv module_bytes : Bytes
  priv launch : @launch_contract.AotLaunchDimensions
}

///|
pub struct ApprovedPagedAttentionReadOnlyCatalog {
  priv manifest : ApprovedPagedAttentionReadOnlyManifest
  priv module_ : ApprovedPagedAttentionReadOnlyModule
  priv canonical : Bytes
  priv digest : ReadOnlyProductionDigest
}

///|
priv struct ReadOnlyLiveDeviceIdentity {
  ordinal : Int
  name_hex : String
  uuid : String
  pci : String
  total_memory_bytes : Int64
  capability : @device.DeviceCapability
} derive(Eq)

///|
pub struct PagedAttentionReadOnlyRuntimeDeviceRecord {
  priv module_ : ApprovedPagedAttentionReadOnlyModule
  priv live_device : ReadOnlyLiveDeviceIdentity
  priv canonical : Bytes
  priv digest : ReadOnlyProductionDigest
}

///|
pub struct ApprovedPagedAttentionReadOnlyRuntimeDevice {
  priv record : PagedAttentionReadOnlyRuntimeDeviceRecord
  priv approval : @luna_capability_manifest.LunaAdmittedExternalRecord
}
