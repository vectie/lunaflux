///|
pub(all) enum PagedCompositeArtifactRule {
  Approval
  Digest
  Module
  KernelAbi
  RuntimeDevice
  Fallback
} derive(Debug, Eq)

///|
pub(all) suberror PagedCompositeArtifactError {
  InvalidPagedCompositeArtifact(PagedCompositeArtifactRule)
  MismatchedPagedCompositeArtifact(PagedCompositeArtifactRule)
} derive(Debug, Eq)

///|
/// Startup-authenticated readonly module identity retained by runtime code.
/// It has no public fields and cannot be created without both opaque external
/// approval records and the exact opened-context device identity.
pub struct ApprovedPagedAttentionReadOnlyRuntimeIdentity {
  priv manifest : String
  priv catalog : String
  priv candidate : String
  priv source : String
  priv recipe : String
  priv fallback_source : String
  priv fallback_recipe : String
  priv compiled_binding : String
  priv module_digest : String
  priv physical_evidence : String
  priv approved_policy : String
  priv toolchain : String
  priv driver : String
  priv target : @catalog.DeviceTarget
  priv operation : Int
  priv activation : Int
  priv input_row_width : Int
  priv profile_id : Int
  priv profile_rows : Int
  priv profile_tokens : Int
  priv profile_page_entries : Int
  priv tokens_per_page : Int
  priv page_stride_bytes : Int64
  priv physical_page_capacity : Int64
  priv symbol : String
  priv abi : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyKernelAbi
  priv launch : @launch_contract.AotLaunchDimensions
  priv module_bytes : Bytes
  priv dispatch_canary : Int
  priv dispatch_canary_cells : Int
  priv device_uuid : String
  priv device_pci : String
  priv device_name_hex : String
  priv device_memory_bytes : Int64
  priv runtime_device : String
}

///|
/// Physically qualified standalone KV-write fallback identity. This identity
/// deliberately carries no module bytes or runtime dispatch authority.
pub struct QualifiedPagedKvWriteFallbackIdentity {
  priv model_content : String
  priv model_plan : String
  priv candidate : String
  priv source : String
  priv recipe : String
  priv compiled_binding : String
  priv module_digest : String
  priv physical_evidence : String
  priv approved_policy : String
  priv fallback_source : String
  priv fallback_recipe : String
  priv operation : @plan.OperationId
  priv activation : @plan.ActivationRef
  priv row_width : Int
  priv target : @catalog.DeviceTarget
  priv profile : @launch_contract.PagedKernelProfile
  priv tokens_per_page : Int
  priv total_page_count : Int
  priv page_stride_bytes : Int64
  priv toolchain : String
  priv driver : String
}

///|
/// Exact canary-free artifacts needed by the production QKV-to-readonly
/// composite. Construction is kernel-owned and returns no authority for a
/// diagnostic or mixed-ABI pair.
pub struct ApprovedFusedQkvReadOnlyProductionArtifacts {
  priv fused_catalog : @luna_fused_artifact_admission.ApprovedFusedKernelCatalog
  priv fused_bundle : @luna_fused_artifact_admission.ApprovedFusedKernelBundle
  priv readonly_identity : ApprovedPagedAttentionReadOnlyRuntimeIdentity
  priv kv_write : QualifiedPagedKvWriteFallbackIdentity
}

///|
/// Digest-pinned inert aggregate received by a child. It exposes only the
/// fused catalog/bundle needed for startup planning; executable QKV authority
/// is withheld until `bind_live_device` joins the opened context identity.
pub struct DigestPinnedFusedProductionRuntime {
  priv artifacts : ApprovedFusedQkvReadOnlyProductionArtifacts
}
