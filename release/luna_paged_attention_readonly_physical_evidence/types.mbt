///|
const READONLY_PHYSICAL_RESULT_LINES : Int = 98

///|
pub struct ApprovedPagedAttentionReadOnlyPhysicalPolicy {
  priv canonical : Bytes
  priv digest : PagedAttentionReadOnlyPhysicalEvidenceDigest
  priv nvcc : PagedAttentionReadOnlyPhysicalEvidenceDigest
  priv ptxas : PagedAttentionReadOnlyPhysicalEvidenceDigest
  priv cuobjdump : PagedAttentionReadOnlyPhysicalEvidenceDigest
  priv nvdisasm : PagedAttentionReadOnlyPhysicalEvidenceDigest
  priv sanitizer : PagedAttentionReadOnlyPhysicalEvidenceDigest
  priv driver_identity : PagedAttentionReadOnlyPhysicalEvidenceDigest
  priv driver_record : PagedAttentionReadOnlyPhysicalEvidenceDigest
  priv device_uuid : String
  priv device_pci_bus_id : String
  priv device_name_hex : String
  priv device_total_memory_bytes : Int64
}

///|
pub struct PagedAttentionReadOnlyPhysicalEvidenceLimits {
  priv maximum_result_bytes : Int
  priv maximum_manifest_bytes : Int
}

///|
pub fn PagedAttentionReadOnlyPhysicalEvidenceLimits::new(
  maximum_result_bytes~ : Int,
  maximum_manifest_bytes~ : Int,
) -> PagedAttentionReadOnlyPhysicalEvidenceLimits raise PagedAttentionReadOnlyPhysicalEvidenceError {
  if maximum_result_bytes <= 0 ||
    maximum_result_bytes > 131072 ||
    maximum_manifest_bytes <= 0 ||
    maximum_manifest_bytes > 524288 {
    raise PagedAttentionReadOnlyPhysicalEvidenceLimitExceeded
  }
  { maximum_result_bytes, maximum_manifest_bytes }
}

///|
pub struct AdmittedPagedAttentionReadOnlyPhysicalEvidence {
  priv candidate_digest : String
  priv compiled_binding_digest : String
  priv cubin_digest : PagedAttentionReadOnlyPhysicalEvidenceDigest
  priv fallback_source_digest : String
  priv fallback_recipe_digest : String
  priv operation_id : @plan.OperationId
  priv input_activation : @plan.ActivationRef
  priv input_row_width : Int
  priv dispatch_canary_per_token : Int
  priv dispatch_canary_cell_count : Int
  priv numeric_case_count : Int
  priv canonical : Bytes
  priv digest : PagedAttentionReadOnlyPhysicalEvidenceDigest
  priv files_manifest_digest : PagedAttentionReadOnlyPhysicalEvidenceDigest
  priv outer_seal_digest : PagedAttentionReadOnlyPhysicalEvidenceDigest
}

///|
pub struct QualifiedPagedAttentionReadOnlyArtifact {
  priv candidate : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCudaCandidate
  priv compiled : @luna_cuda_paged_attention_readonly_aot.PagedAttentionReadOnlyCompiledBinding
  priv evidence : AdmittedPagedAttentionReadOnlyPhysicalEvidence
}
