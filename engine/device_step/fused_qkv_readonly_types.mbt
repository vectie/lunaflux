///|
pub struct FusedQkvReadOnlyAttentionCompositeRequirement {
  priv generation : @worker_protocol.ModelPlanGeneration
  priv identity : @spec.ModelIdentity
  priv fused_catalog : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  priv fused_qkv : @luna_fused_artifact_admission.ApprovedFusedKernelModule
  priv readonly_identity : @luna_paged_composite_artifact_admission.ApprovedPagedAttentionReadOnlyRuntimeIdentity
  priv live_device : @device.ContextDeviceIdentity
  priv live_device_join : FusedQkvReadOnlyLiveDeviceJoin
  priv kv_write : @luna_paged_composite_artifact_admission.QualifiedPagedKvWriteFallbackIdentity
  priv qkv_index : Int
  priv rope_index : Int
  priv attention_index : Int
  priv positioned_qkv : @plan.ActivationRef
  priv full_qkv_row_width : Int
  priv canonical : Bytes
  priv digest : @luna_cuda_fused_parallel_aot.FusedParallelDigest
}

///|
pub fn FusedQkvReadOnlyAttentionCompositeRequirement::model_generation(
  self : FusedQkvReadOnlyAttentionCompositeRequirement,
) -> @worker_protocol.ModelPlanGeneration {
  self.generation
}

///|
pub fn FusedQkvReadOnlyAttentionCompositeRequirement::operation_indices(
  self : FusedQkvReadOnlyAttentionCompositeRequirement,
) -> (Int, Int, Int) {
  (self.qkv_index, self.rope_index, self.attention_index)
}

///|
pub fn FusedQkvReadOnlyAttentionCompositeRequirement::positioned_qkv_activation(
  self : FusedQkvReadOnlyAttentionCompositeRequirement,
) -> @plan.ActivationRef {
  self.positioned_qkv
}

///|
pub fn FusedQkvReadOnlyAttentionCompositeRequirement::full_qkv_row_width(
  self : FusedQkvReadOnlyAttentionCompositeRequirement,
) -> Int {
  self.full_qkv_row_width
}

///|
pub fn FusedQkvReadOnlyAttentionCompositeRequirement::digest(
  self : FusedQkvReadOnlyAttentionCompositeRequirement,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  self.digest
}

///|
pub fn FusedQkvReadOnlyAttentionCompositeRequirement::dispatch_authorized(
  self : FusedQkvReadOnlyAttentionCompositeRequirement,
) -> Bool {
  try self.require_fused_qkv_readonly_retained_authority() catch {
    _ => return false
  } noraise {
    _ => ()
  }
  fused_qkv_readonly_production_abis_match(
    self.fused_qkv.kernel_abi(),
    self.readonly_identity.kernel_abi().canonical_name(),
  ) &&
  self.fused_qkv.artifact_purpose() == ProductionFastPath &&
  self.fused_qkv.dispatch_canary_per_token() == 0U &&
  self.readonly_identity.dispatch_canary_per_token() == 0 &&
  self.readonly_identity.dispatch_canary_cell_count() == 0
}

///|
/// Exact ABI gate for the future executable replacement. Existing approved
/// diagnostic modules intentionally fail and remain qualification inputs only.
fn fused_qkv_readonly_production_abis_match(
  fused : @luna_cuda_fused_parallel_aot.FusedParallelKernelAbi,
  readonly_abi_name : String,
) -> Bool {
  fused == QkvPositionedRopePagedKvWriteProductionAbiV2 &&
  readonly_abi_name ==
  @luna_cuda_paged_attention_readonly_aot.RotatedQueryPagedKvReadOnlyProductionRawPointerV2.canonical_name()
}

///|
fn FusedQkvReadOnlyAttentionCompositeRequirement::require_fused_qkv_readonly_retained_authority(
  self : FusedQkvReadOnlyAttentionCompositeRequirement,
) -> Unit raise FusedGroupedOperationError {
  if self.identity.content().as_hex() !=
    self.kv_write.model_content_digest_sha256() ||
    self.identity.plan().as_hex() != self.kv_write.model_plan_digest_sha256() ||
    self.fused_catalog.as_hex().length() != 64 ||
    self.readonly_identity.catalog_digest_sha256().length() != 64 ||
    self.fused_qkv.module_bytes().is_empty() ||
    self.readonly_identity.module_bytes().is_empty() ||
    self.readonly_identity.approved_device_uuid() != self.live_device.uuid() ||
    self.readonly_identity.approved_device_pci_bus_id() !=
    self.live_device.pci() ||
    !self.live_device_join.is_exact() ||
    fused_qkv_readonly_digest(self.canonical) != self.digest {
    raise MismatchedFusedGroupedOperation(DispatchAuthority)
  }
}

///|
/// Private, root-free proof that the live opened-context identity exactly
/// matches the device named by the qualified readonly module.
priv struct FusedQkvReadOnlyLiveDeviceJoin {
  expected_uuid : String
  expected_pci : String
  expected_name_hex : String
  expected_memory_bytes : Int64
  expected_compute_major : Int
  expected_compute_minor : Int
  expected_bf16 : Bool
  expected_cublas_lt : Bool
  observed_uuid : String
  observed_pci : String
  observed_name_hex : String
  observed_memory_bytes : Int64
  observed_compute_major : Int
  observed_compute_minor : Int
  observed_bf16 : Bool
  observed_cublas_lt : Bool
}

///|
fn FusedQkvReadOnlyLiveDeviceJoin::is_exact(
  self : FusedQkvReadOnlyLiveDeviceJoin,
) -> Bool {
  self.expected_uuid == self.observed_uuid &&
  self.expected_pci == self.observed_pci &&
  self.expected_name_hex == self.observed_name_hex &&
  self.expected_memory_bytes == self.observed_memory_bytes &&
  self.expected_compute_major == self.observed_compute_major &&
  self.expected_compute_minor == self.observed_compute_minor &&
  self.expected_bf16 == self.observed_bf16 &&
  self.expected_cublas_lt == self.observed_cublas_lt
}
