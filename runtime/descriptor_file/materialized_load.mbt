///|
/// Admits a legacy descriptor from exact no-follow source roots while binding
/// every reconstructed worker/bootstrap identity to the separately declared
/// deployment target labels.
///
/// Only an offline no-overwrite materializer should construct these views.
/// Live startup continues to use `load`, which requires the target namespace
/// itself to exist and match both pinned roots.
pub fn load_materialized(
  model : @approved_fs.ApprovedRootMaterializationView,
  kernel : @approved_fs.ApprovedRootMaterializationView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
  luna_approval? : @worker_wire.WorkerBootstrapLunaApproval? = None,
) -> RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  load_bound(
    model.source_root(),
    kernel.source_root(),
    model.target_label(),
    kernel.target_label(),
    descriptor_locator,
    expected_digest,
    limits,
    luna_approval~,
  )
}

///|
/// Production-I8 counterpart of `load_materialized`. It executes the exact v2
/// descriptor path once; it never probes or falls back to the legacy loader.
pub fn load_i8_v2_materialized(
  model : @approved_fs.ApprovedRootMaterializationView,
  kernel : @approved_fs.ApprovedRootMaterializationView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
) -> I8RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  load_i8_v2_bound(
    model.source_root(),
    kernel.source_root(),
    model.target_label(),
    kernel.target_label(),
    descriptor_locator,
    expected_digest,
    limits,
  )
}

///|
/// Production reusable-FP8 counterpart of `load_materialized`.
pub fn load_fp8_reusable_v3_materialized(
  model : @approved_fs.ApprovedRootMaterializationView,
  kernel : @approved_fs.ApprovedRootMaterializationView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
  approval : @worker_wire.WorkerBootstrapLunaApproval,
) -> Fp8RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  load_fp8_reusable_v3_bound(
    model.source_root(),
    kernel.source_root(),
    model.target_label(),
    kernel.target_label(),
    descriptor_locator,
    expected_digest,
    limits,
    approval,
  )
}

///|
/// Production Mistral counterpart of `load_materialized`.
pub fn load_mistral_v1_materialized(
  model : @approved_fs.ApprovedRootMaterializationView,
  kernel : @approved_fs.ApprovedRootMaterializationView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
) -> RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  load_mistral_bound(
    model.source_root(),
    kernel.source_root(),
    model.target_label(),
    kernel.target_label(),
    descriptor_locator,
    expected_digest,
    limits,
  )
}

///|
/// Dense-Qwen3 BF16 counterpart of `load_materialized`.
pub fn load_qwen3_v1_materialized(
  model : @approved_fs.ApprovedRootMaterializationView,
  kernel : @approved_fs.ApprovedRootMaterializationView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
) -> RuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  load_qwen3_bound(
    model.source_root(),
    kernel.source_root(),
    model.target_label(),
    kernel.target_label(),
    descriptor_locator,
    expected_digest,
    limits,
  )
}

///|
/// Production tensor-parallel counterpart of `load_materialized`.
pub fn load_tensor_parallel_materialized(
  model : @approved_fs.ApprovedRootMaterializationView,
  kernel : @approved_fs.ApprovedRootMaterializationView,
  descriptor_locator : @approved_fs.ApprovedRelativeLocator,
  expected_digest : RuntimeDescriptorDigest,
  limits : RuntimeDescriptorFileLimits,
  approval : @worker_wire.WorkerBootstrapLunaApproval,
) -> TensorParallelRuntimeDescriptorAdmission raise RuntimeDescriptorFileError {
  load_tensor_parallel_bound(
    model.source_root(),
    kernel.source_root(),
    model.target_label(),
    kernel.target_label(),
    descriptor_locator,
    expected_digest,
    limits,
    approval,
  )
}
