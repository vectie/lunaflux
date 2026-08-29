///|
/// Re-authenticates an immutable artifact bundle against one exact admitted
/// tensor-parallel launch set without opening modules or backend resources.
///
/// This is the owning-package guard used by later aggregate planners: it
/// checks canonical required-entry/module order, exact content digests,
/// symbol grammar, module indices, uniqueness, and the total-byte scalar.
pub fn KernelArtifactBundle::authenticates_tensor_parallel(
  self : KernelArtifactBundle,
  contracts : @tensor_parallel_launch_contract.TensorParallelAotLaunchContractSet,
  limits : KernelArtifactLimits,
) -> Bool {
  if self.model_identity != contracts.model_identity() ||
    self.device_target != contracts.device_target() ||
    self.catalog_version != contracts.catalog_version() ||
    self.catalog_version != @catalog.CatalogVersion::v3() {
    return false
  }
  if contracts.contracts().is_empty() {
    return false
  }
  let required_entries = required_tensor_parallel_entry_points(
    contracts,
    limits.max_entry_points,
  ) catch {
    _ => return false
  }
  let required_digests = required_modules(required_entries, limits.max_modules) catch {
    _ => return false
  }
  if self.entry_points.length() != required_entries.length() ||
    self.modules.length() != required_digests.length() {
    return false
  }
  let mut total = 0L
  for index, artifact_module in self.modules {
    if artifact_module.digest != required_digests[index] ||
      artifact_module.module_bytes.length() == 0 ||
      artifact_module.module_bytes.length().to_int64() > limits.max_module_bytes ||
      exact_digest(artifact_module.module_bytes) !=
      artifact_module.digest.sha256() {
      return false
    }
    let byte_count = artifact_module.module_bytes.length().to_int64()
    if total > limits.max_total_module_bytes - byte_count {
      return false
    }
    total += byte_count
  }
  if total != self.total_module_bytes {
    return false
  }
  for index, entry in self.entry_points {
    let required = required_entries[index]
    if entry.entry_point != required {
      return false
    }
    guard digest_index(required_digests, required.family().module_digest())
      is Some(module_index) else {
      return false
    }
    if entry.module_index != module_index {
      return false
    }
    validate_symbol(
      entry.entry_point,
      entry.function_symbol,
      limits.max_function_symbol_bytes,
    ) catch {
      _ => return false
    }
  }
  true
}
