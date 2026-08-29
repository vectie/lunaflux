///|
/// Bounds the legacy shared admission path's comparison work before any
/// required-entry allocation or scan. Catalog-v4 artifact admission supports
/// at most this many profile/operation contracts even when callers configured
/// a larger generic launch-contract ceiling.
const ABSOLUTE_MAX_PAGED_V4_ARTIFACT_CONTRACTS : Int = 1024

///|
fn ensure_paged_v4_artifact_contract_count(
  count : Int,
) -> Unit raise KernelArtifactError {
  if count > ABSOLUTE_MAX_PAGED_V4_ARTIFACT_CONTRACTS {
    raise ContractCountExceeded(
      maximum=ABSOLUTE_MAX_PAGED_V4_ARTIFACT_CONTRACTS,
    )
  }
}

///|
fn ensure_paged_v4_catalog_version(
  actual : @catalog.CatalogVersion,
) -> Unit raise KernelArtifactError {
  let expected = @catalog.CatalogVersion::v4()
  if actual != expected {
    raise CatalogVersionMismatch(expected~, actual~)
  }
}

///|
/// Admits exactly the immutable AOT modules and bounded CUDA symbols required
/// by one catalog-v4 full-graph launch-contract set.
///
/// Required entry points retain their first exact profile/operation occurrence
/// from the already-admitted contract order. The shared artifact path then
/// enforces exact model, target, and catalog identity; missing, duplicate, and
/// unreferenced declarations; content digests; symbol grammar; and defensive
/// ownership of aliased module bytes. This remains inert artifact evidence and
/// acquires no loader, device, bootstrap, or execution authority.
pub fn admit_paged_v4(
  contracts : @launch_contract.PagedV4LaunchContractSet,
  module_inputs : ArrayView[KernelModuleInput],
  entry_point_inputs : ArrayView[KernelEntryPointInput],
  limits : KernelArtifactLimits,
) -> KernelArtifactBundle raise KernelArtifactError {
  ensure_paged_v4_artifact_contract_count(contracts.contracts().length())
  ensure_paged_v4_catalog_version(contracts.catalog_version())
  let required_entries : Array[@catalog.AotKernelEntryPoint] = []
  for contract in contracts.contracts() {
    append_required_entry_point(
      required_entries,
      contract.entry_point(),
      limits.max_entry_points,
    )
  }
  admit_required(
    contracts.model_identity(),
    contracts.device_target(),
    contracts.catalog_version(),
    required_entries,
    module_inputs,
    entry_point_inputs,
    limits,
  )
}
