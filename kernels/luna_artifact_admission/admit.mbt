///|
fn luna_artifact_module_index(
  artifacts : @artifact.KernelArtifactBundle,
  digest : @catalog.AotArtifactDigest,
) -> Int? {
  for index, module_ in artifacts.modules() {
    if module_.digest() == digest {
      return Some(index)
    }
  }
  None
}

///|
fn find_luna_specialized_contract(
  manifest : @luna_capability_manifest.LunaAdmittedCapabilityManifestEntry,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> @launch_contract.PagedKvAotLaunchContract raise LunaAotAdmissionError {
  let expected_id = manifest.entry_point_id().as_int()
  let shape = manifest.graph_metadata().shape_class()
  let mut found : @launch_contract.PagedKvAotLaunchContract? = None
  for contract in contracts.contracts() {
    if contract.operation_id() == manifest.operation_id() &&
      contract.entry_point().entry_point_id().as_int() == expected_id {
      let profile = contract.profile()
      let tokens_per_page = contract.tokens_per_page()
      let context = Int64::from_int(shape.context_tokens())
      let page_tokens = Int64::from_int(tokens_per_page)
      let pages_per_row = if context == 0L {
        0L
      } else {
        if context > 9223372036854775807L - (page_tokens - 1L) {
          raise Mismatch(Profile)
        }
        (context + page_tokens - 1L) / page_tokens
      }
      let batch_rows = Int64::from_int(shape.batch_rows())
      if pages_per_row != 0L &&
        batch_rows > 9223372036854775807L / pages_per_row {
        raise Mismatch(Profile)
      }
      let page_entries = pages_per_row * batch_rows
      if profile.max_query_rows() < shape.batch_rows() ||
        Int64::from_int(profile.max_query_tokens()) < shape.token_rows() ||
        Int64::from_int(profile.max_page_table_entries()) < page_entries {
        raise Mismatch(Profile)
      }
      if found is Some(_) {
        raise Ambiguous(EntryPoint)
      }
      found = Some(contract)
    }
  }
  guard found is Some(contract) else { raise Missing(EntryPoint) }
  contract
}

///|
pub fn admit_luna_specialized_artifact(
  manifest : @luna_capability_manifest.LunaAdmittedCapabilityManifestEntry,
  specialization : @luna_specialization.LunaSpecializationRecord,
  artifacts : @artifact.KernelArtifactBundle,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> LunaAotKernelAdmission raise LunaAotAdmissionError {
  if manifest.specialization_digest() != specialization.digest() {
    raise Mismatch(Specialization)
  }
  let identity = manifest.model_identity()
  let target = manifest.target()
  if specialization.identity() != identity ||
    artifacts.model_identity() != identity ||
    contracts.model_identity() != identity {
    raise Mismatch(ModelIdentity)
  }
  if specialization.target() != target ||
    artifacts.device_target() != target ||
    contracts.device_target() != target {
    raise Mismatch(DeviceTarget)
  }
  let catalog_version = manifest.catalog_version()
  if specialization.catalog_version() != catalog_version ||
    artifacts.catalog_version() != catalog_version ||
    contracts.catalog_version() != catalog_version {
    raise Mismatch(CatalogVersion)
  }
  if contracts.scope() != FullGraph {
    raise Unsupported(ContractScope)
  }
  let contract = find_luna_specialized_contract(manifest, contracts)
  if contract.operation_kind() != specialization.operation_kind() ||
    contract.operation_shape() != specialization.operation_shape() {
    raise Mismatch(Operation)
  }
  if contract.capability() != manifest.capability() ||
    contract.capability() != specialization.capability() {
    raise Mismatch(Capability)
  }
  if contract.semantic_version() != manifest.semantic_version() ||
    contract.semantic_version() != specialization.semantic_version() {
    raise Mismatch(SemanticVersion)
  }
  let kv_layout = manifest.kv_layout()
  if contract.kv_layout_version() != kv_layout.stable_version() ||
    contract.tokens_per_page() != kv_layout.tokens_per_page() ||
    contracts.device_kv_layout().stable_layout_version() !=
    kv_layout.stable_version() ||
    contracts.device_kv_layout().tokens_per_page() !=
    kv_layout.tokens_per_page() {
    raise Mismatch(KvLayout)
  }
  if contract.workspace_bytes() != manifest.workspace_bytes() ||
    contract.workspace_alignment() != manifest.workspace_alignment() ||
    specialization.workspace_bytes() != contract.workspace_bytes() ||
    specialization.workspace_alignment() != contract.workspace_alignment() {
    raise Mismatch(Workspace)
  }
  let minimum_alignment_bytes = manifest.minimum_alignment_bytes()
  for operand in contract.operands() {
    if operand.alignment() < minimum_alignment_bytes {
      raise Mismatch(Alignment)
    }
  }
  let entry_point = contract.entry_point()
  let module_digest = manifest.module_digest()
  if specialization.module_digest() != module_digest ||
    entry_point.family().module_digest() != module_digest {
    raise Mismatch(Module)
  }
  guard luna_artifact_module_index(artifacts, module_digest)
    is Some(module_index) else {
    raise Missing(Module)
  }
  guard artifacts.entry_point(entry_point) is Some(artifact_entry) else {
    raise Missing(Artifact)
  }
  if artifact_entry.module_index() != module_index {
    raise Mismatch(Artifact)
  }
  let execution = manifest.execution_disposition()
  if execution != LunaValidatedEagerFallback {
    raise Unsupported(GraphFallback)
  }
  {
    manifest_digest: manifest.manifest_digest(),
    specialization_digest: specialization.digest(),
    identity,
    target,
    catalog_version,
    operation_id: contract.operation_id(),
    operation_kind: contract.operation_kind(),
    operation_shape: contract.operation_shape(),
    capability: contract.capability(),
    entry_point,
    module_digest,
    artifact_module_index: module_index,
    workspace_bytes: contract.workspace_bytes(),
    workspace_alignment: contract.workspace_alignment(),
    minimum_alignment_bytes,
    approved_source: manifest.approved_source_identity(),
    approved_key: manifest.approved_key_identity(),
    detached_signature: manifest.detached_signature_digest(),
    graph: manifest.graph_metadata(),
    execution,
  }
}
