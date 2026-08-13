///|
/// Immutable startup catalog. Entry order has no selection priority.
pub struct KernelCatalog {
  version : CatalogVersion
  entries : ReadOnlyArray[CatalogEntry]
} derive(Debug, Eq)

///|
/// One exact implementation binding aligned with a semantic model operation.
pub struct KernelBinding {
  operation_id : @plan.OperationId
  capability : @plan.KernelCapabilityId
  semantic_version : KernelSemanticVersion
  execution_semantics : KernelExecutionSemantics?
  implementation : KernelImplementationId
  workspace_bytes : Int64
  workspace_alignment : Int64
} derive(Debug, Eq)

///|
struct AotFamilySemantics {
  semantic_version : KernelSemanticVersion
  operation_kind : @plan.OperationKind
  operation_shape : @plan.OperationShape
  capability : @plan.KernelCapabilityId
  execution_semantics : KernelExecutionSemantics?
} derive(Debug, Eq)

///|
fn same_persistent_effect(
  left : @plan.PersistentStateAccess,
  right : @plan.PersistentStateAccess,
) -> Bool {
  match (left, right) {
    (NoPersistentState, NoPersistentState) => true
    (KvCacheReadWrite(_), KvCacheReadWrite(_)) => true
    _ => false
  }
}

///|
fn same_family_execution_semantics(
  left : KernelExecutionSemantics?,
  right : KernelExecutionSemantics?,
) -> Bool {
  match (left, right) {
    (None, None) => true
    (Some(a), Some(b)) =>
      (a.kv_layer is Some(_)) == (b.kv_layer is Some(_)) &&
      a.runtime_inputs == b.runtime_inputs &&
      same_persistent_effect(a.persistent_state, b.persistent_state) &&
      a.kv_layout == b.kv_layout
    _ => false
  }
}

///|
fn same_family_semantics(
  existing : AotFamilySemantics,
  entry : CatalogEntry,
) -> Bool {
  existing.semantic_version == entry.semantic_version &&
  existing.operation_kind == entry.operation_kind &&
  existing.operation_shape == entry.operation_shape &&
  existing.capability == entry.capability &&
  same_family_execution_semantics(
    existing.execution_semantics,
    entry.execution_semantics,
  )
}

///|
/// Immutable result of resolving every operation in one model plan.
pub struct ResolvedKernelCatalog {
  catalog_version : CatalogVersion
  model_identity : @spec.ModelIdentity
  device_target : DeviceTarget
  bindings : ReadOnlyArray[KernelBinding]
  max_workspace_bytes : Int64
  max_workspace_alignment : Int64
} derive(Debug, Eq)

///|
pub fn KernelCatalog::new(
  version~ : CatalogVersion,
  entries~ : ArrayView[CatalogEntry],
) -> KernelCatalog raise CatalogError {
  if version != CatalogVersion::v1() && version != CatalogVersion::v3() {
    raise InvalidCatalog(UnsupportedCatalogVersion)
  }
  if entries.is_empty() {
    raise InvalidCatalog(EmptyCatalog)
  }
  if entries.length() > 65536 {
    raise InvalidCatalog(CatalogTooLarge)
  }
  let expected_semantic_version = if version == CatalogVersion::v1() {
    KernelSemanticVersion::v1()
  } else {
    KernelSemanticVersion::v3()
  }
  let aot_semantics : Map[AotKernelFamily, AotFamilySemantics] = Map([])
  for entry in entries {
    if entry.semantic_version != expected_semantic_version ||
      (version == CatalogVersion::v1()) != (entry.execution_semantics is None) {
      raise InvalidCatalog(CatalogSemanticVersionMismatch)
    }
    match entry.implementation {
      VendorBf16GemmImplicitHeuristicV1 => ()
      ContentAddressedAot(family) =>
        match aot_semantics.get(family) {
          Some(existing) =>
            if !same_family_semantics(existing, entry) {
              raise InvalidCatalog(AotFamilySemanticMismatch)
            }
          None =>
            aot_semantics[family] = {
              semantic_version: entry.semantic_version,
              operation_kind: entry.operation_kind,
              operation_shape: entry.operation_shape,
              capability: entry.capability,
              execution_semantics: entry.execution_semantics,
            }
        }
    }
  }
  { version, entries: ReadOnlyArray::from_array(entries) }
}

///|
pub fn KernelCatalog::version(self : KernelCatalog) -> CatalogVersion {
  self.version
}

///|
pub fn KernelCatalog::entries(
  self : KernelCatalog,
) -> ReadOnlyArray[CatalogEntry] {
  self.entries
}

///|
fn semantic_match(
  entry : CatalogEntry,
  operation : @plan.PlanOperation,
  geometry : @plan.KvCacheGeometry,
  paged_layout : PagedKvLayoutContract?,
) -> Bool {
  let execution_matches = match entry.execution_semantics {
    None =>
      operation.runtime_inputs().is_empty() &&
      operation.persistent_state() == NoPersistentState
    Some(semantics) =>
      if semantics.kv_layer != operation.kv_layer() ||
        semantics.persistent_state != operation.persistent_state() ||
        !runtime_inputs_match(
          semantics.runtime_inputs,
          operation.runtime_inputs(),
        ) {
        false
      } else {
        match semantics.kv_layout {
          Some(layout) =>
            layout.layout == geometry.layout() &&
            layout.stable_version == geometry.layout().stable_version() &&
            Some(layout) == paged_layout
          None => operation.persistent_state() == NoPersistentState
        }
      }
  }
  execution_matches &&
  entry.operation_kind == operation.kind() &&
  entry.operation_shape == operation.shape() &&
  entry.capability == operation.required_capability()
}

///|
fn architecture_match(entry : CatalogEntry, target : DeviceTarget) -> Bool {
  entry.compute_major == target.compute_major &&
  entry.compute_minor == target.compute_minor
}

///|
fn implementation_supported(
  implementation : KernelImplementationId,
  target : DeviceTarget,
) -> Bool {
  if !target.supports_bf16 {
    return false
  }
  match implementation {
    VendorBf16GemmImplicitHeuristicV1 => target.supports_cublas_lt
    ContentAddressedAot(_) => true
  }
}

///|
fn aligned_workspace(
  byte_count : Int64,
  alignment : Int64,
  operation_id : @plan.OperationId,
) -> Int64 raise CatalogError {
  let remainder = byte_count % alignment
  if remainder == 0L {
    return byte_count
  }
  let padding = alignment - remainder
  if byte_count > 9223372036854775807L - padding {
    raise WorkspaceSizeOverflow(operation_id)
  }
  byte_count + padding
}

///|
fn select_entry(
  catalog : KernelCatalog,
  operation : @plan.PlanOperation,
  geometry : @plan.KvCacheGeometry,
  paged_layout : PagedKvLayoutContract?,
  target : DeviceTarget,
) -> CatalogEntry raise CatalogError {
  let semantic : Array[CatalogEntry] = []
  let architecture : Array[CatalogEntry] = []
  let compatible : Array[CatalogEntry] = []
  for entry in catalog.entries {
    if semantic_match(entry, operation, geometry, paged_layout) {
      semantic.push(entry)
      if architecture_match(entry, target) {
        architecture.push(entry)
        if implementation_supported(entry.implementation, target) {
          compatible.push(entry)
        }
      }
    }
  }
  if semantic.is_empty() {
    raise MissingKernel(operation.id())
  }
  if architecture.is_empty() {
    raise IncompatibleTarget(operation.id())
  }
  if compatible.is_empty() {
    raise IncompatibleImplementation(operation.id())
  }
  if compatible.length() != 1 {
    raise AmbiguousKernel(operation.id())
  }
  compatible[0]
}

///|
/// Resolves every operation exactly once. There is no catalog-order priority,
/// architecture range, precision conversion, JIT, or fallback selection.
pub fn KernelCatalog::resolve(
  self : KernelCatalog,
  model_plan : @plan.ModelPlan,
  target : DeviceTarget,
) -> ResolvedKernelCatalog raise CatalogError {
  if self.version != CatalogVersion::v1() ||
    model_plan.kv_execution() != StatelessFullContext {
    raise UnsupportedExecutionContract
  }
  self.resolve_exact(model_plan, target, None)
}

///|
/// Resolves catalog v3 against the exact capacity-independent KV layout.
///
/// Tokens per page is an AOT indexing specialization. Total physical page
/// count is not: it only sizes later admitted Key and Value component spans.
pub fn KernelCatalog::resolve_paged(
  self : KernelCatalog,
  model_plan : @plan.ModelPlan,
  device_kv_layout : @device_layout.DeviceKvLayout,
  target : DeviceTarget,
) -> ResolvedKernelCatalog raise CatalogError {
  if self.version != CatalogVersion::v3() ||
    model_plan.kv_execution() != PagedKeyValue {
    raise UnsupportedExecutionContract
  }
  if device_kv_layout.identity() != model_plan.identity() ||
    device_kv_layout.geometry() != model_plan.kv_cache_geometry() ||
    device_kv_layout.stable_layout_version() !=
    model_plan.kv_cache_geometry().layout().stable_version() {
    raise DeviceKvLayoutMismatch
  }
  let paged_layout = PagedKvLayoutContract::new(
    layout=device_kv_layout.geometry().layout(),
    stable_version=device_kv_layout.stable_layout_version(),
    tokens_per_page=device_kv_layout.tokens_per_page(),
  )
  self.resolve_exact(model_plan, target, Some(paged_layout))
}

///|
fn KernelCatalog::resolve_exact(
  self : KernelCatalog,
  model_plan : @plan.ModelPlan,
  target : DeviceTarget,
  paged_layout : PagedKvLayoutContract?,
) -> ResolvedKernelCatalog raise CatalogError {
  let bindings : Array[KernelBinding] = []
  let mut max_workspace_bytes = 0L
  let mut max_workspace_alignment = 1L
  let workspace_limit = model_plan.workspace_bounds().max_temporary_bytes()
  for operation in model_plan.operations() {
    let entry = select_entry(
      self,
      operation,
      model_plan.kv_cache_geometry(),
      paged_layout,
      target,
    )
    let required = aligned_workspace(
      entry.workspace_bytes,
      entry.workspace_alignment,
      operation.id(),
    )
    if required < operation.workspace_bytes() {
      raise WorkspaceRequirementMismatch(operation.id())
    }
    if required > workspace_limit {
      raise WorkspaceLimitExceeded(operation.id())
    }
    if required > max_workspace_bytes {
      max_workspace_bytes = required
    }
    if entry.workspace_alignment > max_workspace_alignment {
      max_workspace_alignment = entry.workspace_alignment
    }
    bindings.push({
      operation_id: operation.id(),
      capability: operation.required_capability(),
      semantic_version: entry.semantic_version,
      execution_semantics: entry.execution_semantics,
      implementation: entry.implementation,
      workspace_bytes: required,
      workspace_alignment: entry.workspace_alignment,
    })
  }
  {
    catalog_version: self.version,
    model_identity: model_plan.identity(),
    device_target: target,
    bindings: ReadOnlyArray::from_array(bindings),
    max_workspace_bytes,
    max_workspace_alignment,
  }
}

///|
pub fn ResolvedKernelCatalog::model_identity(
  self : ResolvedKernelCatalog,
) -> @spec.ModelIdentity {
  self.model_identity
}

///|
pub fn ResolvedKernelCatalog::device_target(
  self : ResolvedKernelCatalog,
) -> DeviceTarget {
  self.device_target
}

///|
pub fn KernelBinding::operation_id(self : KernelBinding) -> @plan.OperationId {
  self.operation_id
}

///|
pub fn KernelBinding::capability(
  self : KernelBinding,
) -> @plan.KernelCapabilityId {
  self.capability
}

///|
pub fn KernelBinding::semantic_version(
  self : KernelBinding,
) -> KernelSemanticVersion {
  self.semantic_version
}

///|
pub fn KernelBinding::execution_semantics(
  self : KernelBinding,
) -> KernelExecutionSemantics? {
  self.execution_semantics
}

///|
pub fn KernelBinding::implementation(
  self : KernelBinding,
) -> KernelImplementationId {
  self.implementation
}

///|
pub fn KernelBinding::workspace_bytes(self : KernelBinding) -> Int64 {
  self.workspace_bytes
}

///|
pub fn KernelBinding::workspace_alignment(self : KernelBinding) -> Int64 {
  self.workspace_alignment
}

///|
pub fn ResolvedKernelCatalog::catalog_version(
  self : ResolvedKernelCatalog,
) -> CatalogVersion {
  self.catalog_version
}

///|
pub fn ResolvedKernelCatalog::bindings(
  self : ResolvedKernelCatalog,
) -> ReadOnlyArray[KernelBinding] {
  self.bindings
}

///|
pub fn ResolvedKernelCatalog::max_workspace_bytes(
  self : ResolvedKernelCatalog,
) -> Int64 {
  self.max_workspace_bytes
}

///|
pub fn ResolvedKernelCatalog::max_workspace_alignment(
  self : ResolvedKernelCatalog,
) -> Int64 {
  self.max_workspace_alignment
}
