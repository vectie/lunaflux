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
  implementation : KernelImplementationId
  workspace_bytes : Int64
  workspace_alignment : Int64
} derive(Debug, Eq)

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
  if version != CatalogVersion::v1() {
    raise InvalidCatalog(UnsupportedCatalogVersion)
  }
  if entries.is_empty() {
    raise InvalidCatalog(EmptyCatalog)
  }
  if entries.length() > 65536 {
    raise InvalidCatalog(CatalogTooLarge)
  }
  let aot_semantics : Map[
    AotKernelFamily,
    (@plan.OperationKind, @plan.KernelCapabilityId),
  ] = Map([])
  for entry in entries {
    match entry.implementation {
      VendorBf16GemmImplicitHeuristicV1 => ()
      ContentAddressedAot(family) =>
        match aot_semantics.get(family) {
          Some((kind, capability)) =>
            if kind != entry.operation_kind || capability != entry.capability {
              raise InvalidCatalog(AotFamilySemanticMismatch)
            }
          None =>
            aot_semantics[family] = (entry.operation_kind, entry.capability)
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
) -> Bool {
  entry.semantic_version == KernelSemanticVersion::v1() &&
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
  target : DeviceTarget,
) -> CatalogEntry raise CatalogError {
  let semantic : Array[CatalogEntry] = []
  let architecture : Array[CatalogEntry] = []
  let compatible : Array[CatalogEntry] = []
  for entry in catalog.entries {
    if semantic_match(entry, operation) {
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
  // Catalog semantic version 1 describes only the stateless full-context
  // operation ABI. Paged plans must wait for entries that name their runtime
  // metadata and persistent KV operands instead of matching by shape alone.
  if model_plan.kv_execution() != StatelessFullContext {
    raise UnsupportedExecutionContract
  }
  let bindings : Array[KernelBinding] = []
  let mut max_workspace_bytes = 0L
  let mut max_workspace_alignment = 1L
  let workspace_limit = model_plan.workspace_bounds().max_temporary_bytes()
  for operation in model_plan.operations() {
    let entry = select_entry(self, operation, target)
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
