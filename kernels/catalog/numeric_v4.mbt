///|
/// Constructs one catalog-v4 paged entry bound to an exact numeric ABI.
pub fn CatalogEntry::new_paged_v4(
  operation_kind~ : @plan.OperationKind,
  operation_shape~ : @plan.OperationShape,
  capability~ : @plan.KernelCapabilityId,
  execution_semantics~ : KernelExecutionSemantics,
  operation_execution_digest~ : @numeric_contract.OperationExecutionDigest,
  aot_entry_point~ : AotKernelEntryPoint,
  compute_major~ : Int,
  compute_minor~ : Int,
  implementation~ : KernelImplementationId,
  workspace_bytes~ : Int64,
  workspace_alignment~ : Int64,
) -> CatalogEntry raise CatalogError {
  let semantic_version = KernelSemanticVersion::v4()
  if compute_major <= 0 || compute_minor < 0 {
    raise InvalidCatalog(InvalidComputeCapability)
  }
  if workspace_bytes < 0L {
    raise InvalidCatalog(InvalidWorkspaceBytes)
  }
  if !is_power_of_two(workspace_alignment) || workspace_alignment > 65536L {
    raise InvalidCatalog(InvalidWorkspaceAlignment)
  }
  if !operation_shape_is_legal(operation_kind, operation_shape) {
    raise InvalidCatalog(OperationShapeMismatch)
  }
  if !operation_capability_is_legal(
      semantic_version, operation_kind, capability,
    ) {
    raise InvalidCatalog(OperationCapabilityMismatch)
  }
  if !paged_semantics_are_legal(operation_kind, execution_semantics) {
    raise InvalidCatalog(InvalidExecutionSemantics)
  }
  if implementation is VendorBf16GemmImplicitHeuristicV1 {
    raise InvalidCatalog(InvalidVendorImplementationKind)
  }
  guard implementation is ContentAddressedAot(family) else {
    raise InvalidCatalog(InvalidVendorImplementationKind)
  }
  if aot_entry_point.family() != family {
    raise InvalidCatalog(AotEntryPointFamilyMismatch)
  }
  if !vendor_implementation_is_legal(operation_kind, implementation) {
    raise InvalidCatalog(InvalidVendorImplementationKind)
  }
  if !vendor_workspace_is_legal(
      implementation, workspace_bytes, workspace_alignment,
    ) {
    raise InvalidCatalog(InvalidWorkspaceAlignment)
  }
  {
    semantic_version,
    operation_kind,
    operation_shape,
    capability,
    compute_major,
    compute_minor,
    implementation,
    workspace_bytes,
    workspace_alignment,
    execution_semantics: Some(execution_semantics),
    operation_execution_digest: Some(operation_execution_digest),
    aot_entry_point: Some(aot_entry_point),
  }
}

///|
fn operation_numeric_execution(
  model_plan : @plan.ModelPlan,
  operation_id : @plan.OperationId,
) -> @numeric_contract.OperationExecutionContract? {
  for binding in model_plan.numeric_binding().operations() {
    if binding.operation() == operation_id {
      return Some(binding.execution())
    }
  }
  None
}

///|
fn require_operation_execution_digest(
  model_plan : @plan.ModelPlan,
  operation_id : @plan.OperationId,
) -> @numeric_contract.OperationExecutionDigest raise CatalogError {
  guard operation_numeric_execution(model_plan, operation_id) is Some(execution) else {
    raise MissingKernel(operation_id)
  }
  execution.digest()
}

///|
fn execution_contract_uses_i8_or_integer_accumulator(
  execution : @numeric_contract.OperationExecutionContract,
) -> Bool {
  let i8 = @numeric_contract.ComputeDType::i8()
  execution.activation_input() == i8 ||
  execution.activation_compute() == i8 ||
  execution.tensor_input() == i8 ||
  execution.accumulator() == @numeric_contract.AccumulatorDType::i32() ||
  execution.output() == i8
}

///|
fn reject_legacy_i8_execution_contracts(
  model_plan : @plan.ModelPlan,
) -> Unit raise CatalogError {
  for binding in model_plan.numeric_binding().operations() {
    if execution_contract_uses_i8_or_integer_accumulator(binding.execution()) {
      raise UnsupportedNumericContract(binding.operation())
    }
  }
}

///|
/// Resolves catalog v4 against paged execution and exact numeric operation
/// digests. This result remains inert catalog data; it opens no module and
/// constructs no launch contract.
pub fn KernelCatalog::resolve_paged_v4(
  self : KernelCatalog,
  model_plan : @plan.ModelPlan,
  device_kv_layout : @device_layout.DeviceKvLayout,
  target : DeviceTarget,
) -> ResolvedKernelCatalog raise CatalogError {
  if self.version != CatalogVersion::v4() ||
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
