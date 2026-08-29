///|
fn find_exact_contract(
  candidate : PointwiseCudaLowering,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> @launch_contract.PagedKvAotLaunchContract raise PointwiseAotError {
  let mut found : @launch_contract.PagedKvAotLaunchContract? = None
  for contract in contracts.contracts() {
    if contract.operation_id() == candidate.operation_id &&
      contract.profile() == candidate.profile {
      if found is Some(_) {
        raise Mismatch(Operation)
      }
      found = Some(contract)
    }
  }
  guard found is Some(contract) else { raise Mismatch(Operation) }
  contract
}

///|
fn manifest_binding_record(
  candidate : PointwiseCudaLowering,
  contract : @launch_contract.PagedKvAotLaunchContract,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> Bytes {
  let entry = contract.entry_point()
  @utf8.encode(
    "schema=lunaflux-luna-cuda-pointwise-manifest-binding-v1\n" +
    "family=" +
    family_tag(candidate.family) +
    "\n" +
    "operation_id=" +
    candidate.operation_id.as_int().to_string() +
    "\n" +
    "profile_id=" +
    candidate.profile.id().as_int().to_string() +
    "\n" +
    "catalog_version=" +
    contracts.catalog_version().as_int().to_string() +
    "\n" +
    "target=sm_" +
    candidate.target.compute_major().to_string() +
    candidate.target.compute_minor().to_string() +
    "\n" +
    "module_sha256=" +
    entry.family().module_digest().sha256() +
    "\n" +
    "family_id=" +
    entry.family().family_id().as_int().to_string() +
    "\n" +
    "entry_point_id=" +
    entry.entry_point_id().as_int().to_string() +
    "\n" +
    "function_symbol=" +
    candidate.function_symbol +
    "\n" +
    "source_sha256=" +
    candidate.source_digest.as_hex() +
    "\n" +
    "recipe_sha256=" +
    candidate.recipe_digest.as_hex() +
    "\n" +
    "grid=" +
    candidate.dimensions.grid_x().to_string() +
    "," +
    candidate.dimensions.grid_y().to_string() +
    "," +
    candidate.dimensions.grid_z().to_string() +
    "\n" +
    "block=" +
    candidate.dimensions.block_x().to_string() +
    "," +
    candidate.dimensions.block_y().to_string() +
    "," +
    candidate.dimensions.block_z().to_string() +
    "\n" +
    "operands=" +
    operand_recipe(candidate.operands) +
    "\n" +
    "manifest_bindable=true\n",
  )
}

///|
/// Joins one non-circular source candidate to the exact admitted full-graph
/// contract after the content-addressed module family exists.
///
/// No raw entry point, family, module, dimensions, or operand list is accepted
/// by this boundary. Every such value comes from `contracts` and must equal the
/// candidate's independently derived typed evidence.
pub fn bind_manifest_pointwise_cuda_aot(
  candidate : PointwiseCudaLowering,
  operation : @plan.PlanOperation,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> PointwiseManifestCudaLowering raise PointwiseAotError {
  if candidate.family == KvCacheWrite || candidate.manifest_bindable {
    raise Unsupported(Operation)
  }
  let parameters = pointwise_parameters(operation)
  if parameters.family != candidate.family ||
    operation.id() != candidate.operation_id ||
    contracts.scope() != FullGraph ||
    contracts.catalog_version() != @catalog.CatalogVersion::v3() ||
    contracts.device_target() != candidate.target {
    raise Mismatch(Operation)
  }
  let contract = find_exact_contract(candidate, contracts)
  if contract.operation_kind() != operation.kind() ||
    contract.operation_shape() != operation.shape() ||
    contract.capability() != operation.required_capability() ||
    contract.semantic_version() != @catalog.KernelSemanticVersion::v3() ||
    contract.kv_layer() != operation.kv_layer() ||
    contract.runtime_inputs() != operation.runtime_inputs() ||
    contract.persistent_state() != operation.persistent_state() ||
    contract.workspace_bytes() != operation.workspace_bytes() ||
    contract.dimensions() != candidate.dimensions ||
    contract.operands() != candidate.operands ||
    contract.entry_point().entry_point_id() != candidate.entry_point_id {
    raise Mismatch(Operation)
  }
  validate_operands(
    operation,
    parameters,
    candidate.profile,
    contract.operands(),
  )
  let binding_record = manifest_binding_record(candidate, contract, contracts)
  {
    candidate,
    entry_point: contract.entry_point(),
    binding_record,
    binding_digest: PointwiseManifestBindingDigest(
      pointwise_sha256(binding_record),
    ),
  }
}
