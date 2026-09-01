///|
fn projection_find_exact_contract(
  candidate : ProjectionCudaLowering,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> @launch_contract.PagedKvAotLaunchContract raise ProjectionAotError {
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
fn projection_manifest_binding_record(
  candidate : ProjectionCudaLowering,
  contract : @launch_contract.PagedKvAotLaunchContract,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> Bytes {
  let entry = contract.entry_point()
  @utf8.encode(
    "schema=lunaflux-luna-cuda-projection-manifest-binding-v1\n" +
    "family=" +
    projection_family_tag(candidate.family) +
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
    "workspace=" +
    contract.workspace_bytes().to_string() +
    "/" +
    contract.workspace_alignment().to_string() +
    "\n" +
    "operands=" +
    projection_operand_recipe(candidate.operands) +
    "\n" +
    "manifest_bindable=true\n",
  )
}

///|
/// Joins a source candidate to the exact full-graph contract only after the
/// compiled module digest has become catalog authority.
pub fn bind_manifest_projection_cuda_aot(
  candidate : ProjectionCudaLowering,
  operation : @plan.PlanOperation,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> ProjectionManifestCudaLowering raise ProjectionAotError {
  let operation_parameters = projection_parameters(operation)
  if operation_parameters.family != candidate.family ||
    operation.id() != candidate.operation_id ||
    contracts.scope() != FullGraph ||
    contracts.catalog_version() != @catalog.CatalogVersion::v3() ||
    contracts.device_target() != candidate.target {
    raise Mismatch(Operation)
  }
  let contract = projection_find_exact_contract(candidate, contracts)
  let parameters = projection_bind_qkv_widths(
    operation,
    operation_parameters,
    contract.operands(),
  )
  validate_projection_contract(operation, contract, candidate.dispatch_plan)
  if contract.profile() != candidate.profile ||
    contract.dimensions() != candidate.dimensions ||
    contract.operands() != candidate.operands ||
    contract.entry_point().entry_point_id() != candidate.entry_point_id {
    raise Mismatch(Operation)
  }
  validate_projection_operands(
    operation,
    parameters,
    candidate.profile,
    contract.operands(),
    candidate.dispatch_plan,
  )
  let binding_record = projection_manifest_binding_record(
    candidate, contract, contracts,
  )
  {
    candidate,
    entry_point: contract.entry_point(),
    binding_record,
    binding_digest: ProjectionCudaManifestBindingDigest(
      projection_sha256(binding_record),
    ),
  }
}
