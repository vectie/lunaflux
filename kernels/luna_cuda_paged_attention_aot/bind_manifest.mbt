///|
fn paged_attention_find_exact_contract(
  candidate : PagedAttentionCudaLowering,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> @launch_contract.PagedKvAotLaunchContract raise PagedAttentionAotError {
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
fn paged_attention_operand_recipe(
  operands : ArrayView[@launch_contract.AotOperand],
) -> String {
  let result = StringBuilder()
  for index, operand in operands {
    if index > 0 {
      result.write_char(',')
    }
    result.write_string(
      match operand.role() {
        RuntimeInput(StepCounts) => "runtime:step_counts"
        RuntimeInput(QueryPositions) => "runtime:query_positions"
        RuntimeInput(QueryRowOffsets) => "runtime:query_row_offsets"
        RuntimeInput(SequenceLengths) => "runtime:sequence_lengths"
        RuntimeInput(PageTableOffsets) => "runtime:page_table_offsets"
        RuntimeInput(PageTablePageIndices) => "runtime:page_table_page_indices"
        ActivationInput(reference) =>
          "activation_in:" + reference.as_int().to_string()
        ActivationOutput(reference) =>
          "activation_out:" + reference.as_int().to_string()
        PersistentKvComponent(layer, component) =>
          "kv:" +
          layer.as_int().to_string() +
          ":" +
          (match component {
            Key => "key"
            Value => "value"
          })
        ExternalInput(TokenIds) => "external:token_ids"
        WeightInput(reference) => "weight:" + reference.as_int().to_string()
        Workspace => "workspace"
        WeightScaleInput(reference, scale) =>
          "weight_scale:" +
          reference.as_int().to_string() +
          ":" +
          scale.tensor().as_int().to_string()
      },
    )
    result.write_char('@')
    result.write_string(operand.byte_count().to_string())
    result.write_char('/')
    result.write_string(operand.alignment().to_string())
  }
  result.to_string()
}

///|
fn paged_attention_manifest_binding_record(
  candidate : PagedAttentionCudaLowering,
  contract : @launch_contract.PagedKvAotLaunchContract,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> Bytes {
  let entry = contract.entry_point()
  @utf8.encode(
    "schema=lunaflux-paged-attention-manifest-binding-v1\n" +
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
    paged_attention_operand_recipe(candidate.operands) +
    "\n" +
    "manifest_bindable=true\n",
  )
}

///|
/// Joins a deterministic attention candidate to the exact full-graph
/// contract only after the content-addressed module is catalog authority.
pub fn bind_manifest_paged_attention_cuda_aot(
  candidate : PagedAttentionCudaLowering,
  operation : @plan.PlanOperation,
  contracts : @launch_contract.PagedKvLaunchContractSet,
) -> PagedAttentionManifestCudaLowering raise PagedAttentionAotError {
  if operation.id() != candidate.operation_id ||
    contracts.scope() != FullGraph ||
    contracts.catalog_version() != @catalog.CatalogVersion::v3() ||
    contracts.device_target() != candidate.target {
    raise Mismatch(Operation)
  }
  let contract = paged_attention_find_exact_contract(candidate, contracts)
  let layout = contracts.device_kv_layout()
  let (_, _, _, page_stride, page_capacity) = paged_attention_validate_candidate(
    operation,
    candidate.profile,
    layout,
    contract.operands(),
    candidate.target,
    candidate.compiler,
  )
  if contract.operation_kind() != operation.kind() ||
    contract.operation_shape() != operation.shape() ||
    contract.capability() != operation.required_capability() ||
    contract.semantic_version() != @catalog.KernelSemanticVersion::v3() ||
    contract.kv_layer() != operation.kv_layer() ||
    contract.runtime_inputs() != operation.runtime_inputs() ||
    contract.persistent_state() != operation.persistent_state() ||
    contract.workspace_bytes() != operation.workspace_bytes() ||
    contract.workspace_alignment() <= 0L ||
    contract.profile() != candidate.profile ||
    contract.dimensions() != candidate.dimensions ||
    contract.operands() != candidate.operands ||
    contract.tokens_per_page() != candidate.tokens_per_page ||
    page_stride != candidate.page_stride_bytes ||
    page_capacity != candidate.physical_page_capacity ||
    contract.entry_point().entry_point_id() != candidate.entry_point_id {
    raise Mismatch(Contract)
  }
  let binding_record = paged_attention_manifest_binding_record(
    candidate, contract, contracts,
  )
  {
    candidate,
    entry_point: contract.entry_point(),
    binding_record,
    binding_digest: PagedAttentionManifestBindingDigest(
      paged_attention_sha256(binding_record),
    ),
  }
}
