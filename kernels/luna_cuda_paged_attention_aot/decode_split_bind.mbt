///|
fn decode_split_workspace_operand(
  workspace : @luna_attention_multilaunch_contract.AttentionDecodeWorkspaceContract,
) -> @launch_contract.AotOperand raise PagedAttentionAotError {
  @launch_contract.AotOperand::new(
    role=Workspace,
    byte_count=workspace.total_bytes(),
    alignment=workspace.alignment(),
  ) catch {
    _ => raise Mismatch(Contract)
  }
}

///|
fn decode_split_launch_operands(
  operands : ReadOnlyArray[@launch_contract.AotOperand],
  workspace : @luna_attention_multilaunch_contract.AttentionDecodeWorkspaceContract,
) -> (
  ReadOnlyArray[@launch_contract.AotOperand],
  ReadOnlyArray[@launch_contract.AotOperand],
) raise PagedAttentionAotError {
  if operands.length() != 10 {
    raise Mismatch(Contract)
  }
  let workspace_operand = decode_split_workspace_operand(workspace)
  let partial : Array[@launch_contract.AotOperand] = [
    operands[0],
    operands[1],
    operands[2],
    operands[3],
    operands[4],
    operands[5],
    operands[6],
    operands[8],
    operands[9],
    workspace_operand,
  ]
  let merge : Array[@launch_contract.AotOperand] = [
    operands[0],
    operands[2],
    workspace_operand,
    operands[7],
  ]
  (ReadOnlyArray::from_array(partial), ReadOnlyArray::from_array(merge))
}

///|
fn decode_split_expected_launch_operands(
  candidate : PagedAttentionDecodeSplitCudaBundle,
  workspace : @luna_attention_multilaunch_contract.AttentionDecodeWorkspaceContract,
) -> (
  ReadOnlyArray[@launch_contract.AotOperand],
  ReadOnlyArray[@launch_contract.AotOperand],
) raise PagedAttentionAotError {
  decode_split_launch_operands(candidate.operands, workspace)
}

///|
fn decode_split_binding_record(
  candidate : PagedAttentionDecodeSplitCudaBundle,
  contract : @luna_attention_multilaunch_contract.AttentionDecodeMultiLaunchContract,
) -> Bytes {
  let workspace = contract.workspace()
  let (maximum_offset, denominator_offset, numerator_offset) = workspace.region_offsets_bytes()
  let launches = contract.launches()
  let out = StringBuilder()
  out.write_string(
    "schema=lunaflux-paged-attention-decode-split-manifest-binding-v1\n",
  )
  out.write_string(
    "operation_id=" + candidate.operation_id.as_int().to_string() + "\n",
  )
  out.write_string(
    "profile_id=" + candidate.profile.id().as_int().to_string() + "\n",
  )
  out.write_string(
    "target=sm_" +
    candidate.target.compute_major().to_string() +
    candidate.target.compute_minor().to_string() +
    "\n",
  )
  out.write_string("source_sha256=" + candidate.source_digest.as_hex() + "\n")
  out.write_string("recipe_sha256=" + candidate.recipe_digest.as_hex() + "\n")
  out.write_string("launch_count=2\n")
  for launch in launches {
    let prefix = "launch_" + launch.ordinal().to_string() + "_"
    out.write_string(
      prefix +
      "stage=" +
      (match launch.stage() {
        Partial => "partial"
        Merge => "merge"
      }) +
      "\n",
    )
    out.write_string(
      prefix + "module_sha256=" + launch.module_digest().sha256() + "\n",
    )
    out.write_string(
      prefix +
      "family_id=" +
      launch.entry_point().family().family_id().as_int().to_string() +
      "\n",
    )
    out.write_string(
      prefix +
      "entry_point_id=" +
      launch.entry_point().entry_point_id().as_int().to_string() +
      "\n",
    )
    out.write_string(
      prefix + "function_symbol=" + launch.function_symbol() + "\n",
    )
    let dimensions = launch.dimensions()
    out.write_string(
      prefix +
      "grid=" +
      dimensions.grid_x().to_string() +
      "," +
      dimensions.grid_y().to_string() +
      "," +
      dimensions.grid_z().to_string() +
      "\n",
    )
    out.write_string(
      prefix +
      "block=" +
      dimensions.block_x().to_string() +
      "," +
      dimensions.block_y().to_string() +
      "," +
      dimensions.block_z().to_string() +
      "\n",
    )
    out.write_string(
      prefix +
      "shared_memory_bytes=" +
      dimensions.shared_memory_bytes().to_string() +
      "\n",
    )
    out.write_string(
      prefix +
      "operands=" +
      paged_attention_operand_recipe(launch.operands()) +
      "\n",
    )
  }
  out.write_string("workspace_scalar=f32\n")
  out.write_string(
    "workspace_split_count=" + workspace.split_count().to_string() + "\n",
  )
  out.write_string(
    "workspace_partial_count=" + workspace.partial_count().to_string() + "\n",
  )
  out.write_string(
    "workspace_offsets_bytes=" +
    maximum_offset.to_string() +
    "," +
    denominator_offset.to_string() +
    "," +
    numerator_offset.to_string() +
    "\n",
  )
  out.write_string(
    "workspace=" +
    workspace.total_bytes().to_string() +
    "/" +
    workspace.alignment().to_string() +
    "\n",
  )
  out.write_string("launch_order=partial,merge\nmanifest_bindable=true\n")
  @utf8.encode(out.to_string())
}

///|
/// Binds the candidate to one exact content-addressed module, two ordered
/// symbols and the bounded F32 partial workspace at startup. No module loading
/// or request-path validation occurs here.
pub fn bind_manifest_paged_attention_decode_split_cuda_aot(
  candidate : PagedAttentionDecodeSplitCudaBundle,
  operation : @plan.PlanOperation,
  contract : @luna_attention_multilaunch_contract.AttentionDecodeMultiLaunchContract,
  module_bytes : Bytes,
) -> PagedAttentionDecodeSplitManifestCudaBinding raise PagedAttentionAotError {
  if module_bytes.is_empty() || module_bytes.length() > 8388608 {
    raise LimitExceeded(Source)
  }
  if operation.id() != candidate.operation_id ||
    contract.operation_id() != candidate.operation_id ||
    contract.profile() != candidate.profile ||
    contract.target() != candidate.target {
    raise Mismatch(Operation)
  }
  let (query_heads, kv_heads, head_dimension, page_stride, page_capacity) = paged_attention_validate_candidate(
    operation,
    candidate.profile,
    candidate.layout,
    candidate.operands,
    candidate.target,
    candidate.compiler,
  )
  if query_heads != candidate.query_heads ||
    kv_heads != candidate.kv_heads ||
    head_dimension != candidate.head_dimension ||
    page_stride != candidate.page_stride_bytes ||
    page_capacity != candidate.physical_page_capacity {
    raise Mismatch(Contract)
  }
  let workspace = contract.workspace()
  let candidate_workspace = candidate.workspace_plan
  if workspace.scalar() != F32 ||
    workspace.split_count() != candidate_workspace.split_count() ||
    workspace.partial_count() != candidate_workspace.partial_count() ||
    workspace.region_offsets_bytes() !=
    candidate_workspace.region_offsets_bytes() ||
    workspace.total_bytes() != candidate_workspace.total_bytes() {
    raise Mismatch(Contract)
  }
  let launches = contract.launches()
  if launches.length() != 2 ||
    launches[0].ordinal() != 0 ||
    launches[0].stage() != Partial ||
    launches[1].ordinal() != 1 ||
    launches[1].stage() != Merge ||
    launches[0].entry_point().entry_point_id() !=
    candidate.partial_entry_point_id ||
    launches[1].entry_point().entry_point_id() != candidate.merge_entry_point_id ||
    launches[0].function_symbol() != candidate.partial_function_symbol ||
    launches[1].function_symbol() != candidate.merge_function_symbol ||
    launches[0].dimensions() != candidate.partial_dimensions ||
    launches[1].dimensions() != candidate.merge_dimensions {
    raise Mismatch(Contract)
  }
  if launches[0].module_digest().sha256() !=
    paged_attention_sha256(module_bytes) {
    raise Mismatch(Contract)
  }
  let (partial_operands, merge_operands) = decode_split_expected_launch_operands(
    candidate, workspace,
  )
  if launches[0].operands() != partial_operands ||
    launches[1].operands() != merge_operands {
    raise Mismatch(Contract)
  }
  let binding_record = decode_split_binding_record(candidate, contract)
  {
    candidate,
    contract,
    module_bytes,
    binding_record,
    binding_digest: PagedAttentionManifestBindingDigest(
      paged_attention_sha256(binding_record),
    ),
  }
}
