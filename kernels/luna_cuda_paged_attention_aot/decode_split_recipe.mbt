///|
fn paged_attention_decode_split_recipe(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  partial_entry_point_id : @catalog.AotKernelEntryPointId,
  merge_entry_point_id : @catalog.AotKernelEntryPointId,
  partial_function_symbol : String,
  merge_function_symbol : String,
  partial_dimensions : @launch_contract.AotLaunchDimensions,
  merge_dimensions : @launch_contract.AotLaunchDimensions,
  source_digest : PagedAttentionCudaSourceDigest,
  workspace : @luna_attention_strategy.AttentionDecodeWorkspacePlan,
  compiler_readonly? : Bool = false,
) -> Bytes {
  let identity = compiler.identity()
  let (maximum_offset, denominator_offset, numerator_offset) = workspace.region_offsets_bytes()
  let partitions = StringBuilder()
  for index, partition in workspace.partitions() {
    let (begin_token, end_token) = partition.token_range()
    if index > 0 {
      partitions.write_char(',')
    }
    partitions.write_string(partition.split_index().to_string())
    partitions.write_char(':')
    partitions.write_string(begin_token.to_string())
    partitions.write_char('-')
    partitions.write_string(end_token.to_string())
  }
  @utf8.encode(
    "schema=lunaflux-paged-attention-decode-split-cuda-aot-recipe-v1\n" +
    "manifest_bindable=false\n" +
    "reason=requires-versioned-two-stage-launch-and-workspace-contract\n" +
    "source_sha256=" +
    source_digest.as_hex() +
    "\ntoolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\ncompiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\noperation_id=" +
    operation.id().as_int().to_string() +
    "\nprofile=" +
    profile.max_query_rows().to_string() +
    "," +
    profile.max_query_tokens().to_string() +
    "," +
    profile.max_page_table_entries().to_string() +
    "\npartial_entry_point_id=" +
    partial_entry_point_id.as_int().to_string() +
    "\nmerge_entry_point_id=" +
    merge_entry_point_id.as_int().to_string() +
    "\npartial_function_symbol=" +
    partial_function_symbol +
    "\nmerge_function_symbol=" +
    merge_function_symbol +
    "\npartial_grid=" +
    partial_dimensions.grid_x().to_string() +
    "," +
    partial_dimensions.grid_y().to_string() +
    "," +
    partial_dimensions.grid_z().to_string() +
    "\nmerge_grid=" +
    merge_dimensions.grid_x().to_string() +
    "," +
    merge_dimensions.grid_y().to_string() +
    "," +
    merge_dimensions.grid_z().to_string() +
    "\nblock=" +
    partial_dimensions.block_x().to_string() +
    ",1,1\nworkspace_scalar=f32\nworkspace_partial_count=" +
    workspace.partial_count().to_string() +
    "\nworkspace_offsets_bytes=" +
    maximum_offset.to_string() +
    "," +
    denominator_offset.to_string() +
    "," +
    numerator_offset.to_string() +
    "\nworkspace_total_bytes=" +
    workspace.total_bytes().to_string() +
    "\npartition_mode=" +
    (if compiler_readonly {
      "live-tile-balanced"
    } else {
      "live-page-balanced"
    }) +
    "\nmaximum_partition_envelope=" +
    partitions.to_string() +
    "\npartial_abi=step_counts,query_positions,query_row_offsets,sequence_lengths," +
    "page_table_offsets,page_table_page_indices,qkv,key_cache,value_cache,workspace\n" +
    "merge_abi=step_counts,query_row_offsets,workspace,output\n" +
    "merge_order=ascending-split-index\n" +
    "numerics=bf16-input-fp32-partial-online-softmax-deterministic-fp32-merge\n" +
    "fmad=" +
    paged_attention_bool(compiler.fused_multiply_add()) +
    "\nreassociate=false\n",
  )
}
