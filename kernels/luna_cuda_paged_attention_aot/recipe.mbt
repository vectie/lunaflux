///|
fn paged_attention_bool(value : Bool) -> String {
  if value {
    "true"
  } else {
    "false"
  }
}

///|
fn paged_attention_recipe(
  operation : @plan.PlanOperation,
  profile : @launch_contract.PagedKernelProfile,
  entry_point_id : @catalog.AotKernelEntryPointId,
  tokens_per_page : Int,
  operands : ArrayView[@launch_contract.AotOperand],
  dimensions : @launch_contract.AotLaunchDimensions,
  target : @catalog.DeviceTarget,
  compiler : @luna_specialization.LunaCompilerPolicy,
  source_digest : PagedAttentionCudaSourceDigest,
  function_symbol : String,
  query_heads : Int,
  kv_heads : Int,
  head_dimension : Int,
  page_stride_bytes : Int64,
  physical_page_capacity : Int64,
) -> Bytes {
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-paged-attention-cuda-aot-recipe-v1\n" +
    "mode=mixed-prefill-decode\n" +
    "source_sha256=" +
    source_digest.as_hex() +
    "\n" +
    "toolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\n" +
    "compiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\n" +
    "target=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\n" +
    "output=cubin\n" +
    "language_standard=c++17\n" +
    "optimization=" +
    compiler.optimization_level().to_string() +
    "\n" +
    "fmad=" +
    paged_attention_bool(compiler.fused_multiply_add()) +
    "\n" +
    "reassociate=false\n" +
    "max_registers=" +
    compiler.max_registers().to_string() +
    "\n" +
    "operation_id=" +
    operation.id().as_int().to_string() +
    "\n" +
    "entry_point_id=" +
    entry_point_id.as_int().to_string() +
    "\n" +
    "function_symbol=" +
    function_symbol +
    "\n" +
    "shape=" +
    query_heads.to_string() +
    "," +
    kv_heads.to_string() +
    "," +
    head_dimension.to_string() +
    "\n" +
    "tokens_per_page=" +
    tokens_per_page.to_string() +
    "\n" +
    "page_stride_bytes=" +
    page_stride_bytes.to_string() +
    "\n" +
    "physical_page_capacity=" +
    physical_page_capacity.to_string() +
    "\n" +
    "profile=" +
    profile.max_query_rows().to_string() +
    "," +
    profile.max_query_tokens().to_string() +
    "," +
    profile.max_page_table_entries().to_string() +
    "\n" +
    "grid=" +
    dimensions.grid_x().to_string() +
    "," +
    dimensions.grid_y().to_string() +
    ",1\n" +
    "block=" +
    dimensions.block_x().to_string() +
    ",1,1\n" +
    "shared_memory_bytes=" +
    dimensions.shared_memory_bytes().to_string() +
    "\n" +
    "manifest_bindable=false\n" +
    "abi=step_counts,query_positions,query_row_offsets,sequence_lengths," +
    "page_table_offsets,page_table_page_indices,qkv,output,key_cache,value_cache\n" +
    "operands=" +
    paged_attention_operand_recipe(operands) +
    "\n" +
    "numerics=bf16-input-fp32-accum-online-softmax-warp-tiled-v3\n" +
    "kv_write=fused-current-token\n" +
    "in_chunk_kv_source=qkv-activation\n",
  )
}
