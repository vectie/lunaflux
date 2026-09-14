///|
fn qwen_compiler_attention_recipe(
  operation : Int,
  family : String,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  compiled : @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation,
  frontier_sha256? : String,
  variant_stable_id? : Int,
) -> Bytes {
  let compiler = set.compiler_policy()
  let identity = compiler.identity()
  let source = compiled.source()
  let lowering = compiled.lowering()
  let candidate = compiled.functional().plan().candidate()
  let (query_tile_rows, key_value_tile_rows, head_columns) = candidate
    .tile()
    .dimensions()
  let (grid_x, grid_y, block_x, shared) = lowering.launch()
  let frontier_record = match (frontier_sha256, variant_stable_id) {
    (Some(frontier), Some(stable_id)) =>
      "\nfrontier_sha256=" +
      frontier +
      "\nvariant_stable_id=" +
      stable_id.to_string()
    _ => ""
  }
  @utf8.encode(
    "schema=lunaflux-attention-tile-compiler-cuda-aot-candidate.v1\n" +
    "family=" +
    family +
    "\n" +
    "model_operation_id=" +
    operation.to_string() +
    "\nfunction_symbol=" +
    source.symbol().to_owned() +
    "\ntarget=sm_120\ntoolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\ncompiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\nsource_sha256=" +
    source.digest().as_hex() +
    "\noptimization=" +
    compiler.optimization_level().to_string() +
    "\nfmad=" +
    compiler.fused_multiply_add().to_string() +
    "\nreassociate=" +
    compiler.reassociate().to_string() +
    "\nmax_registers=" +
    lowering.compiler_register_ceiling(compiler.max_registers()).to_string() +
    qwen_compiler_single_launch_record(grid_x, grid_y, block_x, shared) +
    "\nfunctional_sha256=" +
    compiled.functional().digest().as_hex() +
    "\nschedule_sha256=" +
    compiled.functional().schedule().digest().as_hex() +
    "\nlowering_sha256=" +
    lowering.digest().as_hex() +
    "\ncandidate_stable_id=" +
    candidate.stable_id().to_string() +
    "\nquery_tile_rows=" +
    query_tile_rows.to_string() +
    "\nquery_metadata_version=" +
    compiled.source().abi().query_metadata_version().to_string() +
    "\nkey_value_tile_rows=" +
    key_value_tile_rows.to_string() +
    "\nhead_columns=" +
    head_columns.to_string() +
    frontier_record +
    "\n",
  )
}

///|
/// The compiler's selected schedule remains authoritative for the large-query
/// alias. Merely having a wider tile is not a performance observation. Keep
/// alternatives in the exported frontier for offline measurement, without
/// silently replacing the selected plan with the largest resource footprint.
fn qwen_wide_query_prefill_variant(
  frontier : @luna_cuda_attention_tile_aot.CudaAttentionTileAotFrontier,
) -> @luna_cuda_attention_tile_aot.CudaAttentionTileAotCompilation? {
  Some(frontier.selected())
}

///|
/// Canonical launch fragment for one compiler-generated kernel. Partitioned
/// attention deliberately uses a different recipe constructor because it owns
/// two entry points and a workspace.
fn qwen_compiler_single_launch_record(
  grid_x : Int,
  grid_y : Int,
  block_x : Int,
  shared_memory_bytes : Int64,
) -> String {
  "\ngrid=" +
  grid_x.to_string() +
  "," +
  grid_y.to_string() +
  ",1\nblock=" +
  block_x.to_string() +
  ",1,1\nshared_memory_bytes=" +
  shared_memory_bytes.to_string()
}

///|
fn qwen_compiler_partitioned_attention_recipe(
  operation : Int,
  set : @luna_bf16_kernel_producer.LunaBf16CandidateSet,
  compiled : @luna_cuda_attention_tile_aot.CudaPartitionedAttentionTileAotCompilation,
  frontier_sha256~ : String,
) -> Bytes {
  let compiler = set.compiler_policy()
  let identity = compiler.identity()
  let source = compiled.source()
  let lowering = compiled.lowering()
  let (grid_x, grid_y, block_x, shared) = lowering.launch()
  let (partial_symbol, merge_symbol) = source.symbols()
  let (query_tile_rows, _, _) = compiled
    .functional()
    .plan()
    .candidate()
    .tile()
    .dimensions()
  let workspace = source.workspace()
  let (maximum_offset, denominator_offset, numerator_offset) = workspace.region_offsets_bytes()
  @utf8.encode(
    "schema=lunaflux-attention-tile-compiler-partitioned-cuda-aot-candidate.v1\n" +
    "family=paged-attention-prefill-functional-partitioned-tile\n" +
    "frontier_sha256=" +
    frontier_sha256 +
    "\nschedule_sha256=" +
    compiled.functional().schedule().digest().as_hex() +
    "\n" +
    "model_operation_id=" +
    operation.to_string() +
    "\npartial_function_symbol=" +
    partial_symbol.to_owned() +
    "\nmerge_function_symbol=" +
    merge_symbol.to_owned() +
    "\ntarget=sm_120\ntoolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\ncompiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\nsource_sha256=" +
    source.digest().as_hex() +
    "\noptimization=" +
    compiler.optimization_level().to_string() +
    "\nfmad=" +
    compiler.fused_multiply_add().to_string() +
    "\nreassociate=" +
    compiler.reassociate().to_string() +
    "\nmax_registers=" +
    lowering.compiler_register_ceiling(compiler.max_registers()).to_string() +
    qwen_compiler_partitioned_launch_record(
      grid_x,
      grid_y,
      lowering.key_value_partitions(),
      block_x,
      shared,
    ) +
    "\nworkspace_bytes=" +
    workspace.total_bytes().to_string() +
    "\nquery_metadata_version=" +
    compiled.source().abi().query_metadata_version().to_string() +
    "\nquery_tile_rows=" +
    query_tile_rows.to_string() +
    "\nworkspace_offsets=" +
    maximum_offset.to_string() +
    "," +
    denominator_offset.to_string() +
    "," +
    numerator_offset.to_string() +
    "\nfunctional_sha256=" +
    compiled.functional().digest().as_hex() +
    "\nlowering_sha256=" +
    lowering.digest().as_hex() +
    "\n",
  )
}

///|
/// Canonical launch fragment for the partial and merge entry points emitted by
/// the partitioned functional attention compilation.
fn qwen_compiler_partitioned_launch_record(
  grid_x : Int,
  grid_y : Int,
  partitions : Int,
  block_x : Int,
  partial_shared_memory_bytes : Int64,
) -> String {
  "\npartial_grid=" +
  grid_x.to_string() +
  "," +
  grid_y.to_string() +
  "," +
  partitions.to_string() +
  "\nmerge_grid=" +
  grid_x.to_string() +
  "," +
  grid_y.to_string() +
  ",1\nblock=" +
  block_x.to_string() +
  ",1,1\npartial_shared_memory_bytes=" +
  partial_shared_memory_bytes.to_string() +
  "\nmerge_shared_memory_bytes=0"
}
