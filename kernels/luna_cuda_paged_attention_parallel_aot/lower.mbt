///|
fn parallel_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn parallel_recipe(
  reference : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
  entry_point_id : @catalog.AotKernelEntryPointId,
  dimensions : @launch_contract.AotLaunchDimensions,
  maximum_context_tokens : Int,
  tile_program_digest : @luna_tile_ir.LunaTileProgramDigest,
  tile_aot_digest : @luna_tile_ir.LunaTileCudaAotDigest,
  numerical : PagedAttentionParallelNumericalPolicy,
  source_digest : PagedAttentionParallelDigest,
  symbol : String,
) -> Bytes {
  let target = reference.target()
  let compiler = reference.compiler_policy()
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-paged-attention-parallel-cuda-aot-candidate-v1\n" +
    "family=paged_attention_parallel_block128\n" +
    "mode=mixed-prefill-decode\n" +
    "target=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\nentry_point_id=" +
    entry_point_id.as_int().to_string() +
    "\nfunction_symbol=" +
    symbol +
    "\nsource_sha256=" +
    source_digest.as_hex() +
    "\noutput=cubin\nlanguage_standard=c++17" +
    "\nreference_fallback_source_sha256=" +
    reference.source_digest().as_hex() +
    "\nreference_fallback_recipe_sha256=" +
    reference.recipe_digest().as_hex() +
    "\nlunatile_program_sha256=" +
    tile_program_digest.as_hex() +
    "\nlunatile_aot_sha256=" +
    tile_aot_digest.as_hex() +
    "\ntoolchain_sha256=" +
    identity.toolchain_digest_sha256() +
    "\ncompiler_version=" +
    identity.major().to_string() +
    "." +
    identity.minor().to_string() +
    "." +
    identity.patch().to_string() +
    "\noptimization=" +
    compiler.optimization_level().to_string() +
    "\nfmad=false\nreassociate=false\nmax_registers=" +
    compiler.max_registers().to_string() +
    "\nshape=" +
    reference.query_heads().to_string() +
    "," +
    reference.kv_heads().to_string() +
    "," +
    reference.head_dimension().to_string() +
    "\ntokens_per_page=" +
    reference.tokens_per_page().to_string() +
    "\nmaximum_context_tokens=" +
    maximum_context_tokens.to_string() +
    "\ngrid=" +
    dimensions.grid_x().to_string() +
    "," +
    dimensions.grid_y().to_string() +
    ",1\nblock=128,1,1\nshared_memory_bytes=" +
    dimensions.shared_memory_bytes().to_string() +
    "\nnumerics=" +
    numerical.name() +
    "\nmaximum_absolute_error_ppb=" +
    numerical.maximum_absolute_error_ppb().to_string() +
    "\nmaximum_relative_error_ppb=" +
    numerical.maximum_relative_error_ppb().to_string() +
    "\nrequires_token_agreement=true\ndispatch_canary=parallel-block128-symbol-v1\n" +
    "release_binding=candidate-only-promotion-required\nmanifest_bindable=false\n" +
    "abi=step_counts,query_positions,query_row_offsets,sequence_lengths," +
    "page_table_offsets,page_table_page_indices,qkv,output,key_cache,value_cache\n",
  )
}

///|
/// Derives one genuinely block-parallel candidate from an independently
/// admitted serial reference lowering. No operation, layout, compiler, or
/// operand authority is reconstructed from strings.
pub fn lower_paged_attention_parallel_cuda_aot_candidate(
  reference : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
  limits : PagedAttentionParallelAotLimits,
) -> PagedAttentionParallelCudaCandidate raise PagedAttentionParallelAotError {
  let target = reference.target()
  let profile = reference.profile()
  let compiler = reference.compiler_policy()
  let head_dimension = reference.head_dimension()
  let tokens_per_page = reference.tokens_per_page()
  if target.compute_major() < 8 ||
    target.compute_major() > 12 ||
    target.compute_minor() < 0 ||
    target.compute_minor() > 9 ||
    !target.supports_bf16() {
    raise PagedAttentionParallelUnsupported(DeviceTarget)
  }
  if head_dimension < 32 ||
    head_dimension > 128 ||
    head_dimension % 16 != 0 ||
    reference.query_heads() <= 0 ||
    reference.kv_heads() <= 0 ||
    reference.query_heads() % reference.kv_heads() != 0 {
    raise PagedAttentionParallelUnsupported(Shape)
  }
  if !parallel_is_power_of_two(tokens_per_page) ||
    tokens_per_page < 8 ||
    tokens_per_page > 256 ||
    reference.page_stride_bytes() <= 0L ||
    reference.physical_page_capacity() <= 0L {
    raise PagedAttentionParallelUnsupported(KvLayout)
  }
  if profile.max_query_rows() <= 0 ||
    profile.max_query_rows() > 1024 ||
    profile.max_query_tokens() < profile.max_query_rows() ||
    profile.max_page_table_entries() <= 0 {
    raise PagedAttentionParallelUnsupported(Profile)
  }
  if compiler.optimization_level() != 3 ||
    compiler.fused_multiply_add() ||
    compiler.reassociate() ||
    compiler.max_registers() < 64 {
    raise PagedAttentionParallelUnsupported(CompilerPolicy)
  }
  let operands = reference.operands()
  if operands.length() != 10 ||
    operands[6].alignment() < 16L ||
    operands[7].alignment() < 16L ||
    operands[8].alignment() < 256L ||
    operands[9].alignment() < 256L {
    raise PagedAttentionParallelUnsupported(StaticResources)
  }
  let maximum_context_tokens = parallel_checked_product(
    profile.max_page_table_entries(),
    tokens_per_page,
  )
  if maximum_context_tokens > PAGED_ATTENTION_PARALLEL_MAX_CONTEXT {
    raise PagedAttentionParallelUnsupported(Profile)
  }
  let shared_memory_bytes = parallel_checked_product(
    maximum_context_tokens + PAGED_ATTENTION_PARALLEL_BLOCK_THREADS,
    4,
  )
  if shared_memory_bytes > PAGED_ATTENTION_PARALLEL_MAX_SHARED_BYTES {
    raise PagedAttentionParallelUnsupported(StaticResources)
  }
  let (tile_program, tile_aot) = parallel_tile_aot(
    maximum_context_tokens,
    shared_memory_bytes,
    limits.max_canonical_bytes,
  )
  let entry_point_id = @catalog.AotKernelEntryPointId::from_stable_int(1001) catch {
    _ => raise PagedAttentionParallelInvalid(ReferenceCandidate)
  }
  let dimensions = @launch_contract.AotLaunchDimensions::new(
    grid_x=profile.max_query_tokens(),
    grid_y=reference.query_heads(),
    grid_z=1,
    block_x=PAGED_ATTENTION_PARALLEL_BLOCK_THREADS,
    block_y=1,
    block_z=1,
    shared_memory_bytes~,
  ) catch {
    _ => raise PagedAttentionParallelUnsupported(StaticResources)
  }
  let symbol = "lunaflux_paged_attention_bf16_parallel_strict_v1_ep_1001"
  let source = parallel_cuda_source(symbol, reference, maximum_context_tokens)
  if source.length() > limits.max_source_bytes {
    raise PagedAttentionParallelLimitExceeded(Source)
  }
  let source_digest = PagedAttentionParallelDigest(parallel_sha256(source))
  let numerical_policy = paged_attention_parallel_numerical_policy()
  let recipe = parallel_recipe(
    reference,
    entry_point_id,
    dimensions,
    maximum_context_tokens,
    tile_program.digest(),
    tile_aot.digest(),
    numerical_policy,
    source_digest,
    symbol,
  )
  if recipe.length() > limits.max_recipe_bytes {
    raise PagedAttentionParallelLimitExceeded(Recipe)
  }
  let recipe_digest = PagedAttentionParallelDigest(parallel_sha256(recipe))
  let candidate_digest = PagedAttentionParallelDigest(
    parallel_sha256(
      @utf8.encode(
        "lunaflux.paged-attention-parallel-candidate.v1\n" +
        reference.source_digest().as_hex() +
        "\n" +
        source_digest.as_hex() +
        "\n" +
        recipe_digest.as_hex() +
        "\n" +
        tile_program.digest().as_hex() +
        "\n" +
        tile_aot.digest().as_hex() +
        "\n",
      ),
    ),
  )
  {
    reference,
    entry_point_id,
    dimensions,
    maximum_context_tokens,
    shared_memory_bytes,
    tile_program_digest: tile_program.digest(),
    tile_aot_digest: tile_aot.digest(),
    numerical_policy,
    function_symbol: symbol,
    source,
    source_digest,
    recipe,
    recipe_digest,
    candidate_digest,
  }
}
