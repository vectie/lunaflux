///|
fn readonly_bool(value : Bool) -> String {
  if value {
    "true"
  } else {
    "false"
  }
}

///|
fn readonly_launch_dimensions(
  reference : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
) -> @launch_contract.AotLaunchDimensions raise PagedAttentionReadOnlyAotError {
  @launch_contract.AotLaunchDimensions::new(
    grid_x=Int::min(reference.profile().max_query_tokens(), 128),
    grid_y=reference.query_heads(),
    grid_z=1,
    block_x=128,
    block_y=1,
    block_z=1,
    shared_memory_bytes=0,
  ) catch {
    _ => raise PagedAttentionReadOnlyInvalid(Profile)
  }
}

///|
fn readonly_require_exact_reference(
  reference : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
  entry_point_id : @catalog.AotKernelEntryPointId,
) -> Unit raise PagedAttentionReadOnlyAotError {
  let target = reference.target()
  let profile = reference.profile()
  let compiler = reference.compiler_policy()
  if entry_point_id == reference.entry_point_id() {
    raise PagedAttentionReadOnlyMismatch(Symbol)
  }
  if target.compute_major() < 8 ||
    target.compute_major() > 12 ||
    target.compute_minor() < 0 ||
    target.compute_minor() > 9 ||
    !target.supports_bf16() {
    raise PagedAttentionReadOnlyInvalid(DeviceTarget)
  }
  if profile.max_query_rows() <= 0 ||
    profile.max_query_tokens() < profile.max_query_rows() ||
    profile.max_page_table_entries() <= 0 {
    raise PagedAttentionReadOnlyInvalid(Profile)
  }
  if reference.query_heads() <= 0 ||
    reference.kv_heads() <= 0 ||
    reference.head_dimension() <= 0 ||
    reference.query_heads() % reference.kv_heads() != 0 {
    raise PagedAttentionReadOnlyInvalid(Operation)
  }
  if reference.tokens_per_page() <= 0 ||
    reference.page_stride_bytes() <= 0L ||
    reference.physical_page_capacity() <= 0L {
    raise PagedAttentionReadOnlyInvalid(KvLayout)
  }
  if compiler.fused_multiply_add() ||
    compiler.reassociate() ||
    compiler.max_registers() <= 0 {
    raise PagedAttentionReadOnlyInvalid(CompilerPolicy)
  }
  let operands = reference.operands()
  let input_row_width = (reference.query_heads() + 2 * reference.kv_heads()) *
    reference.head_dimension()
  let exact_input_bytes = profile.max_query_tokens().to_int64() *
    input_row_width.to_int64() *
    2L
  if operands.length() != 10 ||
    !(operands[6].role() is ActivationInput(_)) ||
    !(operands[7].role() is ActivationOutput(_)) ||
    !(operands[8].role() is PersistentKvComponent(_, Key)) ||
    !(operands[9].role() is PersistentKvComponent(_, Value)) ||
    operands[6].byte_count() != exact_input_bytes {
    raise PagedAttentionReadOnlyMismatch(KernelAbi)
  }
  let _input_activation = match operands[6].role() {
    ActivationInput(value) => value
    _ => raise PagedAttentionReadOnlyMismatch(KernelAbi)
  }
  let fallback_recipe = @utf8.decode(reference.recipe_bytes()) catch {
    _ => raise PagedAttentionReadOnlyInvalid(Reference)
  }
  let fallback_source = @utf8.decode(reference.source_bytes()) catch {
    _ => raise PagedAttentionReadOnlyInvalid(Reference)
  }
  for
    exact in [
      "schema=lunaflux-paged-attention-cuda-aot-recipe-v1", "kv_write=fused-current-token",
      "in_chunk_kv_source=qkv-activation", "manifest_bindable=false",
    ] {
    if !fallback_recipe.contains(exact) {
      raise PagedAttentionReadOnlyMismatch(Reference)
    }
  }
  if !fallback_source.contains("key_cache[cache_index] = qkv[") ||
    !fallback_source.contains("value_cache[cache_index] = qkv[") {
    raise PagedAttentionReadOnlyMismatch(Reference)
  }
}

///|
fn readonly_recipe(
  reference : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
  entry_point_id : @catalog.AotKernelEntryPointId,
  symbol : String,
  abi : PagedAttentionReadOnlyKernelAbi,
  source_digest : PagedAttentionReadOnlyDigest,
  input_row_width : Int,
  input_activation : @plan.ActivationRef,
  dispatch_canary_per_token : Int,
  launch : @launch_contract.AotLaunchDimensions,
) -> Bytes {
  let target = reference.target()
  let profile = reference.profile()
  let compiler = reference.compiler_policy()
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-cuda-aot-candidate.v1\n" +
    "family=paged-attention-rotated-q-paged-kv-readonly\n" +
    "mode=mixed-prefill-decode-post-kvwrite\nmodel_operation_id=" +
    reference.operation_id().as_int().to_string() +
    "\nmodel_operation_kind=causal_attention\n" +
    "model_persistent_state=kv_cache_read_write\n" +
    "selection_precondition=standalone-positioned-rope-paged-kvwrite-complete-v1\n" +
    "fallback_source_sha256=" +
    reference.source_digest().as_hex() +
    "\nfallback_recipe_sha256=" +
    reference.recipe_digest().as_hex() +
    "\nsource_sha256=" +
    source_digest.as_hex() +
    "\nentry_point_id=" +
    entry_point_id.as_int().to_string() +
    "\nfunction_symbol=" +
    symbol +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
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
    "\nfmad=" +
    readonly_bool(compiler.fused_multiply_add()) +
    "\nreassociate=" +
    readonly_bool(compiler.reassociate()) +
    "\nmax_registers=" +
    compiler.max_registers().to_string() +
    "\nshape=" +
    reference.query_heads().to_string() +
    "," +
    reference.kv_heads().to_string() +
    "," +
    reference.head_dimension().to_string() +
    "\ninput_row_width=" +
    input_row_width.to_string() +
    "\ninput_activation_ref=" +
    input_activation.as_int().to_string() +
    "\nprofile_id=" +
    profile.id().as_int().to_string() +
    "\nprofile=" +
    profile.max_query_rows().to_string() +
    "," +
    profile.max_query_tokens().to_string() +
    "," +
    profile.max_page_table_entries().to_string() +
    "\ntokens_per_page=" +
    reference.tokens_per_page().to_string() +
    "\nkv_layout=layer_major_split_key_value_v1\nkv_layout_stable_version=1" +
    "\npage_stride_bytes=" +
    reference.page_stride_bytes().to_string() +
    "\nphysical_page_capacity=" +
    reference.physical_page_capacity().to_string() +
    "\ngrid=" +
    launch.grid_x().to_string() +
    "," +
    launch.grid_y().to_string() +
    ",1\nblock=" +
    launch.block_x().to_string() +
    ",1,1\nshared_memory_bytes=0\n" +
    "attention_key_tile=16\nattention_key_warps=4\nattention_keys_per_warp=4\n" +
    "attention_block_threads=128\nattention_token_grid_cap=128\n" +
    "raw_pointer_abi=" +
    abi.canonical_name() +
    "\nrotated_query_only=true\nfull_qkv_row_stride_bound=true\n" +
    "dispatch_canary_per_token=" +
    dispatch_canary_per_token.to_string() +
    "\ndispatch_canary_cell_count=" +
    profile.max_query_tokens().to_string() +
    "\ndispatch_canary_publication=exactly-once-after-output\n" +
    "kv_cache_access=read_only\nkv_cache_mutation=none\n" +
    "numerics=bf16-input-fp32-accum-stable-softmax-reference-v1\n" +
    "fallback_required=true\nqualification_only=true\nmanifest_bindable=false\n" +
    "promotion_authority=none\n",
  )
}

///|
fn readonly_production_recipe(
  qualification : PagedAttentionReadOnlyCudaCandidate,
  symbol : String,
  source_digest : PagedAttentionReadOnlyDigest,
  launch : @launch_contract.AotLaunchDimensions,
) -> Bytes {
  let reference = qualification.reference
  let target = reference.target()
  let profile = reference.profile()
  let compiler = reference.compiler_policy()
  let identity = compiler.identity()
  @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-cuda-aot-candidate.v2\n" +
    "family=paged-attention-rotated-q-paged-kv-readonly-production\n" +
    "execution_policy=production-fast-path\nqualification_candidate_sha256=" +
    qualification.candidate_digest.as_hex() +
    "\nsource_sha256=" +
    source_digest.as_hex() +
    "\nfunction_symbol=" +
    symbol +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
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
    "\nfmad=" +
    readonly_bool(compiler.fused_multiply_add()) +
    "\nreassociate=" +
    readonly_bool(compiler.reassociate()) +
    "\nmax_registers=" +
    compiler.max_registers().to_string() +
    "\nmodel_operation_id=" +
    reference.operation_id().as_int().to_string() +
    "\nprofile_id=" +
    profile.id().as_int().to_string() +
    "\nprofile=" +
    profile.max_query_rows().to_string() +
    "," +
    profile.max_query_tokens().to_string() +
    "," +
    profile.max_page_table_entries().to_string() +
    "\ninput_row_width=" +
    qualification.input_row_width.to_string() +
    "\ninput_activation_ref=" +
    qualification.input_activation.as_int().to_string() +
    "\ntokens_per_page=" +
    reference.tokens_per_page().to_string() +
    "\npage_stride_bytes=" +
    reference.page_stride_bytes().to_string() +
    "\nphysical_page_capacity=" +
    reference.physical_page_capacity().to_string() +
    "\ngrid=" +
    launch.grid_x().to_string() +
    "," +
    launch.grid_y().to_string() +
    ",1\nblock=" +
    launch.block_x().to_string() +
    ",1,1\nshared_memory_bytes=0\n" +
    "attention_key_tile=16\nattention_key_warps=4\nattention_keys_per_warp=4\n" +
    "attention_block_threads=128\nattention_token_grid_cap=128\nraw_pointer_abi=" +
    RotatedQueryPagedKvReadOnlyProductionRawPointerV2.canonical_name() +
    "\ndiagnostic_canary=absent\nrotated_query_only=true\n" +
    "full_qkv_row_stride_bound=true\nkv_cache_access=read_only\n" +
    "kv_cache_mutation=none\naot_only=true\nruntime_jit=false\n" +
    "release_binding=external-approval-and-production-physical-gate-required\n" +
    "manifest_bindable=false\n",
  )
}

///|
pub fn lower_paged_attention_readonly_cuda_aot_candidate(
  reference : @luna_cuda_paged_attention_aot.PagedAttentionCudaLowering,
  entry_point_id : @catalog.AotKernelEntryPointId,
  limits : PagedAttentionReadOnlyAotLimits,
) -> PagedAttentionReadOnlyCudaCandidate raise PagedAttentionReadOnlyAotError {
  readonly_require_exact_reference(reference, entry_point_id)
  let abi = RotatedQueryPagedKvReadOnlyRawPointerV1
  let symbol = readonly_symbol(entry_point_id)
  let input_row_width = (reference.query_heads() + 2 * reference.kv_heads()) *
    reference.head_dimension()
  let input_activation = match reference.operands()[6].role() {
    ActivationInput(value) => value
    _ => raise PagedAttentionReadOnlyMismatch(KernelAbi)
  }
  let dispatch_canary_per_token = 65521 + entry_point_id.as_int()
  if dispatch_canary_per_token <= 0 ||
    dispatch_canary_per_token.to_int64() *
    reference.profile().max_query_tokens().to_int64() >
    4294967295L {
    raise PagedAttentionReadOnlyInvalid(KernelAbi)
  }
  let source = readonly_cuda_source(
    symbol,
    reference,
    input_row_width,
    Some(dispatch_canary_per_token),
  )
  if source.length() > limits.maximum_source_bytes {
    raise PagedAttentionReadOnlyLimitExceeded(Source)
  }
  let source_digest = readonly_digest(source)
  let launch = readonly_launch_dimensions(reference)
  let recipe = readonly_recipe(
    reference, entry_point_id, symbol, abi, source_digest, input_row_width, input_activation,
    dispatch_canary_per_token, launch,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise PagedAttentionReadOnlyLimitExceeded(Recipe)
  }
  let recipe_digest = readonly_digest(recipe)
  let canonical = @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-candidate.v1\n" +
    "operation_id=" +
    reference.operation_id().as_int().to_string() +
    "\nfallback_source_sha256=" +
    reference.source_digest().as_hex() +
    "\nfallback_recipe_sha256=" +
    reference.recipe_digest().as_hex() +
    "\nsource_sha256=" +
    source_digest.as_hex() +
    "\nrecipe_sha256=" +
    recipe_digest.as_hex() +
    "\ntarget=sm_" +
    reference.target().compute_major().to_string() +
    reference.target().compute_minor().to_string() +
    "\nprofile_id=" +
    reference.profile().id().as_int().to_string() +
    "\ninput_row_width=" +
    input_row_width.to_string() +
    "\ninput_activation_ref=" +
    input_activation.as_int().to_string() +
    "\ntokens_per_page=" +
    reference.tokens_per_page().to_string() +
    "\nkv_layout=layer_major_split_key_value_v1\nkv_layout_stable_version=1" +
    "\npage_stride_bytes=" +
    reference.page_stride_bytes().to_string() +
    "\nphysical_page_capacity=" +
    reference.physical_page_capacity().to_string() +
    "\nfunction_symbol=" +
    symbol +
    "\nraw_pointer_abi=" +
    abi.canonical_name() +
    "\ndispatch_canary_per_token=" +
    dispatch_canary_per_token.to_string() +
    "\ndispatch_canary_cell_count=" +
    reference.profile().max_query_tokens().to_string() +
    "\ndispatch_canary_publication=exactly-once-after-output\n" +
    "kv_cache_mutation=none\nmanifest_bindable=false\npromotion_authority=none\n",
  )
  if canonical.length() > limits.maximum_canonical_bytes {
    raise PagedAttentionReadOnlyLimitExceeded(Evidence)
  }
  {
    reference,
    entry_point_id,
    function_symbol: symbol,
    kernel_abi: abi,
    input_row_width,
    input_activation,
    dispatch_canary_per_token,
    launch,
    source,
    source_digest,
    recipe,
    recipe_digest,
    canonical,
    candidate_digest: readonly_digest(canonical),
  }
}

///|
/// Derives a canary-free readonly production candidate from the exact
/// diagnostic candidate. The result remains inert until separately approved.
pub fn lower_paged_attention_readonly_production_cuda_aot_candidate(
  qualification : PagedAttentionReadOnlyCudaCandidate,
  limits : PagedAttentionReadOnlyAotLimits,
) -> PagedAttentionReadOnlyProductionCudaCandidate raise PagedAttentionReadOnlyAotError {
  let symbol = "lunaflux_paged_attention_bf16_readonly_production_v2_ep_" +
    qualification.entry_point_id.as_int().to_string()
  let source = readonly_production_cuda_source(
    symbol,
    qualification.reference,
    qualification.input_row_width,
  )
  if source.length() > limits.maximum_source_bytes {
    raise PagedAttentionReadOnlyLimitExceeded(Source)
  }
  let source_digest = readonly_digest(source)
  let launch = qualification.launch
  let recipe = readonly_production_recipe(
    qualification, symbol, source_digest, launch,
  )
  if recipe.length() > limits.maximum_recipe_bytes {
    raise PagedAttentionReadOnlyLimitExceeded(Recipe)
  }
  let recipe_digest = readonly_digest(recipe)
  let canonical = @utf8.encode(
    "schema=lunaflux-paged-attention-readonly-production-candidate.v2\n" +
    "qualification_candidate_sha256=" +
    qualification.candidate_digest.as_hex() +
    "\nsource_sha256=" +
    source_digest.as_hex() +
    "\nrecipe_sha256=" +
    recipe_digest.as_hex() +
    "\nfunction_symbol=" +
    symbol +
    "\nraw_pointer_abi=" +
    RotatedQueryPagedKvReadOnlyProductionRawPointerV2.canonical_name() +
    "\ndiagnostic_canary=absent\nmanifest_bindable=false\n",
  )
  if canonical.length() > limits.maximum_canonical_bytes {
    raise PagedAttentionReadOnlyLimitExceeded(Evidence)
  }
  {
    qualification,
    function_symbol: symbol,
    kernel_abi: RotatedQueryPagedKvReadOnlyProductionRawPointerV2,
    launch,
    source,
    source_digest,
    recipe,
    recipe_digest,
    canonical,
    candidate_digest: readonly_digest(canonical),
  }
}
