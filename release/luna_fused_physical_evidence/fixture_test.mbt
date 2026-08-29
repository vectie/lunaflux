///|
priv struct FusedPhysicalBlackBoxFixture {
  qkv : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate
  qkv_compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding
  residual : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate
  residual_compiled : @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactBinding
}

///|
fn fused_physical_test_sha256(bytes : Bytes) -> String {
  @crypto.bytes_to_hex_string(@crypto.sha256(bytes))
}

///|
fn fused_physical_test_digest(
  bytes : Bytes,
) -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256(
    fused_physical_test_sha256(bytes),
  )
}

///|
fn fused_physical_test_policy_digest() -> @luna_cuda_fused_parallel_aot.FusedParallelDigest {
  @luna_cuda_fused_parallel_aot.FusedParallelDigest::from_sha256("9".repeat(64)) catch {
    _ => abort("fixed test policy digest is invalid")
  }
}

///|
fn fused_physical_test_receipt(
  source : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  recipe : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  compiler : @luna_specialization.LunaCompilerPolicy,
  target : @catalog.DeviceTarget,
  symbol : String,
  cubin : Bytes,
) -> Bytes {
  let module_digest = fused_physical_test_sha256(cubin)
  @utf8.encode(
    "schema=lunaflux-luna-cuda-aot-evidence-v1\n" +
    "source_sha256=" +
    source.as_hex() +
    "\nrecipe_sha256=" +
    recipe.as_hex() +
    "\ntoolchain_sha256=" +
    compiler.identity().toolchain_digest_sha256() +
    "\ndriver_identity_sha256=" +
    "d".repeat(64) +
    "\ntarget=sm_" +
    target.compute_major().to_string() +
    target.compute_minor().to_string() +
    "\nfunction_symbol=" +
    symbol +
    "\nartifact_sha256=" +
    module_digest +
    "\nfirst_build_sha256=" +
    module_digest +
    "\nsecond_build_sha256=" +
    module_digest +
    "\ndeterministic=1\n",
  )
}

///|
fn fused_physical_black_box_fixture() -> FusedPhysicalBlackBoxFixture raise {
  let model = @spec.LlamaModelSpec::new(
    vocabulary_size=256,
    hidden_size=128,
    intermediate_size=256,
    layer_count=1,
    attention_head_count=4,
    kv_head_count=2,
    context_length=256,
    rope_theta=10000.0,
    rms_norm_epsilon=0.00001,
    weight_dtype=BFloat16,
    kv_dtype=BFloat16,
  )
  let plan = @llama.build_paged(
    @spec.LlamaModelMetadata::from_verified_content(
      model,
      @spec.ContentDigest::from_sha256("a".repeat(64)),
    ),
    batch=@llama.LlamaPagedBatchEnvelope::single_row(),
  )
  let layout = @device_layout.plan(
    plan.identity(),
    plan.kv_cache_geometry(),
    @device_layout.DeviceKvLimits::new(
      tokens_per_page=16,
      total_page_count=4,
      max_arena_bytes=67108864L,
    ),
  )
  let profile = @launch_contract.PagedKernelProfile::new(
    id=@launch_contract.KernelProfileId::from_stable_int(7),
    max_query_rows=1,
    max_query_tokens=8,
    max_page_table_entries=4,
  )
  let target = @catalog.DeviceTarget::new(
    compute_major=12,
    compute_minor=0,
    supports_bf16=true,
    supports_cublas_lt=true,
  )
  let compiler = @luna_specialization.LunaCompilerPolicy::new(
    identity=@luna_specialization.LunaCompilerIdentity::new(
      toolchain_digest_sha256="b".repeat(64),
      major=13,
      minor=1,
      patch=115,
    ),
    optimization_level=3,
    fused_multiply_add=false,
    reassociate=false,
    max_registers=128,
  )
  let reference = @luna_bf16_kernel_producer.prepare_luna_bf16_candidate_set(
    plan,
    layout,
    profile,
    target,
    compiler,
    @luna_bf16_kernel_producer.LunaBf16CandidateSetLimits::new(
      max_operations=64,
      max_source_bytes=1048576,
      max_recipe_bytes=65536,
      max_total_bytes=33554432,
    ),
  )
  let set = @luna_fused_candidate_export.prepare_qualification_set(
    plan, layout, reference,
  )
  let qkv = set.qkv_candidate()
  let residual = set.residual_norm_candidate()
  let qkv_cubin = b"ELF-CUBIN-physical-qkv-v1"
  let residual_cubin = b"ELF-CUBIN-physical-residual-v1"
  let qkv_receipt = fused_physical_test_receipt(
    qkv.source_digest(),
    qkv.recipe_digest(),
    qkv.compiler_policy(),
    qkv.target(),
    qkv.function_symbol(),
    qkv_cubin,
  )
  let residual_receipt = fused_physical_test_receipt(
    residual.source_digest(),
    residual.recipe_digest(),
    residual.compiler_policy(),
    residual.target(),
    residual.function_symbol(),
    residual_cubin,
  )
  let limits = @luna_cuda_fused_parallel_aot.FusedParallelCompiledArtifactLimits::new(
    maximum_module_bytes=1048576L,
    maximum_receipt_bytes=65536,
    maximum_binding_bytes=65536,
  )
  let qkv_compiled = @luna_cuda_fused_parallel_aot.bind_fused_qkv_rope_kv_compiled_candidate(
    qkv,
    qkv_cubin,
    qkv_cubin,
    qkv_receipt,
    fused_physical_test_digest(qkv_receipt),
    limits,
  )
  let residual_compiled = @luna_cuda_fused_parallel_aot.bind_fused_residual_rmsnorm_compiled_candidate(
    residual,
    residual_cubin,
    residual_cubin,
    residual_receipt,
    fused_physical_test_digest(residual_receipt),
    limits,
  )
  { qkv, qkv_compiled, residual, residual_compiled }
}

///|
fn fused_physical_test_join_ints(values : ReadOnlyArray[Int]) -> String {
  [ for value in values => value.to_string() ].join("|")
}

///|
fn fused_physical_test_referee_digest() -> @luna_cuda_fused_parallel_aot.FusedParallelDigest raise {
  let output = StringBuilder()
  output.write_string("schema=lunaflux-fused-scalar-referee-shape-set.v1\n")
  let shapes = @fused_parallel_qualification.declared_qualification_shapes()
  output.write_string("shape_count=" + shapes.length().to_string() + "\n")
  for shape in shapes {
    output.write_string(
      "shape=" +
      shape.ordinal().to_string() +
      "," +
      shape.name() +
      "," +
      fused_physical_test_join_ints(shape.positions()) +
      "," +
      fused_physical_test_join_ints(shape.page_table()) +
      "\n",
    )
  }
  fused_physical_test_digest(@utf8.encode(output.to_string()))
}
