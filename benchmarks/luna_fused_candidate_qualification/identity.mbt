///|
priv struct FusedBenchmarkCandidateIdentity {
  family : @luna_cuda_fused_parallel_aot.FusedParallelFamily
  family_name : String
  candidate : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  source : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  recipe : @luna_cuda_fused_parallel_aot.FusedParallelDigest
  target_compute_major : Int
  target_compute_minor : Int
  target_bf16 : Bool
  target_cublas_lt : Bool
  profile_id : Int
  max_query_rows : Int
  max_query_tokens : Int
  max_page_table_entries : Int
  grid_x : Int
  grid_y : Int
  grid_z : Int
  block_x : Int
  block_y : Int
  block_z : Int
  shared_memory_bytes : Int
  toolchain : String
  compiler_major : Int
  compiler_minor : Int
  compiler_patch : Int
  optimization_level : Int
  fused_multiply_add : Bool
  reassociate : Bool
  max_registers : Int
  numerical_policy : String
  function_symbol : String
  dispatch_canary : UInt
  fallback_lines : String
}

///|
fn fused_benchmark_common_identity(
  family : @luna_cuda_fused_parallel_aot.FusedParallelFamily,
  family_name : String,
  candidate : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  source : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  recipe : @luna_cuda_fused_parallel_aot.FusedParallelDigest,
  target : @catalog.DeviceTarget,
  profile : @launch_contract.PagedKernelProfile,
  dimensions : @launch_contract.AotLaunchDimensions,
  compiler : @luna_specialization.LunaCompilerPolicy,
  numerical_policy : String,
  function_symbol : String,
  dispatch_canary : UInt,
  fallback_lines : String,
) -> FusedBenchmarkCandidateIdentity {
  let compiler_identity = compiler.identity()
  {
    family,
    family_name,
    candidate,
    source,
    recipe,
    target_compute_major: target.compute_major(),
    target_compute_minor: target.compute_minor(),
    target_bf16: target.supports_bf16(),
    target_cublas_lt: target.supports_cublas_lt(),
    profile_id: profile.id().as_int(),
    max_query_rows: profile.max_query_rows(),
    max_query_tokens: profile.max_query_tokens(),
    max_page_table_entries: profile.max_page_table_entries(),
    grid_x: dimensions.grid_x(),
    grid_y: dimensions.grid_y(),
    grid_z: dimensions.grid_z(),
    block_x: dimensions.block_x(),
    block_y: dimensions.block_y(),
    block_z: dimensions.block_z(),
    shared_memory_bytes: dimensions.shared_memory_bytes(),
    toolchain: compiler_identity.toolchain_digest_sha256(),
    compiler_major: compiler_identity.major(),
    compiler_minor: compiler_identity.minor(),
    compiler_patch: compiler_identity.patch(),
    optimization_level: compiler.optimization_level(),
    fused_multiply_add: compiler.fused_multiply_add(),
    reassociate: compiler.reassociate(),
    max_registers: compiler.max_registers(),
    numerical_policy,
    function_symbol,
    dispatch_canary,
    fallback_lines,
  }
}

///|
fn fused_benchmark_qkv_identity(
  candidate : @luna_cuda_fused_parallel_aot.FusedQkvRopeKvWriteCudaCandidate,
) -> FusedBenchmarkCandidateIdentity {
  let qkv = candidate.qkv_fallback()
  let rope = candidate.rope_fallback()
  let kv_write = candidate.kv_write_fallback()
  let fallback_lines = "fallback=qkv,operation=" +
    qkv.operation_id().as_int().to_string() +
    ",source_sha256=" +
    qkv.source_digest().as_hex() +
    ",recipe_sha256=" +
    qkv.recipe_digest().as_hex() +
    "\n" +
    "fallback=positioned-rope,operation=" +
    rope.operation_id().as_int().to_string() +
    ",source_sha256=" +
    rope.source_digest().as_hex() +
    ",recipe_sha256=" +
    rope.recipe_digest().as_hex() +
    "\n" +
    "fallback=paged-kv-write,operation=" +
    kv_write.operation_id().as_int().to_string() +
    ",source_sha256=" +
    kv_write.source_digest().as_hex() +
    ",recipe_sha256=" +
    kv_write.recipe_digest().as_hex() +
    "\n"
  fused_benchmark_common_identity(
    QkvPositionedRopePagedKvWrite,
    "qkv-positioned-rope-paged-kvwrite",
    candidate.candidate_digest(),
    candidate.source_digest(),
    candidate.recipe_digest(),
    candidate.target(),
    candidate.profile(),
    candidate.launch_dimensions(),
    candidate.compiler_policy(),
    candidate.numerical_policy().name(),
    candidate.function_symbol(),
    candidate.dispatch_canary(),
    fallback_lines,
  )
}

///|
fn fused_benchmark_residual_identity(
  candidate : @luna_cuda_fused_parallel_aot.FusedResidualRmsNormCudaCandidate,
) -> FusedBenchmarkCandidateIdentity {
  let residual = candidate.residual_fallback()
  let norm = candidate.norm_fallback()
  let fallback_lines = "fallback=residual,operation=" +
    residual.operation_id().as_int().to_string() +
    ",source_sha256=" +
    residual.source_digest().as_hex() +
    ",recipe_sha256=" +
    residual.recipe_digest().as_hex() +
    "\n" +
    "fallback=rmsnorm,operation=" +
    norm.operation_id().as_int().to_string() +
    ",source_sha256=" +
    norm.source_digest().as_hex() +
    ",recipe_sha256=" +
    norm.recipe_digest().as_hex() +
    "\n"
  fused_benchmark_common_identity(
    ResidualRmsNorm,
    "residual-rmsnorm",
    candidate.candidate_digest(),
    candidate.source_digest(),
    candidate.recipe_digest(),
    candidate.target(),
    candidate.profile(),
    candidate.launch_dimensions(),
    candidate.compiler_policy(),
    candidate.numerical_policy().name(),
    candidate.function_symbol(),
    candidate.dispatch_canary(),
    fallback_lines,
  )
}
