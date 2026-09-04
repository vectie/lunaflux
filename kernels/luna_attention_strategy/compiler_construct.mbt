///|
pub fn AttentionMatrixShape::new(
  rows~ : Int,
  columns~ : Int,
  reduction~ : Int,
) -> AttentionMatrixShape raise AttentionStrategyError {
  if rows <= 0 || columns <= 0 || reduction <= 0 {
    raise InvalidAttentionStrategy(Tile)
  }
  { rows, columns, reduction }
}

///|
pub fn AttentionCompilerCapabilities::new(
  subgroup_width~ : Int,
  max_threads_per_block~ : Int,
  max_shared_memory_bytes~ : Int64,
  max_pipeline_stages~ : Int,
  supports_async_copy~ : Bool,
  parallel_workgroup_target? : Int = 32,
  matrix_shape? : AttentionMatrixShape? = None,
) -> AttentionCompilerCapabilities raise AttentionStrategyError {
  if subgroup_width <= 0 ||
    subgroup_width > 128 ||
    (subgroup_width & (subgroup_width - 1)) != 0 ||
    max_threads_per_block < subgroup_width ||
    max_threads_per_block > 2048 ||
    max_threads_per_block % subgroup_width != 0 ||
    max_shared_memory_bytes <= 0L ||
    max_pipeline_stages <= 0 ||
    max_pipeline_stages > 8 ||
    parallel_workgroup_target <= 0 ||
    parallel_workgroup_target > 1048576 {
    raise InvalidAttentionStrategy(Capability)
  }
  {
    subgroup_width,
    max_threads_per_block,
    max_shared_memory_bytes,
    max_pipeline_stages,
    supports_async_copy,
    parallel_workgroup_target,
    matrix_shape,
  }
}

///|
/// Returns a new portable capability value with a different scheduling
/// parallelism objective. Offline frontier construction uses this instead of
/// mutating compiler state or introducing backend names into the optimizer.
pub fn AttentionCompilerCapabilities::with_parallel_workgroup_target(
  self : AttentionCompilerCapabilities,
  parallel_workgroup_target : Int,
) -> AttentionCompilerCapabilities raise AttentionStrategyError {
  AttentionCompilerCapabilities::new(
    subgroup_width=self.subgroup_width,
    max_threads_per_block=self.max_threads_per_block,
    max_shared_memory_bytes=self.max_shared_memory_bytes,
    max_pipeline_stages=self.max_pipeline_stages,
    supports_async_copy=self.supports_async_copy,
    parallel_workgroup_target~,
    matrix_shape=self.matrix_shape,
  )
}

///|
pub fn AttentionCompileProblem::new(
  shape~ : AttentionShape,
  phase~ : AttentionExecutionPhase,
  numeric_type~ : AttentionNumericType,
  batch_rows? : Int = 1,
  query_rows~ : Int,
  context_tokens~ : Int,
  causal~ : Bool,
) -> AttentionCompileProblem raise AttentionStrategyError {
  if phase == MixedCompatibility ||
    batch_rows <= 0 ||
    batch_rows > 1048576 ||
    query_rows <= 0 ||
    query_rows > 1048576 ||
    context_tokens <= 0 ||
    context_tokens > 1048576 ||
    (phase == Decode && query_rows != 1) {
    raise InvalidAttentionStrategy(Problem)
  }
  { shape, phase, numeric_type, batch_rows, query_rows, context_tokens, causal }
}

///|
fn valid_attention_compiler_identity(value : StringView) -> Bool {
  value.length() > 0 && value.length() <= 128
}

///|
pub fn AttentionBackendId::new(
  stable_name : String,
) -> AttentionBackendId raise AttentionStrategyError {
  if !valid_attention_compiler_identity(stable_name) {
    raise InvalidAttentionStrategy(Backend)
  }
  { stable_name, }
}

///|
pub fn AttentionTargetId::new(
  stable_name : String,
) -> AttentionTargetId raise AttentionStrategyError {
  if !valid_attention_compiler_identity(stable_name) {
    raise InvalidAttentionStrategy(Target)
  }
  { stable_name, }
}

///|
pub fn AttentionAutotuneKey::new(
  backend~ : AttentionBackendId,
  target~ : AttentionTargetId,
  problem~ : AttentionCompileProblem,
  candidate~ : AttentionTileCandidate,
) -> AttentionAutotuneKey {
  { backend, target, problem, candidate }
}

///|
pub fn AttentionAutotuneRecord::new(
  key~ : AttentionAutotuneKey,
  median_latency_ns~ : Int64,
  sample_count~ : Int,
) -> AttentionAutotuneRecord raise AttentionStrategyError {
  if median_latency_ns <= 0L || sample_count < 3 {
    raise InvalidAttentionStrategy(Measurement)
  }
  { key, median_latency_ns, sample_count }
}
