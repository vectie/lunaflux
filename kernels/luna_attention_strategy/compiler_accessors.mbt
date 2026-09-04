///|
pub fn AttentionMatrixShape::dimensions(
  self : AttentionMatrixShape,
) -> (Int, Int, Int) {
  (self.rows, self.columns, self.reduction)
}

///|
pub fn AttentionCompilerCapabilities::subgroup_width(
  self : AttentionCompilerCapabilities,
) -> Int {
  self.subgroup_width
}

///|
pub fn AttentionCompilerCapabilities::limits(
  self : AttentionCompilerCapabilities,
) -> (Int, Int64, Int) {
  (
    self.max_threads_per_block,
    self.max_shared_memory_bytes,
    self.max_pipeline_stages,
  )
}

///|
pub fn AttentionCompilerCapabilities::supports_async_copy(
  self : AttentionCompilerCapabilities,
) -> Bool {
  self.supports_async_copy
}

///|
pub fn AttentionCompilerCapabilities::parallel_workgroup_target(
  self : AttentionCompilerCapabilities,
) -> Int {
  self.parallel_workgroup_target
}

///|
pub fn AttentionCompilerCapabilities::matrix_shape(
  self : AttentionCompilerCapabilities,
) -> AttentionMatrixShape? {
  self.matrix_shape
}

///|
pub fn AttentionCompileProblem::shape(
  self : AttentionCompileProblem,
) -> AttentionShape {
  self.shape
}

///|
pub fn AttentionCompileProblem::phase(
  self : AttentionCompileProblem,
) -> AttentionExecutionPhase {
  self.phase
}

///|
pub fn AttentionCompileProblem::numeric_type(
  self : AttentionCompileProblem,
) -> AttentionNumericType {
  self.numeric_type
}

///|
pub fn AttentionCompileProblem::batch_rows(
  self : AttentionCompileProblem,
) -> Int {
  self.batch_rows
}

///|
pub fn AttentionCompileProblem::dimensions(
  self : AttentionCompileProblem,
) -> (Int, Int) {
  (self.query_rows, self.context_tokens)
}

///|
pub fn AttentionCompileProblem::is_causal(
  self : AttentionCompileProblem,
) -> Bool {
  self.causal
}

///|
pub fn AttentionTileShape::dimensions(
  self : AttentionTileShape,
) -> (Int, Int, Int) {
  (self.query_rows, self.key_value_rows, self.head_columns)
}

///|
pub fn AttentionTileCandidate::stable_id(self : AttentionTileCandidate) -> Int {
  self.stable_id
}

///|
pub fn AttentionTileCandidate::compute(
  self : AttentionTileCandidate,
) -> AttentionTileCompute {
  self.compute
}

///|
pub fn AttentionTileCandidate::memory(
  self : AttentionTileCandidate,
) -> AttentionTileMemory {
  self.memory
}

///|
pub fn AttentionTileCandidate::tile(
  self : AttentionTileCandidate,
) -> AttentionTileShape {
  self.tile
}

///|
pub fn AttentionTileCandidate::execution_geometry(
  self : AttentionTileCandidate,
) -> (Int, Int, Int) {
  (self.subgroups_per_block, self.pipeline_stages, self.vector_bytes)
}

///|
pub fn AttentionTileCandidate::shared_memory_bytes(
  self : AttentionTileCandidate,
) -> Int64 {
  self.shared_memory_bytes
}

///|
pub fn AttentionBackendId::stable_name(self : AttentionBackendId) -> StringView {
  self.stable_name
}

///|
pub fn AttentionTargetId::stable_name(self : AttentionTargetId) -> StringView {
  self.stable_name
}

///|
pub fn AttentionAutotuneKey::backend(
  self : AttentionAutotuneKey,
) -> AttentionBackendId {
  self.backend
}

///|
pub fn AttentionAutotuneKey::target(
  self : AttentionAutotuneKey,
) -> AttentionTargetId {
  self.target
}

///|
pub fn AttentionAutotuneKey::problem(
  self : AttentionAutotuneKey,
) -> AttentionCompileProblem {
  self.problem
}

///|
pub fn AttentionAutotuneKey::candidate(
  self : AttentionAutotuneKey,
) -> AttentionTileCandidate {
  self.candidate
}

///|
pub fn AttentionAutotuneRecord::key(
  self : AttentionAutotuneRecord,
) -> AttentionAutotuneKey {
  self.key
}

///|
pub fn AttentionAutotuneRecord::median_latency_ns(
  self : AttentionAutotuneRecord,
) -> Int64 {
  self.median_latency_ns
}

///|
pub fn AttentionAutotuneRecord::sample_count(
  self : AttentionAutotuneRecord,
) -> Int {
  self.sample_count
}

///|
pub fn CompiledAttentionTilePlan::problem(
  self : CompiledAttentionTilePlan,
) -> AttentionCompileProblem {
  self.problem
}

///|
pub fn CompiledAttentionTilePlan::candidate(
  self : CompiledAttentionTilePlan,
) -> AttentionTileCandidate {
  self.candidate
}

///|
pub fn CompiledAttentionTilePlan::source(
  self : CompiledAttentionTilePlan,
) -> AttentionCandidateSource {
  self.source
}

///|
pub fn CompiledAttentionTilePlan::static_cost(
  self : CompiledAttentionTilePlan,
) -> Int64 {
  self.static_cost
}
