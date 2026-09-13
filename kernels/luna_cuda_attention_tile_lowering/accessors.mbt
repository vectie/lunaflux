///|
pub fn CudaAttentionTileLowering::program_digest(
  self : CudaAttentionTileLowering,
) -> @luna_attention_tile_ir.AttentionTileProgramDigest {
  self.program_digest
}

///|
pub fn CudaAttentionTileLowering::optimization_digest(
  self : CudaAttentionTileLowering,
) -> @luna_attention_tile_optimizer.AttentionTileOptimizationDigest {
  self.optimization_digest
}

///|
pub fn CudaAttentionTileLowering::schedule_digest(
  self : CudaAttentionTileLowering,
) -> @luna_attention_tile_schedule.AttentionTileScheduleDigest {
  self.schedule_digest
}

///|
pub fn CudaAttentionTileLowering::target(
  self : CudaAttentionTileLowering,
) -> @catalog.DeviceTarget {
  self.target
}

///|
pub fn CudaAttentionTileLowering::phase(
  self : CudaAttentionTileLowering,
) -> @luna_attention_strategy.AttentionExecutionPhase {
  self.phase
}

///|
pub fn CudaAttentionTileLowering::compute(
  self : CudaAttentionTileLowering,
) -> CudaAttentionComputePrimitive {
  self.compute
}

///|
pub fn CudaAttentionTileLowering::copy(
  self : CudaAttentionTileLowering,
) -> CudaAttentionCopyPrimitive {
  self.copy
}

///|
pub fn CudaAttentionTileLowering::matrix_storage_layout(
  self : CudaAttentionTileLowering,
) -> @luna_attention_tile_schedule.AttentionTileStorageLayout? {
  self.matrix_storage_layout
}

///|
pub fn CudaAttentionTileLowering::launch(
  self : CudaAttentionTileLowering,
) -> (Int, Int, Int, Int64) {
  (self.grid_x, self.grid_y, self.block_threads, self.shared_memory_bytes)
}

///|
/// Number of independent KV partitions in the generic reduction schedule.
/// CUDA maps this value to grid Z only in the partitioned pipeline emitter.
pub fn CudaAttentionTileLowering::key_value_partition_grain(
  self : CudaAttentionTileLowering,
) -> Int {
  self.key_value_partition_grain
}

///|
pub fn CudaAttentionTileLowering::key_value_partitions(
  self : CudaAttentionTileLowering,
) -> Int {
  self.key_value_partitions
}

///|
pub fn CudaAttentionTileLowering::tile(
  self : CudaAttentionTileLowering,
) -> (Int, Int, Int) {
  (self.query_tile, self.key_value_tile, self.head_dimension)
}

///|
pub fn CudaAttentionTileLowering::pipeline(
  self : CudaAttentionTileLowering,
) -> (Int, Int) {
  (self.pipeline_stages, self.vector_bytes)
}

///|
pub fn CudaAttentionTileLowering::hoists_paged_key_value_page_lookup(
  self : CudaAttentionTileLowering,
) -> Bool {
  self.hoist_paged_key_value_page_lookup
}

///|
pub fn CudaAttentionTileLowering::reuses_online_softmax_tile_storage(
  self : CudaAttentionTileLowering,
) -> Bool {
  self.reuse_online_softmax_tile_storage
}

///|
pub fn CudaAttentionTileLowering::factors_shared_key_query_map(
  self : CudaAttentionTileLowering,
) -> Bool {
  self.factor_shared_key_query_map
}

///|
pub fn CudaAttentionTileLowering::canonical_bytes(
  self : CudaAttentionTileLowering,
) -> Bytes {
  self.canonical
}

///|
pub fn CudaAttentionTileLowering::digest(
  self : CudaAttentionTileLowering,
) -> CudaAttentionTileLoweringDigest {
  self.digest
}

///|
pub fn CudaAttentionTileLoweringDigest::as_hex(
  self : CudaAttentionTileLoweringDigest,
) -> String {
  self.0
}

///|
pub fn CudaAttentionTileLowering::transfer_readiness(
  self : CudaAttentionTileLowering,
) -> @luna_attention_tile_schedule.AttentionTransferReadiness {
  self.transfer_readiness
}

///|
pub fn CudaAttentionTileLowering::retains_output_fold(
  self : CudaAttentionTileLowering,
) -> Bool {
  self.resident_output_fold
}

///|
/// A synchronous tile may use direct global-to-shared transport without a
/// lookahead buffer. Completion is still required before its first consumer;
/// the generic staged-memory schedule and its shorter lifetimes are unchanged.
pub fn CudaAttentionTileLowering::direct_prefill_staging(
  self : CudaAttentionTileLowering,
) -> Bool {
  self.phase == Prefill &&
  self.compute == CudaMatrixQkPvMultiplyAccumulate &&
  self.copy == CudaCooperativeSharedLoad &&
  self.pipeline_stages == 1 &&
  self.vector_bytes == 16 &&
  self.target.compute_major() >= 8
}

///|
pub fn CudaAttentionTileLowering::query_owned_fragments(
  self : CudaAttentionTileLowering,
) -> Bool {
  self.query_owned_fragments
}

///|
pub fn CudaAttentionTileLowering::reads_dense_current_key_value(
  self : CudaAttentionTileLowering,
) -> Bool {
  self.reads_dense_current_key_value
}
