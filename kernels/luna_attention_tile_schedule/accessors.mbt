///|
pub fn AttentionLoopCombinator::kind(
  self : AttentionLoopCombinator,
) -> AttentionIterationKind {
  self.kind
}

///|
pub fn AttentionLoopCombinator::axis(
  self : AttentionLoopCombinator,
) -> AttentionIterationAxis {
  self.axis
}

///|
pub fn AttentionLoopCombinator::geometry(
  self : AttentionLoopCombinator,
) -> (Int, Int) {
  (self.extent, self.tile)
}

///|
pub fn AttentionTileSchedule::program_digest(
  self : AttentionTileSchedule,
) -> @luna_attention_tile_ir.AttentionTileProgramDigest {
  self.program_digest
}

///|
pub fn AttentionTileSchedule::loops(
  self : AttentionTileSchedule,
) -> ReadOnlyArray[AttentionLoopCombinator] {
  self.loops
}

///|
pub fn AttentionTileSchedule::compute(
  self : AttentionTileSchedule,
) -> @luna_attention_strategy.AttentionTileCompute {
  self.compute
}

///|
pub fn AttentionTileSchedule::memory(
  self : AttentionTileSchedule,
) -> @luna_attention_strategy.AttentionTileMemory {
  self.memory
}

///|
pub fn AttentionTileSchedule::execution_geometry(
  self : AttentionTileSchedule,
) -> (Int, Int, Int, Int64) {
  (
    self.subgroups_per_block,
    self.pipeline_stages,
    self.vector_bytes,
    self.shared_memory_bytes,
  )
}

///|
pub fn AttentionTileSchedule::key_value_partitions(
  self : AttentionTileSchedule,
) -> Int {
  self.key_value_partitions
}

///|
pub fn AttentionTileSchedule::hoists_paged_key_value_page_lookup(
  self : AttentionTileSchedule,
) -> Bool {
  self.hoist_paged_key_value_page_lookup
}

///|
pub fn AttentionTileSchedule::reuses_online_softmax_tile_storage(
  self : AttentionTileSchedule,
) -> Bool {
  self.reuse_online_softmax_tile_storage
}

///|
pub fn AttentionTileSchedule::canonical_bytes(
  self : AttentionTileSchedule,
) -> Bytes {
  self.canonical
}

///|
pub fn AttentionTileSchedule::digest(
  self : AttentionTileSchedule,
) -> AttentionTileScheduleDigest {
  self.digest
}

///|
pub fn AttentionTileScheduleDigest::as_hex(
  self : AttentionTileScheduleDigest,
) -> String {
  self.0
}
