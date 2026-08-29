///|
/// Tensor-core selection is intentionally evidence-aware. Observation may
/// emit a parallel SIMT candidate while retaining eligibility facts. Requiring
/// a tensor core never accepts caller assertion as qualification.
pub(all) enum LunaTileTensorCorePolicy {
  ObserveEligibilityUseParallelSimt
  RequireExternallyQualifiedTensorCore
} derive(Debug, Eq)

///|
pub(all) enum LunaTileTensorCoreEligibility {
  TensorCoreEligibleUnqualified
  TensorCoreIneligibleTarget
  TensorCoreIneligibleShape
  TensorCoreNoMma
} derive(Debug, Eq)

///|
pub(all) enum LunaTileParallelExecutionPath {
  ParallelSimtCandidate
} derive(Debug, Eq)

///|
pub struct LunaTileParallelPolicy {
  priv compute_major : Int
  priv compute_minor : Int
  priv block_threads : Int
  priv block_count : Int
  priv tile_m : Int
  priv tile_n : Int
  priv tile_k : Int
  priv pipeline_stages : Int
  priv vector_bytes : Int
  priv max_shared_memory_bytes : Int64
  priv tensor_core_policy : LunaTileTensorCorePolicy
} derive(Debug, Eq)

///|
pub fn LunaTileParallelPolicy::new(
  compute_major~ : Int,
  compute_minor~ : Int,
  block_threads~ : Int,
  block_count~ : Int,
  tile_m~ : Int,
  tile_n~ : Int,
  tile_k~ : Int,
  pipeline_stages~ : Int,
  vector_bytes~ : Int,
  max_shared_memory_bytes~ : Int64,
  tensor_core_policy~ : LunaTileTensorCorePolicy,
) -> LunaTileParallelPolicy raise LunaTileError {
  if compute_major < 7 ||
    compute_major > 99 ||
    compute_minor < 0 ||
    compute_minor > 9 ||
    block_threads < 32 ||
    block_threads > 256 ||
    block_threads % 32 != 0 ||
    block_count <= 0 ||
    block_count > 65535 ||
    tile_m <= 0 ||
    tile_n <= 0 ||
    tile_k <= 0 ||
    pipeline_stages < 2 ||
    pipeline_stages > 4 ||
    !(vector_bytes == 4 || vector_bytes == 8 || vector_bytes == 16) ||
    max_shared_memory_bytes <= 0L {
    raise InvalidProgram(ParallelPolicy)
  }
  {
    compute_major,
    compute_minor,
    block_threads,
    block_count,
    tile_m,
    tile_n,
    tile_k,
    pipeline_stages,
    vector_bytes,
    max_shared_memory_bytes,
    tensor_core_policy,
  }
}

///|
pub struct LunaTileParallelMapping {
  priv block_threads : Int
  priv block_count : Int
  priv warps_per_block : Int
  priv tile_m : Int
  priv tile_n : Int
  priv tile_k : Int
  priv tile_count : Int64
  priv rounds_per_worker : Int64
} derive(Debug, Eq)

///|
pub fn LunaTileParallelMapping::block_threads(
  self : LunaTileParallelMapping,
) -> Int {
  self.block_threads
}

///|
pub fn LunaTileParallelMapping::block_count(
  self : LunaTileParallelMapping,
) -> Int {
  self.block_count
}

///|
pub fn LunaTileParallelMapping::warps_per_block(
  self : LunaTileParallelMapping,
) -> Int {
  self.warps_per_block
}

///|
pub fn LunaTileParallelMapping::tile_shape(
  self : LunaTileParallelMapping,
) -> (Int, Int, Int) {
  (self.tile_m, self.tile_n, self.tile_k)
}

///|
pub fn LunaTileParallelMapping::tile_count(
  self : LunaTileParallelMapping,
) -> Int64 {
  self.tile_count
}

///|
pub fn LunaTileParallelMapping::rounds_per_worker(
  self : LunaTileParallelMapping,
) -> Int64 {
  self.rounds_per_worker
}

///|
pub struct LunaTileSharedPlacement {
  priv tensor : LunaTileTensorId
  priv offset_bytes : Int64
  priv size_bytes : Int64
  priv first_instruction : Int
  priv last_instruction : Int
  priv reused_slot : Bool
} derive(Debug, Eq)

///|
pub fn LunaTileSharedPlacement::tensor(
  self : LunaTileSharedPlacement,
) -> LunaTileTensorId {
  self.tensor
}

///|
pub fn LunaTileSharedPlacement::offset_bytes(
  self : LunaTileSharedPlacement,
) -> Int64 {
  self.offset_bytes
}

///|
pub fn LunaTileSharedPlacement::size_bytes(
  self : LunaTileSharedPlacement,
) -> Int64 {
  self.size_bytes
}

///|
pub fn LunaTileSharedPlacement::first_instruction(
  self : LunaTileSharedPlacement,
) -> Int {
  self.first_instruction
}

///|
pub fn LunaTileSharedPlacement::last_instruction(
  self : LunaTileSharedPlacement,
) -> Int {
  self.last_instruction
}

///|
pub fn LunaTileSharedPlacement::reused_slot(
  self : LunaTileSharedPlacement,
) -> Bool {
  self.reused_slot
}

///|
pub struct LunaTileParallelSharedPlan {
  priv placements : ReadOnlyArray[LunaTileSharedPlacement]
  priv arena_bytes : Int64
  priv total_shared_memory_bytes : Int64
  priv reuse_count : Int
} derive(Debug, Eq)

///|
pub fn LunaTileParallelSharedPlan::placement_count(
  self : LunaTileParallelSharedPlan,
) -> Int {
  self.placements.length()
}

///|
pub fn LunaTileParallelSharedPlan::placement_at(
  self : LunaTileParallelSharedPlan,
  index : Int,
) -> LunaTileSharedPlacement? {
  self.placements.get(index)
}

///|
pub fn LunaTileParallelSharedPlan::arena_bytes(
  self : LunaTileParallelSharedPlan,
) -> Int64 {
  self.arena_bytes
}

///|
pub fn LunaTileParallelSharedPlan::total_shared_memory_bytes(
  self : LunaTileParallelSharedPlan,
) -> Int64 {
  self.total_shared_memory_bytes
}

///|
pub fn LunaTileParallelSharedPlan::reuse_count(
  self : LunaTileParallelSharedPlan,
) -> Int {
  self.reuse_count
}

///|
pub struct LunaTileParallelDigest(String) derive(Debug, Eq, Hash)

///|
pub fn LunaTileParallelDigest::from_sha256(
  value : String,
) -> LunaTileParallelDigest raise LunaTileError {
  if !luna_tile_is_sha256(value) {
    raise InvalidProgram(CanonicalEncoding)
  }
  LunaTileParallelDigest(value)
}

///|
pub fn LunaTileParallelDigest::as_hex(self : LunaTileParallelDigest) -> String {
  self.0
}

///|
pub struct LunaTileParallelCandidate {
  priv serial_oracle_digest : LunaTileCudaTranslationDigest
  priv program_digest : LunaTileProgramDigest
  priv planning_digest : LunaTileCudaAotDigest
  priv compute_major : Int
  priv compute_minor : Int
  priv mapping : LunaTileParallelMapping
  priv shared : LunaTileParallelSharedPlan
  priv tensor_core : LunaTileTensorCoreEligibility
  priv path : LunaTileParallelExecutionPath
  priv entry_point : String
  priv source : Bytes
  priv source_digest : LunaTileCudaSourceDigest
  priv canonical : Bytes
  priv digest : LunaTileParallelDigest
} derive(Debug, Eq)

///|
pub fn LunaTileParallelCandidate::serial_oracle_digest(
  self : LunaTileParallelCandidate,
) -> LunaTileCudaTranslationDigest {
  self.serial_oracle_digest
}

///|
pub fn LunaTileParallelCandidate::program_digest(
  self : LunaTileParallelCandidate,
) -> LunaTileProgramDigest {
  self.program_digest
}

///|
pub fn LunaTileParallelCandidate::planning_digest(
  self : LunaTileParallelCandidate,
) -> LunaTileCudaAotDigest {
  self.planning_digest
}

///|
pub fn LunaTileParallelCandidate::compute_capability(
  self : LunaTileParallelCandidate,
) -> (Int, Int) {
  (self.compute_major, self.compute_minor)
}

///|
pub fn LunaTileParallelCandidate::mapping(
  self : LunaTileParallelCandidate,
) -> LunaTileParallelMapping {
  self.mapping
}

///|
pub fn LunaTileParallelCandidate::shared_plan(
  self : LunaTileParallelCandidate,
) -> LunaTileParallelSharedPlan {
  self.shared
}

///|
pub fn LunaTileParallelCandidate::tensor_core_eligibility(
  self : LunaTileParallelCandidate,
) -> LunaTileTensorCoreEligibility {
  self.tensor_core
}

///|
pub fn LunaTileParallelCandidate::execution_path(
  self : LunaTileParallelCandidate,
) -> LunaTileParallelExecutionPath {
  self.path
}

///|
pub fn LunaTileParallelCandidate::entry_point(
  self : LunaTileParallelCandidate,
) -> String {
  self.entry_point
}

///|
pub fn LunaTileParallelCandidate::source_bytes(
  self : LunaTileParallelCandidate,
) -> Bytes {
  self.source
}

///|
pub fn LunaTileParallelCandidate::source_digest(
  self : LunaTileParallelCandidate,
) -> LunaTileCudaSourceDigest {
  self.source_digest
}

///|
pub fn LunaTileParallelCandidate::canonical_bytes(
  self : LunaTileParallelCandidate,
) -> Bytes {
  self.canonical
}

///|
pub fn LunaTileParallelCandidate::digest(
  self : LunaTileParallelCandidate,
) -> LunaTileParallelDigest {
  self.digest
}
