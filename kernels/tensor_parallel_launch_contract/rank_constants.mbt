///|
const TP_RANK_CONSTANTS_VERSION : Int = 2

///|
const TP_RANK_CONSTANTS_BYTES : Int64 = 32L

///|
const TP_RANK_CONSTANTS_ALIGNMENT : Int64 = 16L

///|
fn build_rank_constants(
  rank : Int,
  world_size : Int,
  target : @catalog.DeviceTarget,
  layout : @device_layout.DeviceKvLayout,
) -> TensorParallelRankConstants {
  {
    version: TP_RANK_CONSTANTS_VERSION,
    byte_count: TP_RANK_CONSTANTS_BYTES,
    alignment: TP_RANK_CONSTANTS_ALIGNMENT,
    rank,
    world_size,
    compute_major: target.compute_major(),
    compute_minor: target.compute_minor(),
    kv_layout_version: layout.stable_layout_version(),
    tokens_per_page: layout.tokens_per_page(),
  }
}

///|
/// Encodes the canonical v2 rank/runtime-wide record into caller-owned
/// storage without allocating. Operation dimensions live only on their exact
/// operation contracts.
pub fn TensorParallelRankConstants::encode_into(
  self : TensorParallelRankConstants,
  destination : FixedArray[Byte],
  destination_offset~ : Int,
) -> Unit raise TensorParallelLaunchContractError {
  if destination_offset < 0 ||
    destination_offset > destination.length() ||
    destination.length() - destination_offset < 32 {
    raise invalid(RankConstantsDestination)
  }
  rank_constants_write_i32(destination, destination_offset, 0, self.version)
  rank_constants_write_i32(destination, destination_offset, 1, self.rank)
  rank_constants_write_i32(destination, destination_offset, 2, self.world_size)
  rank_constants_write_i32(
    destination,
    destination_offset,
    3,
    self.compute_major,
  )
  rank_constants_write_i32(
    destination,
    destination_offset,
    4,
    self.compute_minor,
  )
  rank_constants_write_i32(
    destination,
    destination_offset,
    5,
    self.kv_layout_version,
  )
  rank_constants_write_i32(
    destination,
    destination_offset,
    6,
    self.tokens_per_page,
  )
  rank_constants_write_i32(destination, destination_offset, 7, 0)
}

///|
fn rank_constants_write_i32(
  destination : FixedArray[Byte],
  destination_offset : Int,
  field : Int,
  value : Int,
) -> Unit {
  destination.unsafe_write_uint32_le(
    destination_offset + field * 4,
    value.reinterpret_as_uint(),
  )
}

///|
pub fn TensorParallelRankConstants::version(
  self : TensorParallelRankConstants,
) -> Int {
  self.version
}

///|
pub fn TensorParallelRankConstants::byte_count(
  self : TensorParallelRankConstants,
) -> Int64 {
  self.byte_count
}

///|
pub fn TensorParallelRankConstants::alignment(
  self : TensorParallelRankConstants,
) -> Int64 {
  self.alignment
}

///|
pub fn TensorParallelRankConstants::rank(
  self : TensorParallelRankConstants,
) -> Int {
  self.rank
}

///|
pub fn TensorParallelRankConstants::world_size(
  self : TensorParallelRankConstants,
) -> Int {
  self.world_size
}

///|
pub fn TensorParallelRankConstants::compute_major(
  self : TensorParallelRankConstants,
) -> Int {
  self.compute_major
}

///|
pub fn TensorParallelRankConstants::compute_minor(
  self : TensorParallelRankConstants,
) -> Int {
  self.compute_minor
}

///|
pub fn TensorParallelRankConstants::kv_layout_version(
  self : TensorParallelRankConstants,
) -> Int {
  self.kv_layout_version
}

///|
pub fn TensorParallelRankConstants::tokens_per_page(
  self : TensorParallelRankConstants,
) -> Int {
  self.tokens_per_page
}
