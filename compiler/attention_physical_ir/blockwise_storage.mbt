///|
/// BF16 operand rings and F32 score exchange for a blockwise fold. Validity
/// cells are a distinct lifetime-owned region, not unconditional padding.
/// Device launch reservation may exceed this live extent for a measured
/// resource policy; the storage plan does not select residency or a device.
pub struct BlockwiseTileStorage {
  priv operand_bytes : Int64
  priv score_bytes : Int64
  priv validity_bytes : Int64
} derive(Debug, Eq)

///|
pub extend BlockwiseTileStorage with Eq::{equal, not_equal}

///|
pub extend BlockwiseTileStorage with @debug.Debug::{to_repr}

///|
pub fn BlockwiseTileStorage::new(
  dimensions~ : Int,
  keys~ : Int,
  query_consumers~ : Int,
  stages~ : Int,
  validity_slots~ : Int,
) -> BlockwiseTileStorage? {
  // Keep every region in the signed launch-byte domain. Check divisions
  // before products, including hostile Int::max_value geometry.
  guard dimensions > 0 &&
    dimensions % 2 == 0 &&
    keys > 0 &&
    query_consumers > 0 &&
    (stages == 1 || stages == 2) &&
    validity_slots >= 0 &&
    validity_slots <= stages else {
    return None
  }
  let ceiling = 2147483647L
  let ring_bytes = stages.to_int64() * 4L
  guard dimensions.to_int64() <= ceiling / ring_bytes &&
    keys.to_int64() <= ceiling / (dimensions.to_int64() * ring_bytes) &&
    query_consumers.to_int64() <= ceiling / 4L &&
    keys.to_int64() <= ceiling / (query_consumers.to_int64() * 4L) else {
    return None
  }
  let operand_bytes = dimensions.to_int64() * keys.to_int64() * ring_bytes
  let score_bytes = query_consumers.to_int64() * keys.to_int64() * 4L
  let validity_bytes = validity_slots.to_int64() * 4L
  guard operand_bytes <= ceiling - score_bytes - validity_bytes else {
    return None
  }
  Some({ operand_bytes, score_bytes, validity_bytes, })
}

///|
/// Half-open byte offsets for operands, scores and optional validity.
pub fn BlockwiseTileStorage::region_offsets_bytes(
  self : BlockwiseTileStorage,
) -> (Int64, Int64, Int64) {
  (0L, self.operand_bytes, self.operand_bytes + self.score_bytes)
}

///|
pub fn BlockwiseTileStorage::region_sizes_bytes(
  self : BlockwiseTileStorage,
) -> (Int64, Int64, Int64) {
  (self.operand_bytes, self.score_bytes, self.validity_bytes)
}

///|
pub fn BlockwiseTileStorage::minimum_bytes(
  self : BlockwiseTileStorage,
) -> Int64 {
  self.operand_bytes + self.score_bytes + self.validity_bytes
}

///|
/// Live-region containment only. A larger reservation can deliberately alter
/// residency; legality does not make it a measured performance choice.
pub fn BlockwiseTileStorage::accepts_reservation(
  self : BlockwiseTileStorage,
  bytes : Int64,
) -> Bool {
  bytes >= self.minimum_bytes() && bytes <= 2147483647L
}
