///|
/// Scalarize a strided key map into the finite score vector owned by one
/// partition of a decode tile. Partition p visits p + offset, guarded by the
/// active-key bound; empty partitions produce the softmax merge identity.
///
/// Each vector is summarized with a tile maximum and weighted sum, then merged
/// into the running online-softmax state once. This is an associative fold
/// reassociation, not bit-exact preservation of a per-key floating-point fold.
/// The backend must use max-delta exponentials and retain the active-key guard.
/// No subgroup width, address space, model family, or device name is encoded.
pub fn scalarize_attention_decode_softmax_tile(
  key_value_tile : Int,
  key_partitions : Int,
) -> ReadOnlyArray[Int] raise AttentionTileScheduleError {
  if key_value_tile <= 0 ||
    key_value_tile > 128 ||
    key_partitions <= 0 ||
    key_partitions > 32 {
    raise InvalidAttentionTileSchedule(InvalidTile)
  }
  let slots = 1 + (key_value_tile - 1) / key_partitions
  [
    for slot in 0..<slots => slot * key_partitions
  ]
}
