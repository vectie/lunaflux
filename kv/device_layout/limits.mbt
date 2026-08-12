///|
const CANONICAL_SEGMENT_ALIGNMENT_BYTES : Int64 = 256L

///|
/// Constructs the physical-page and byte envelope for layout V1.
pub fn DeviceKvLimits::new(
  tokens_per_page~ : Int,
  total_page_count~ : Int,
  max_arena_bytes~ : Int64,
) -> DeviceKvLimits raise DeviceKvLayoutError {
  if tokens_per_page <= 0 {
    raise InvalidLimit(TokensPerPage)
  }
  if total_page_count <= 0 {
    raise InvalidLimit(TotalPageCount)
  }
  if max_arena_bytes <= 0L {
    raise InvalidLimit(ArenaBytes)
  }
  { tokens_per_page, total_page_count, max_arena_bytes }
}
