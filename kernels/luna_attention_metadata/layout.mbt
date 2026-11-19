///|
/// Device-independent, bounded CSR tile metadata layout. Four query granularities
/// share one step lifetime. Records are [row, token_begin, row_begin, row_end,
/// sequence_length, page_begin, page_end, exclusive_key_end].
pub struct AttentionMetadataLayout {
  priv maximum_rows : Int32
  priv maximum_tokens : Int32
} derive(Eq, Debug)

///|
pub extend AttentionMetadataLayout with @moonbitlang/core/debug.Debug::{to_repr}

///|
pub extend AttentionMetadataLayout with Eq::{not_equal, equal}

///|
pub fn AttentionMetadataLayout::new(
  maximum_rows~ : Int32,
  maximum_tokens~ : Int32,
) -> AttentionMetadataLayout? {
  if maximum_rows <= 0 ||
    maximum_tokens < maximum_rows ||
    maximum_tokens > 1048576 {
    None
  } else {
    Some({ maximum_rows, maximum_tokens, })
  }
}

///|
pub fn AttentionMetadataLayout::bucket(
  self : AttentionMetadataLayout,
  query_tile : Int32,
) -> (Int32, Int32, Int32)? {
  let mut offset = 4
  for bucket in 0..<4 {
    let width = 16 << bucket
    let capacity = self.maximum_rows +
      (self.maximum_tokens - self.maximum_rows) / width
    if width == query_tile {
      return Some((bucket, offset, capacity))
    }
    offset += capacity * 8
  }
  None
}

///|
pub fn AttentionMetadataLayout::cell_count(
  self : AttentionMetadataLayout,
) -> Int32 {
  let mut cells = 4
  for bucket in 0..<4 {
    cells += (
        self.maximum_rows +
        (self.maximum_tokens - self.maximum_rows) / (16 << bucket)
      ) *
      8
  }
  cells
}

///|
pub fn AttentionMetadataLayout::bucket_offset(
  self : AttentionMetadataLayout,
  ordinal : Int32,
) -> Int32 {
  let mut offset = 4
  for bucket in 0..<ordinal {
    offset += (
        self.maximum_rows +
        (self.maximum_tokens - self.maximum_rows) / (16 << bucket)
      ) *
      8
  }
  offset
}
