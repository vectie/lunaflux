///|
/// Select compressed-cache row IDs, not expanded raw token IDs. Index dot
/// products and weighted head products publish BF16 before the final BF16
/// head reduction. Causality admits floor((position+1)/ratio) cache rows.
pub struct CompressedIndexSelectionPrecision {
  priv heads : Int32
  priv width : Int32
  priv ratio : Int32
  priv rows : Int32
  priv positions : Int32
  priv top_k : Int32
  priv coordinates : CompressedRowCoordinates
  priv frames : ReadOnlyArray[Int64]
  priv workspace : Int64
}

///|
pub fn CompressedIndexSelectionPrecision::new(
  head_count~ : Int32,
  head_width~ : Int32,
  compression_ratio~ : Int32,
  maximum_rows~ : Int32,
  maximum_positions~ : Int32,
  top_k~ : Int32,
  coordinates~ : CompressedRowCoordinates = OffsetRelative,
  maximum_workspace_bytes~ : Int64,
) -> Self raise PrecisionPlanError {
  guard head_count > 0 &&
    head_count <= 512 &&
    head_width > 0 &&
    head_width <= 65536 &&
    compression_ratio > 1 &&
    compression_ratio <= 1048576 &&
    maximum_rows > 0 &&
    maximum_rows <= 65536 &&
    maximum_positions >= maximum_rows &&
    maximum_positions <= 1048576 &&
    top_k > 0 &&
    top_k <= 65536 else {
    raise IllegalPlacement
  }
  let capacity = maximum_positions / compression_ratio
  let selected = Int32::min(top_k, capacity)
  let sizes = [
    maximum_rows.to_int64() * Int32::max(capacity, 1).to_int64() * 2L,
    maximum_rows.to_int64() * Int32::max(selected, 1).to_int64() * 4L,
    maximum_rows.to_int64() * 4L,
  ]
  guard sizes.all(bytes => bytes <= 4294967294L) else {
    raise ArithmeticOverflow
  }
  let workspace = sizes.fold(init=0L, (sum, bytes) => sum + bytes)
  guard workspace <= maximum_workspace_bytes else { raise MemoryBudgetExceeded }
  {
    heads: head_count,
    width: head_width,
    ratio: compression_ratio,
    rows: maximum_rows,
    positions: maximum_positions,
    top_k,
    coordinates,
    frames: ReadOnlyArray::from_array(sizes),
    workspace,
  }
}

///|
pub fn CompressedIndexSelectionPrecision::coordinates(
  self : Self,
) -> CompressedRowCoordinates {
  self.coordinates
}

///|
pub fn CompressedIndexSelectionPrecision::geometry(
  self : Self,
) -> (Int32, Int32, Int32, Int32, Int32, Int32) {
  (self.heads, self.width, self.ratio, self.rows, self.positions, self.top_k)
}

///|
pub fn CompressedIndexSelectionPrecision::cache_capacity(self : Self) -> Int32 {
  self.positions / self.ratio
}

///|
pub fn CompressedIndexSelectionPrecision::selected_width(self : Self) -> Int32 {
  Int32::min(self.top_k, self.cache_capacity())
}

///|
pub fn CompressedIndexSelectionPrecision::frame_bytes(
  self : Self,
) -> ReadOnlyArray[Int64] {
  self.frames
}

///|
pub fn CompressedIndexSelectionPrecision::workspace_bytes(self : Self) -> Int64 {
  self.workspace
}

///|
pub fn CompressedIndexSelectionPrecision::effect(_self : Self) -> NumericEffect {
  StepWrite
}
