///|
/// Independent matrices sharing a left operand, with separate weights and
/// publication buffers. Concatenating the output domain does not concatenate
/// reductions or change intermediate rounding. Constructed only at startup.
pub struct SegmentedMatrixPlan {
  priv segments : ReadOnlyArray[MatrixTilePlan]
  priv offsets : ReadOnlyArray[Int]
  priv tiles : Int
}

///|
pub fn SegmentedMatrixPlan::new(
  segments : ArrayView[MatrixTilePlan],
) -> Self raise @precision.PrecisionPlanError {
  guard segments.length() > 1 && segments.length() <= 32 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let (rows, _, inner) = segments[0].geometry()
  let offsets = []
  let mut tiles = 0L
  for segment in segments {
    let (sr, _, sk) = segment.geometry()
    guard sr == rows && sk == inner && segment.law() == segments[0].law() else {
      raise @precision.PrecisionPlanError::IllegalPlacement
    }
    offsets.push(tiles.to_int())
    tiles += segment.tile_count().to_int64()
    guard tiles <= 2147483647L else {
      raise @precision.PrecisionPlanError::MemoryBudgetExceeded
    }
  }
  {
    segments: ReadOnlyArray::from_array(segments.to_owned()),
    offsets: ReadOnlyArray::from_array(offsets),
    tiles: tiles.to_int(),
  }
}

///|
pub fn SegmentedMatrixPlan::segments(self : Self) -> ArrayView[MatrixTilePlan] {
  self.segments[:]
}

///|
pub fn SegmentedMatrixPlan::offsets(self : Self) -> ArrayView[Int] {
  self.offsets[:]
}

///|
pub fn SegmentedMatrixPlan::tile_count(self : Self) -> Int {
  self.tiles
}

///|
/// Mutually exclusive regions share one scratch lifetime. Matrix tile geometry
/// is common to all segments, while allocation element sizes belong to lowering.
pub fn SegmentedMatrixPlan::scratch_elements(self : Self) -> (Int, Int, Int) {
  let (m, n, k) = self.segments[0].tile()
  (m * k, n * k, m * n)
}
