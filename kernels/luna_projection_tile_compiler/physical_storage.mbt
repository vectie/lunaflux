///|
/// Peak live storage, not the sum of disjoint operand and epilogue lifetimes.
/// The first shared result plane reuses the retired operand ring; a second
/// plane, when present, belongs to separately admitted dynamic storage.
pub fn ProjectionPhysicalPipeline::static_storage_bytes(
  self : ProjectionPhysicalPipeline,
  element_bytes~ : Int,
  result_bytes~ : Int,
) -> Int64 {
  let operands = self.operands.allocation_elements().to_int64() *
    element_bytes.to_int64()
  if self.sibling_reuse {
    Int64::max(
      operands,
      self.schedule.row_width().to_int64() *
      self.schedule.output_tiles().to_int64() *
      self.schedule.tile_width().to_int64() *
      result_bytes.to_int64(),
    )
  } else {
    operands
  }
}
