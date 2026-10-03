///|
/// A subgroup owns all page facts for an aligned epoch. Every subgroup uses
/// identical immutable page metadata and uniform row bounds; the reduction is
/// therefore identical across the workgroup without a shared atomic/reload.
fn decode_independent_validity_transport(
  abi : CudaPagedAttentionTileAbi,
  hoist : Bool,
  tile_rows : Int,
  start_alignment : Int,
) -> @attention_physical_ir.PagedValidityOwnership? {
  let (_, heads, dimensions, page_rows) = abi.shape()
  let (_, stride, capacity) = abi.layout()
  guard hoist &&
    @attention_physical_ir.PagedCopyOrigin::new(
      page_stride=stride,
      page_capacity=capacity,
      within_page_offset=(heads - 1).to_int64() * dimensions.to_int64(),
    )
    is Some(_) &&
    @attention_physical_ir.PagedLookupLookahead::new(
      tile_rows~,
      page_rows~,
      start_alignment~,
      owners=32,
    )
    is Some(_) else {
    return None
  }
  @attention_physical_ir.PagedValidityOwnership::new(
    tile_rows~,
    page_rows~,
    start_alignment~,
    owners=32,
  )
}
