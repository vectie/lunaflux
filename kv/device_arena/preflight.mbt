///|
const CANONICAL_REGION_ALIGNMENT : Int64 = 256L

///|
const MAX_DEVICE_KV_BYTES : Int64 = 9223372036854775807L

///|
fn checked_multiply(
  left : Int64,
  right : Int64,
  rule : DeviceKvArenaRule,
) -> Int64 raise DeviceKvArenaError {
  if left <= 0L || right <= 0L || left > MAX_DEVICE_KV_BYTES / right {
    raise InvalidLayout(rule)
  }
  left * right
}

///|
fn require_equal(
  actual : Int64,
  expected : Int64,
  rule : DeviceKvArenaRule,
) -> Unit raise DeviceKvArenaError {
  if actual != expected {
    raise InvalidLayout(rule)
  }
}

///|
fn require_region(
  actual : @device_layout.DeviceKvRegion,
  expected_offset : Int64,
  expected_end : Int64,
  rule : DeviceKvArenaRule,
) -> Unit raise DeviceKvArenaError {
  if actual.offset_bytes() != expected_offset ||
    actual.end_offset_bytes() != expected_end ||
    actual.alignment_bytes() != CANONICAL_REGION_ALIGNMENT ||
    actual.logical_bytes() <= 0L ||
    actual.reserved_bytes() < actual.logical_bytes() {
    raise InvalidLayout(rule)
  }
}

///|
/// Validates the complete semantic byte envelope before native allocation.
///
/// The expected identity must come from the loaded model generation that will
/// own this arena. The returned plan is immutable and owns no device resource.
pub fn preflight(
  expected_identity : @spec.ModelIdentity,
  layout : @device_layout.DeviceKvLayout,
) -> DeviceKvArenaPlan raise DeviceKvArenaError {
  if layout.identity() != expected_identity {
    raise InvalidLayout(Identity)
  }
  let geometry = layout.geometry()
  if layout.stable_layout_version() != geometry.layout().stable_version() ||
    layout.stable_layout_version() != 1 {
    raise InvalidLayout(LayoutVersion)
  }
  if layout.segment_alignment_bytes() != CANONICAL_REGION_ALIGNMENT {
    raise InvalidLayout(SegmentAlignment)
  }
  let page_stride = layout.page_stride_bytes()
  let page_logical = layout.page_logical_bytes()
  if page_logical <= 0L || page_stride < page_logical {
    raise InvalidLayout(PageLogicalBytes)
  }
  if page_stride % CANONICAL_REGION_ALIGNMENT != 0L {
    raise InvalidLayout(PageStride)
  }
  let component_stride = checked_multiply(
    page_stride,
    Int64::from_int(layout.total_page_count()),
    ComponentStride,
  )
  require_equal(
    layout.component_stride_bytes(),
    component_stride,
    ComponentStride,
  )
  let layer_stride = checked_multiply(component_stride, 2L, LayerStride)
  require_equal(layout.layer_stride_bytes(), layer_stride, LayerStride)
  let arena_bytes = checked_multiply(
    layer_stride,
    Int64::from_int(geometry.layer_count()),
    ArenaBytes,
  )
  require_equal(layout.arena_bytes(), arena_bytes, ArenaBytes)
  if arena_bytes > layout.limits().max_arena_bytes() {
    raise InvalidLayout(ArenaLimit)
  }
  let first = layout.region(0, Key, 0) catch {
    _ => raise InvalidLayout(FirstRegion)
  }
  require_region(first, 0L, page_stride, FirstRegion)
  let last = layout.region(
    geometry.layer_count() - 1,
    Value,
    layout.total_page_count() - 1,
  ) catch {
    _ => raise InvalidLayout(LastRegion)
  }
  require_region(last, arena_bytes - page_stride, arena_bytes, LastRegion)
  { layout, }
}
