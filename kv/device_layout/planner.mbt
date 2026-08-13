///|
const MAX_DEVICE_KV_BYTES : Int64 = 9223372036854775807L

///|
const BF16_BYTES : Int64 = 2L

///|
fn checked_multiply(
  left : Int64,
  right : Int64,
  stage : DeviceKvSizeStage,
) -> Int64 raise DeviceKvLayoutError {
  if left < 0L ||
    right < 0L ||
    (right != 0L && left > MAX_DEVICE_KV_BYTES / right) {
    raise SizeOverflow(stage)
  }
  left * right
}

///|
fn checked_add(
  left : Int64,
  right : Int64,
  stage : DeviceKvSizeStage,
) -> Int64 raise DeviceKvLayoutError {
  if left < 0L || right < 0L || left > MAX_DEVICE_KV_BYTES - right {
    raise SizeOverflow(stage)
  }
  left + right
}

///|
fn checked_align(
  value : Int64,
  alignment : Int64,
) -> Int64 raise DeviceKvLayoutError {
  let remainder = value % alignment
  if remainder == 0L {
    value
  } else {
    checked_add(value, alignment - remainder, PageStride)
  }
}

///|
fn validate_geometry(
  geometry : @plan.KvCacheGeometry,
) -> Unit raise DeviceKvLayoutError {
  match geometry.dtype() {
    BFloat16 => ()
    _ => raise UnsupportedGeometry(DType)
  }
  match geometry.layout() {
    LayerMajorSplitKeyValueV1 => ()
  }
  if geometry.layout().stable_version() != 1 {
    raise UnsupportedGeometry(Layout)
  }
}

///|
/// Derives the canonical persistent-KV byte layout without allocating storage.
///
/// V1 order is `[layer][key-or-value][physical-page]`; every page component
/// reserves one independently aligned segment containing
/// `[token][kv-head][head-dimension]` BF16 values. All arithmetic and the exact
/// arena ceiling are checked before the immutable scalar layout is returned.
pub fn plan(
  identity : @spec.ModelIdentity,
  geometry : @plan.KvCacheGeometry,
  limits : DeviceKvLimits,
) -> DeviceKvLayout raise DeviceKvLayoutError {
  validate_geometry(geometry)
  let page_elements = checked_multiply(
    Int64::from_int(limits.tokens_per_page()),
    Int64::from_int(geometry.kv_head_count()),
    PageElements,
  )
  let page_values = checked_multiply(
    page_elements,
    Int64::from_int(geometry.head_dimension()),
    PageElements,
  )
  let page_logical_bytes = checked_multiply(
    page_values,
    BF16_BYTES,
    PageLogicalBytes,
  )
  let page_stride_bytes = checked_align(
    page_logical_bytes,
    limits.segment_alignment_bytes(),
  )
  let component_stride_bytes = checked_multiply(
    page_stride_bytes,
    Int64::from_int(limits.total_page_count()),
    ComponentStride,
  )
  let layer_stride_bytes = checked_multiply(
    component_stride_bytes,
    2L,
    LayerStride,
  )
  let arena_bytes = checked_multiply(
    layer_stride_bytes,
    Int64::from_int(geometry.layer_count()),
    ArenaBytes,
  )
  if arena_bytes > limits.max_arena_bytes() {
    raise ArenaLimitExceeded
  }
  {
    identity,
    geometry,
    limits,
    stable_layout_version: geometry.layout().stable_version(),
    page_logical_bytes,
    page_stride_bytes,
    component_stride_bytes,
    layer_stride_bytes,
    arena_bytes,
  }
}

///|
/// Re-admits every derived scalar against the canonical planner.
///
/// This focused startup predicate lets aggregate packages reject a forged or
/// stale immutable layout before using any filesystem or native authority.
pub fn DeviceKvLayout::is_canonical_for(
  self : DeviceKvLayout,
  identity : @spec.ModelIdentity,
  geometry : @plan.KvCacheGeometry,
) -> Bool {
  let expected = plan(identity, geometry, self.limits) catch {
    _ => return false
  }
  self == expected
}

///|
fn component_index(component : DeviceKvComponent) -> Int64 {
  match component {
    Key => 0L
    Value => 1L
  }
}

///|
/// Resolves one layer/component/physical-page segment in constant time.
pub fn DeviceKvLayout::region(
  self : DeviceKvLayout,
  layer_index : Int,
  component : DeviceKvComponent,
  page_index : Int,
) -> DeviceKvRegion raise DeviceKvLayoutError {
  if layer_index < 0 || layer_index >= self.geometry.layer_count() {
    raise IndexOutOfBounds(Layer)
  }
  if page_index < 0 || page_index >= self.limits.total_page_count() {
    raise IndexOutOfBounds(Page)
  }
  let layer_offset = checked_multiply(
    Int64::from_int(layer_index),
    self.layer_stride_bytes,
    RegionOffset,
  )
  let component_offset = checked_multiply(
    component_index(component),
    self.component_stride_bytes,
    RegionOffset,
  )
  let page_offset = checked_multiply(
    Int64::from_int(page_index),
    self.page_stride_bytes,
    RegionOffset,
  )
  let offset_bytes = checked_add(
    checked_add(layer_offset, component_offset, RegionOffset),
    page_offset,
    RegionOffset,
  )
  {
    layer_index,
    component,
    page_index,
    offset_bytes,
    logical_bytes: self.page_logical_bytes,
    reserved_bytes: self.page_stride_bytes,
  }
}

///|
fn check_element_index(
  index : Int,
  upper_bound : Int,
  kind : DeviceKvIndexKind,
) -> Unit raise DeviceKvLayoutError {
  if index < 0 || index >= upper_bound {
    raise IndexOutOfBounds(kind)
  }
}

///|
/// Resolves one BF16 scalar in canonical
/// `[layer][key-or-value][page][token][kv-head][head-dimension]` order.
///
/// The result addresses the scalar's first byte inside the arena. Page padding
/// is never addressable as a logical element.
pub fn DeviceKvLayout::element_offset_bytes(
  self : DeviceKvLayout,
  layer_index : Int,
  component : DeviceKvComponent,
  page_index : Int,
  token_index : Int,
  kv_head_index : Int,
  head_dimension_index : Int,
) -> Int64 raise DeviceKvLayoutError {
  check_element_index(token_index, self.tokens_per_page(), Token)
  check_element_index(kv_head_index, self.geometry.kv_head_count(), KvHead)
  check_element_index(
    head_dimension_index,
    self.geometry.head_dimension(),
    HeadDimension,
  )
  let region = self.region(layer_index, component, page_index)
  let token_start = checked_multiply(
    Int64::from_int(token_index),
    Int64::from_int(self.geometry.kv_head_count()),
    RegionOffset,
  )
  let head_cell = checked_add(
    token_start,
    Int64::from_int(kv_head_index),
    RegionOffset,
  )
  let head_start = checked_multiply(
    head_cell,
    Int64::from_int(self.geometry.head_dimension()),
    RegionOffset,
  )
  let element_index = checked_add(
    head_start,
    Int64::from_int(head_dimension_index),
    RegionOffset,
  )
  let byte_offset = checked_multiply(element_index, BF16_BYTES, RegionOffset)
  checked_add(region.offset_bytes(), byte_offset, RegionOffset)
}
