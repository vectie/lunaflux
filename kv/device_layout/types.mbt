///|
/// Stable key/value component within one layer-major split cache layout.
pub(all) enum DeviceKvComponent {
  Key
  Value
} derive(Debug, Eq)

///|
/// Narrow startup bounds for one persistent device-KV arena.
///
/// Layout V1 owns its canonical 256-byte segment alignment; callers select
/// only page geometry and the complete arena ceiling.
pub struct DeviceKvLimits {
  tokens_per_page : Int
  total_page_count : Int
  max_arena_bytes : Int64
} derive(Debug, Eq)

///|
/// One page-sized key or value segment in the persistent device-KV arena.
/// This is an immutable scalar address range, not a device allocation or view
/// with lifecycle authority.
#valtype
pub struct DeviceKvRegion {
  layer_index : Int
  component : DeviceKvComponent
  page_index : Int
  offset_bytes : Int64
  logical_bytes : Int64
  reserved_bytes : Int64
} derive(Debug, Eq)

///|
/// Canonical layer-major, split-key/value BF16 persistent-KV byte layout.
///
/// The value owns no storage. A later device package may allocate exactly
/// `arena_bytes` and use these checked scalar offsets, while this package stays
/// independent of native resources and execution policy.
pub struct DeviceKvLayout {
  identity : @spec.ModelIdentity
  geometry : @plan.KvCacheGeometry
  limits : DeviceKvLimits
  stable_layout_version : Int
  page_logical_bytes : Int64
  page_stride_bytes : Int64
  component_stride_bytes : Int64
  layer_stride_bytes : Int64
  arena_bytes : Int64
} derive(Debug, Eq)

///|
pub fn DeviceKvLimits::tokens_per_page(self : DeviceKvLimits) -> Int {
  self.tokens_per_page
}

///|
pub fn DeviceKvLimits::total_page_count(self : DeviceKvLimits) -> Int {
  self.total_page_count
}

///|
pub fn DeviceKvLimits::segment_alignment_bytes(_self : DeviceKvLimits) -> Int64 {
  CANONICAL_SEGMENT_ALIGNMENT_BYTES
}

///|
pub fn DeviceKvLimits::max_arena_bytes(self : DeviceKvLimits) -> Int64 {
  self.max_arena_bytes
}

///|
pub fn DeviceKvRegion::layer_index(self : DeviceKvRegion) -> Int {
  self.layer_index
}

///|
pub fn DeviceKvRegion::component(self : DeviceKvRegion) -> DeviceKvComponent {
  self.component
}

///|
pub fn DeviceKvRegion::page_index(self : DeviceKvRegion) -> Int {
  self.page_index
}

///|
pub fn DeviceKvRegion::offset_bytes(self : DeviceKvRegion) -> Int64 {
  self.offset_bytes
}

///|
pub fn DeviceKvRegion::logical_bytes(self : DeviceKvRegion) -> Int64 {
  self.logical_bytes
}

///|
pub fn DeviceKvRegion::reserved_bytes(self : DeviceKvRegion) -> Int64 {
  self.reserved_bytes
}

///|
pub fn DeviceKvRegion::alignment_bytes(_self : DeviceKvRegion) -> Int64 {
  CANONICAL_SEGMENT_ALIGNMENT_BYTES
}

///|
pub fn DeviceKvRegion::end_offset_bytes(self : DeviceKvRegion) -> Int64 {
  self.offset_bytes + self.reserved_bytes
}

///|
pub fn DeviceKvLayout::identity(self : DeviceKvLayout) -> @spec.ModelIdentity {
  self.identity
}

///|
pub fn DeviceKvLayout::geometry(self : DeviceKvLayout) -> @plan.KvCacheGeometry {
  self.geometry
}

///|
pub fn DeviceKvLayout::limits(self : DeviceKvLayout) -> DeviceKvLimits {
  self.limits
}

///|
pub fn DeviceKvLayout::stable_layout_version(self : DeviceKvLayout) -> Int {
  self.stable_layout_version
}

///|
pub fn DeviceKvLayout::tokens_per_page(self : DeviceKvLayout) -> Int {
  self.limits.tokens_per_page()
}

///|
pub fn DeviceKvLayout::total_page_count(self : DeviceKvLayout) -> Int {
  self.limits.total_page_count()
}

///|
pub fn DeviceKvLayout::segment_alignment_bytes(self : DeviceKvLayout) -> Int64 {
  self.limits.segment_alignment_bytes()
}

///|
pub fn DeviceKvLayout::page_logical_bytes(self : DeviceKvLayout) -> Int64 {
  self.page_logical_bytes
}

///|
pub fn DeviceKvLayout::page_stride_bytes(self : DeviceKvLayout) -> Int64 {
  self.page_stride_bytes
}

///|
pub fn DeviceKvLayout::component_stride_bytes(self : DeviceKvLayout) -> Int64 {
  self.component_stride_bytes
}

///|
pub fn DeviceKvLayout::layer_stride_bytes(self : DeviceKvLayout) -> Int64 {
  self.layer_stride_bytes
}

///|
pub fn DeviceKvLayout::arena_bytes(self : DeviceKvLayout) -> Int64 {
  self.arena_bytes
}
