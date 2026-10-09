///|
/// Immutable full-page copy geometry. Every slot contains K and V for all
/// layers in canonical layer/component order, excluding device alignment gaps.
pub struct Plan {
  priv layout : @device_layout.DeviceKvLayout
  priv limits : @residency.Limits
  priv components : Int
  priv bytes : Int64
  priv page_stride : Int64
  priv component_stride : Int64
  priv layer_stride : Int64
  priv pages : Int
}

///|
pub fn Plan::new(
  layout : @device_layout.DeviceKvLayout,
  slots~ : Int,
  transfer_lanes~ : Int,
  host_budget~ : Int64,
  separate_host_memory~ : Bool,
) -> Plan raise @residency.ResidencyError {
  let layers = layout.geometry().layer_count()
  if layers <= 0 || layers > 2048 || transfer_lanes > 64 {
    raise @residency.ResidencyError::InvalidLimits
  }
  let components = layers * 2
  let bytes = layout.page_logical_bytes()
  if bytes > 9223372036854775807L / components.to_int64() {
    raise @residency.ResidencyError::InvalidLimits
  }
  let limits = @residency.Limits::new(
    slots~,
    transfers=transfer_lanes,
    bytes_per_slot=bytes * components.to_int64(),
    host_budget~,
    separate_host_memory~,
  )
  {
    layout,
    limits,
    components,
    bytes,
    page_stride: layout.page_stride_bytes(),
    component_stride: layout.component_stride_bytes(),
    layer_stride: layout.layer_stride_bytes(),
    pages: layout.total_page_count(),
  }
}

///|
pub fn Plan::layout(self : Plan) -> @device_layout.DeviceKvLayout {
  self.layout
}

///|
pub fn Plan::host_bytes(self : Plan) -> Int64 {
  self.limits.host_bytes()
}

///|
pub fn Plan::slot_bytes(self : Plan) -> Int64 {
  self.limits.bytes_per_slot()
}

///|
pub fn Plan::slots(self : Plan) -> Int {
  self.limits.slots()
}

///|
pub fn Plan::transfer_lanes(self : Plan) -> Int {
  self.limits.transfers()
}

///|
#valtype
priv struct Region {
  host_offset : Int64
  device_offset : Int64
  bytes : Int64
}

///|
/// Pure scalar mapping; all products were bounded by canonical startup layout.
fn Plan::region(self : Plan, slot : Int, page : Int, component : Int) -> Region {
  {
    host_offset: slot.to_int64() * self.slot_bytes() +
    component.to_int64() * self.bytes,
    device_offset: (component / 2).to_int64() * self.layer_stride +
    (component % 2).to_int64() * self.component_stride +
    page.to_int64() * self.page_stride,
    bytes: self.bytes,
  }
}
