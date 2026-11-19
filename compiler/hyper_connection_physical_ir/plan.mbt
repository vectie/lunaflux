///|
/// Distribute independent control dots without reassociating a single fold.
/// Retained products are consumed in increasing logical-column order. The
/// three dispatch boundaries publish inverse -> mixes -> collapse/controls.
/// Worker counts are abstract; no subgroup or vendor instruction is semantic.
pub struct OrderedControlPlan {
  priv rows : Int32
  priv controls : Int32
  priv workers : Int32
  priv tile : Int32
  priv scratch : Int64
}

///|
pub fn OrderedControlPlan::new(
  precision : @precision.HyperConnectionPrecision,
  workers~ : Int32,
  product_tile~ : Int32,
) -> Self raise @precision.PrecisionPlanError {
  let (streams, _, rows) = precision.geometry()
  guard precision.control_square_law() is OrderedFloat32 &&
    precision.control_dot_law() is OrderedFloat32 &&
    workers > 0 &&
    workers <= 1024 &&
    product_tile > 0 &&
    product_tile <= 4096 &&
    product_tile % workers == 0 &&
    streams <= 16 &&
    rows.to_int64() * streams.to_int64() * (streams.to_int64() + 2L) <=
    2147483647L else {
    raise NoImplementation
  }
  let controls = streams * (streams + 2)
  {
    rows,
    controls,
    workers,
    tile: product_tile,
    scratch: rows.to_int64() * (controls.to_int64() + 1L) * 4L,
  }
}

///|
pub fn OrderedControlPlan::workers(self : Self) -> Int32 {
  self.workers
}

///|
pub fn OrderedControlPlan::product_tile(self : Self) -> Int32 {
  self.tile
}

///|
pub fn OrderedControlPlan::workspace_bytes(self : Self) -> Int64 {
  self.scratch
}

///|
pub fn OrderedControlPlan::workgroups(
  self : Self,
  stage : Int32,
) -> Int32 raise @precision.PrecisionPlanError {
  match stage {
    0 | 2 => self.rows
    1 => self.rows * self.controls
    _ => raise IllegalPlacement
  }
}

///|
pub fn OrderedControlPlan::local_bytes(self : Self) -> Int64 {
  self.tile.to_int64() * 4L
}

///|
pub fn OrderedControlPlan::geometry(self : Self) -> (Int32, Int32) {
  (self.rows, self.controls)
}
