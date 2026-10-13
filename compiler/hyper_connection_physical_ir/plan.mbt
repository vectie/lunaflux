///|
/// Distribute independent control dots without reassociating a single fold.
/// Retained products are consumed in increasing logical-column order. The
/// three dispatch boundaries publish inverse -> mixes -> collapse/controls.
/// Worker counts are abstract; no subgroup or vendor instruction is semantic.
pub struct OrderedControlPlan {
  priv rows : Int
  priv controls : Int
  priv workers : Int
  priv tile : Int
  priv scratch : Int64
}

///|
pub fn OrderedControlPlan::new(
  precision : @precision.HyperConnectionPrecision,
  workers~ : Int,
  product_tile~ : Int,
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
    raise @precision.PrecisionPlanError::NoImplementation
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
pub fn OrderedControlPlan::workers(self : Self) -> Int {
  self.workers
}

///|
pub fn OrderedControlPlan::product_tile(self : Self) -> Int {
  self.tile
}

///|
pub fn OrderedControlPlan::workspace_bytes(self : Self) -> Int64 {
  self.scratch
}

///|
pub fn OrderedControlPlan::workgroups(
  self : Self,
  stage : Int,
) -> Int raise @precision.PrecisionPlanError {
  match stage {
    0 | 2 => self.rows
    1 => self.rows * self.controls
    _ => raise @precision.PrecisionPlanError::IllegalPlacement
  }
}

///|
pub fn OrderedControlPlan::local_bytes(self : Self) -> Int64 {
  self.tile.to_int64() * 4L
}

///|
pub fn OrderedControlPlan::geometry(self : Self) -> (Int, Int) {
  (self.rows, self.controls)
}
