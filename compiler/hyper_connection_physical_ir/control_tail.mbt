///|
/// Row-local softmax/collapse/normalization plus iterative Sinkhorn. Control
/// projection remains distributed. Initial and final matrices stay distinct;
/// only the iterative matrix storage becomes workgroup-local.
pub struct ControlTailFusion {
  priv rows : Int
  priv workers : Int
  priv matrix_cells : Int
  priv subgroup : Int
}

///|
pub fn ControlTailFusion::new(
  precision : @precision.HyperConnectionPrecision,
  subgroup_width? : Int = 1,
) -> Self raise @precision.PrecisionPlanError {
  let schedule = ParallelControlPlan::new(precision)
  let (streams, _, rows) = precision.geometry()
  guard subgroup_width > 0 &&
    subgroup_width <= schedule.workers() &&
    (subgroup_width & (subgroup_width - 1)) == 0 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  {
    rows,
    workers: schedule.workers(),
    matrix_cells: streams * streams,
    subgroup: if streams * streams <= subgroup_width {
      subgroup_width
    } else {
      1
    },
  }
}

///|
pub fn ControlTailFusion::geometry(self : Self) -> (Int, Int, Int) {
  (self.rows, self.workers, self.matrix_cells)
}

///|
/// Independent matrix cells share a subgroup; the reduction order within each
/// row/column is unchanged. Oversized matrices retain workgroup-local storage.
pub fn ControlTailFusion::matrix_subgroup(self : Self) -> Int {
  self.subgroup
}
