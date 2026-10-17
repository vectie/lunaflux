///|
/// Independent numerical partitions mapped onto a bounded workgroup. This
/// plan does not authorize reassociation: the precision input already names
/// the exact tree. Fusion retains collapse's BF16 publication boundary.
pub struct ParallelControlPlan {
  priv rows : Int
  priv controls : Int
  priv partitions : Int
  priv fused_normalization : Bool
}

///|
pub fn ParallelControlPlan::new(
  precision : @precision.HyperConnectionPrecision,
) -> Self raise @precision.PrecisionPlanError {
  let (streams, hidden, rows) = precision.geometry()
  guard precision.control_law() is NormalizedBf16Projection &&
    precision.control_square_law()
    is VectorizedTreeFloat32(partitions~, vector_width=1) &&
    precision.control_dot_law() is StripedTreeFloat32(partitions=dot_partitions) &&
    dot_partitions == partitions &&
    partitions >= 32 &&
    partitions <= 1024 &&
    streams <= 16 &&
    hidden <= 65536 &&
    rows.to_int64() * streams.to_int64() * (streams + 2).to_int64() <=
    2147483647L else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  let fused_normalization = match precision.normalization_square_law() {
    VectorizedTreeFloat32(partitions=p, vector_width=1) => p == partitions
    _ => false
  }
  { rows, controls: streams * (streams + 2), partitions, fused_normalization, }
}

///|
pub fn ParallelControlPlan::fuses_normalization(self : Self) -> Bool {
  self.fused_normalization
}

///|
pub fn ParallelControlPlan::workers(self : Self) -> Int {
  self.partitions
}

///|
pub fn ParallelControlPlan::geometry(self : Self) -> (Int, Int) {
  (self.rows, self.controls)
}

///|
pub fn ParallelControlPlan::workspace_bytes(self : Self) -> Int64 {
  self.rows.to_int64() * (self.controls + 1).to_int64() * 4L
}

///|
pub fn ParallelControlPlan::workgroups(
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
pub fn ParallelControlPlan::local_bytes(self : Self) -> Int64 {
  self.partitions.to_int64() * 4L
}
