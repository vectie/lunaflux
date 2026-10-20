///|
/// A residual publication retained in private row partitions until the next
/// control projection consumes it. Each control tile repeats the producer, but
/// exactly one tile publishes the residual and inverse. Numerical ownership
/// (the striped folds and BF16 round) is independent of this work distribution.
pub struct DeferredPostPrePlan {
  priv streams : Int
  priv hidden : Int
  priv rows : Int
  priv workers : Int
  priv controls_per_group : Int
  priv workgroup_budget : Int
}

///|
pub fn DeferredPostPrePlan::new(
  producer : @precision.HyperConnectionPrecision,
  consumer : @precision.HyperConnectionPrecision,
  controls_per_group~ : Int,
  workgroup_budget? : Int = 2147483647,
) -> Self raise @precision.PrecisionPlanError {
  let (streams, hidden, rows, workers) = PostSquareFusion::new(
    producer, consumer,
  ).geometry()
  guard controls_per_group > 0 &&
    controls_per_group <= 8 &&
    workgroup_budget > 0 else {
    raise @precision.PrecisionPlanError::NoImplementation
  }
  { streams, hidden, rows, workers, controls_per_group, workgroup_budget, }
}

///|
pub fn DeferredPostPrePlan::geometry(
  self : Self,
) -> (Int, Int, Int, Int, Int) {
  (self.streams, self.hidden, self.rows, self.workers, self.controls_per_group)
}

///|
pub fn DeferredPostPrePlan::workgroups(self : Self) -> Int {
  let controls = self.streams * (self.streams + 2)
  Int::min(
    self.workgroup_budget,
    self.rows *
    ((controls + self.controls_per_group - 1) / self.controls_per_group),
  )
}

///|
pub fn DeferredPostPrePlan::retained_values_per_worker(self : Self) -> Int {
  (self.streams * self.hidden + self.workers - 1) / self.workers
}

///|
/// A complete component partition can reuse its source streams across every
/// destination. Otherwise lowering retains the general flattened traversal.
pub fn DeferredPostPrePlan::reuses_component_operands(self : Self) -> Bool {
  self.hidden % self.workers == 0
}
