///|
/// Transfer completion is an explicit effect, independent of instruction set.
/// Overlap issues a future immutable operand before consuming the current one;
/// serial transfer issues it afterward. Both publish before the next consume.
pub(all) enum ProjectionTransferExecution {
  SerialTransfer
  OverlappedTransfer
} derive(Debug, Eq)

///|
/// Finite, immutable control plan for a ring of operand materializations.
/// Counts describe transfer groups, not CUDA async instructions or subgroups.
pub struct ProjectionFoldEffectPlan {
  priv stages : Int
  priv width : Int
  priv extent : Int
  priv execution : ProjectionTransferExecution
} derive(Debug, Eq)

///|
pub fn ProjectionFoldPipeline::effect_plan(
  self : ProjectionFoldPipeline,
  execution : ProjectionTransferExecution,
) -> ProjectionFoldEffectPlan {
  {
    stages: self.stage_count(),
    width: self.transfer_width(),
    extent: self.transfer_count() * self.transfer_width(),
    execution,
  }
}

///|
/// (ring slots, transfer width, reduction extent).
pub fn ProjectionFoldEffectPlan::geometry(
  self : ProjectionFoldEffectPlan,
) -> (Int, Int, Int) {
  (self.stages, self.width, self.extent)
}

///|
pub fn ProjectionFoldEffectPlan::prime_count(
  self : ProjectionFoldEffectPlan,
) -> Int {
  self.stages - 1
}

///|
pub fn ProjectionFoldEffectPlan::future_distance(
  self : ProjectionFoldEffectPlan,
) -> Int {
  self.prime_count() * self.width
}

///|
pub fn ProjectionFoldEffectPlan::overlaps(
  self : ProjectionFoldEffectPlan,
) -> Bool {
  self.execution == OverlappedTransfer
}

///|
/// At most this many younger issued groups may remain pending when the
/// current operand is consumed. At the tail, drain all groups instead.
pub fn ProjectionFoldEffectPlan::pending_group_limit(
  self : ProjectionFoldEffectPlan,
) -> Int {
  if self.overlaps() {
    self.stages - 2
  } else {
    0
  }
}
