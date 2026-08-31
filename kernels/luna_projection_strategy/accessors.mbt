///|
pub fn ProjectionShape::family(
  self : ProjectionShape,
) -> ProjectionSemanticFamily {
  self.family
}

///|
pub fn ProjectionShape::dimensions(self : ProjectionShape) -> (Int, Int, Int) {
  (self.input_width, self.output_width, self.intermediate_width)
}

///|
pub fn ProjectionTile::dimensions(self : ProjectionTile) -> (Int, Int, Int) {
  (self.rows, self.columns, self.reduction)
}

///|
pub fn ProjectionCapabilities::subgroup_width(
  self : ProjectionCapabilities,
) -> Int {
  self.subgroup_width
}

///|
pub fn ProjectionCapabilities::matrix_tile(
  self : ProjectionCapabilities,
) -> ProjectionTile? {
  self.matrix_tile
}

///|
pub fn ProjectionStrategy::stable_id(self : ProjectionStrategy) -> Int {
  self.stable_id
}

///|
pub fn ProjectionStrategy::phase(self : ProjectionStrategy) -> ProjectionPhase {
  self.phase
}

///|
pub fn ProjectionStrategy::kernel_class(
  self : ProjectionStrategy,
) -> ProjectionKernelClass {
  self.kernel_class
}

///|
pub fn ProjectionStrategy::row_bucket(self : ProjectionStrategy) -> Int {
  self.row_bucket
}

///|
pub fn ProjectionStrategy::tile(self : ProjectionStrategy) -> ProjectionTile {
  self.tile
}

///|
pub fn ProjectionDispatchPlan::shape(
  self : ProjectionDispatchPlan,
) -> ProjectionShape {
  self.shape
}

///|
pub fn ProjectionDispatchPlan::decode(
  self : ProjectionDispatchPlan,
) -> ProjectionStrategy {
  self.decode
}

///|
pub fn ProjectionDispatchPlan::prefill(
  self : ProjectionDispatchPlan,
) -> ProjectionStrategy {
  self.prefill
}

///|
pub fn ProjectionAutotuneRecord::strategy(
  self : ProjectionAutotuneRecord,
) -> ProjectionStrategy {
  self.strategy
}

///|
pub fn ProjectionAutotuneRecord::median_latency_ns(
  self : ProjectionAutotuneRecord,
) -> Int64 {
  self.median_latency_ns
}

///|
pub fn ProjectionAutotuneRecord::sample_count(
  self : ProjectionAutotuneRecord,
) -> Int {
  self.sample_count
}
