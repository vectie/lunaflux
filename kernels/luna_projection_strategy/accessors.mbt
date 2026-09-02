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
pub fn ProjectionBackendId::stable_name(
  self : ProjectionBackendId,
) -> StringView {
  self.stable_name
}

///|
pub fn ProjectionTargetId::stable_name(self : ProjectionTargetId) -> StringView {
  self.stable_name
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
pub fn ProjectionStrategy::strategy_class(
  self : ProjectionStrategy,
) -> ProjectionStrategyClass {
  self.strategy_class
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
  self.key.strategy
}

///|
pub fn ProjectionAutotuneRecord::key(
  self : ProjectionAutotuneRecord,
) -> ProjectionAutotuneKey {
  self.key
}

///|
pub fn ProjectionAutotuneKey::backend(
  self : ProjectionAutotuneKey,
) -> ProjectionBackendId {
  self.backend
}

///|
pub fn ProjectionAutotuneKey::target(
  self : ProjectionAutotuneKey,
) -> ProjectionTargetId {
  self.target
}

///|
pub fn ProjectionAutotuneKey::numeric_type(
  self : ProjectionAutotuneKey,
) -> ProjectionNumericType {
  self.numeric_type
}

///|
pub fn ProjectionAutotuneKey::shape(
  self : ProjectionAutotuneKey,
) -> ProjectionShape {
  self.shape
}

///|
pub fn ProjectionAutotuneKey::strategy(
  self : ProjectionAutotuneKey,
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

///|
pub fn ProjectionSelectedStrategy::strategy(
  self : ProjectionSelectedStrategy,
) -> ProjectionStrategy {
  self.strategy
}

///|
pub fn ProjectionSelectedStrategy::source(
  self : ProjectionSelectedStrategy,
) -> ProjectionSelectionSource {
  self.source
}

///|
pub fn ProjectionRuntimeDispatch::backend(
  self : ProjectionRuntimeDispatch,
) -> ProjectionBackendId {
  self.backend
}

///|
pub fn ProjectionRuntimeDispatch::target(
  self : ProjectionRuntimeDispatch,
) -> ProjectionTargetId {
  self.target
}

///|
pub fn ProjectionRuntimeDispatch::numeric_type(
  self : ProjectionRuntimeDispatch,
) -> ProjectionNumericType {
  self.numeric_type
}

///|
pub fn ProjectionRuntimeDispatch::shape(
  self : ProjectionRuntimeDispatch,
) -> ProjectionShape {
  self.shape
}

///|
pub fn ProjectionRuntimeDispatch::maximum_rows(
  self : ProjectionRuntimeDispatch,
) -> Int {
  self.maximum_rows
}
