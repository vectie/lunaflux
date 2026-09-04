///|
pub fn ProjectionTileValueId::as_int(self : ProjectionTileValueId) -> Int {
  self.0
}

///|
pub fn ProjectionTileBinding::id(
  self : ProjectionTileBinding,
) -> ProjectionTileValueId {
  self.id
}

///|
pub fn ProjectionTileBinding::kind(
  self : ProjectionTileBinding,
) -> ProjectionTileValueKind {
  self.kind
}

///|
pub fn ProjectionTileBinding::expression(
  self : ProjectionTileBinding,
) -> ProjectionTileExpression {
  self.expression
}

///|
pub fn ProjectionTileProgram::bindings(
  self : ProjectionTileProgram,
) -> ReadOnlyArray[ProjectionTileBinding] {
  self.bindings
}

///|
pub fn ProjectionTileProgram::root(
  self : ProjectionTileProgram,
) -> ProjectionTileValueId {
  self.root
}

///|
pub fn ProjectionTileProgram::demand(
  self : ProjectionTileProgram,
) -> ProjectionResultDemand {
  self.demand
}

///|
pub fn ProjectionTileProgram::digest(
  self : ProjectionTileProgram,
) -> ProjectionTileProgramDigest {
  self.digest
}

///|
pub fn ProjectionTileProgramDigest::as_hex(
  self : ProjectionTileProgramDigest,
) -> String {
  self.0
}

///|
pub fn ProjectionTileRewriteStep::rule(
  self : ProjectionTileRewriteStep,
) -> ProjectionTileRewriteRule {
  self.rule
}

///|
pub fn ProjectionTileRewriteStep::changed(
  self : ProjectionTileRewriteStep,
) -> Bool {
  self.changed
}

///|
pub fn ProjectionTileOptimization::trace(
  self : ProjectionTileOptimization,
) -> ReadOnlyArray[ProjectionTileRewriteStep] {
  self.trace
}

///|
pub fn ProjectionTileOptimization::digest(
  self : ProjectionTileOptimization,
) -> ProjectionTileOptimizationDigest {
  self.digest
}

///|
pub fn ProjectionTileOptimizationDigest::as_hex(
  self : ProjectionTileOptimizationDigest,
) -> String {
  self.0
}

///|
pub fn ProjectionLoopCombinator::components(
  self : ProjectionLoopCombinator,
) -> (ProjectionIterationKind, ProjectionIterationAxis, Int, Int) {
  (self.kind, self.axis, self.extent, self.tile)
}

///|
pub fn ProjectionTileSchedule::demand(
  self : ProjectionTileSchedule,
) -> ProjectionResultDemand {
  self.demand
}

///|
pub fn ProjectionTileSchedule::loops(
  self : ProjectionTileSchedule,
) -> ReadOnlyArray[ProjectionLoopCombinator] {
  self.loops
}

///|
pub fn ProjectionTileSchedule::reuse_input_tile(
  self : ProjectionTileSchedule,
) -> Bool {
  self.reuse_input_tile
}

///|
pub fn ProjectionTileSchedule::reuse_intermediate_tile(
  self : ProjectionTileSchedule,
) -> Bool {
  self.reuse_intermediate_tile
}

///|
pub fn ProjectionTileSchedule::digest(
  self : ProjectionTileSchedule,
) -> ProjectionTileScheduleDigest {
  self.digest
}

///|
pub fn ProjectionTileScheduleDigest::as_hex(
  self : ProjectionTileScheduleDigest,
) -> String {
  self.0
}

///|
pub fn ScheduledProjectionTileCompilation::program(
  self : ScheduledProjectionTileCompilation,
) -> ProjectionTileProgram {
  self.optimized.semantic.program
}

///|
pub fn ScheduledProjectionTileCompilation::optimization(
  self : ScheduledProjectionTileCompilation,
) -> ProjectionTileOptimization {
  self.optimized.optimization
}

///|
pub fn ScheduledProjectionTileCompilation::schedule(
  self : ScheduledProjectionTileCompilation,
) -> ProjectionTileSchedule {
  self.schedule
}

///|
pub fn ScheduledProjectionTileCompilation::digest(
  self : ScheduledProjectionTileCompilation,
) -> ProjectionTileCompilationDigest {
  self.digest
}

///|
pub fn ProjectionTileCompilationDigest::as_hex(
  self : ProjectionTileCompilationDigest,
) -> String {
  self.0
}
