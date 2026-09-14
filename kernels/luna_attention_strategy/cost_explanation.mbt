///|
/// Dimensionless fallback estimates, not nanoseconds or hardware counters.
/// Keep the individual terms visible so an occupancy or memory improvement
/// cannot conceal an increase in repeated fold/transfer work.
pub struct AttentionCostBreakdown {
  arithmetic : Int64
  memory : Int64
  scheduling : Int64
  fold : Int64
  transfer : Int64
  total : Int64
} derive(Debug, Eq)

///|
/// Explain the existing fallback model without changing candidate selection.
/// Measurements remain the authoritative latency input. This is a compiler
/// operation, not request-path telemetry.
pub fn CompiledAttentionTilePlan::estimated_cost_breakdown(
  self : CompiledAttentionTilePlan,
) -> AttentionCostBreakdown raise AttentionStrategyError {
  attention_candidate_cost_breakdown(self.problem, self.candidate)
}
