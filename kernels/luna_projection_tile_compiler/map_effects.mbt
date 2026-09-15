///|
pub(all) enum ProjectionMapBoundary {
  InputWritten
  InputRead
  ResultWritten
  ResultRead
} derive(Debug, Eq)

///|
/// Visibility/reuse scope, not a device instruction or a fixed lane width.
pub enum ProjectionMapFence {
  WorkgroupFence
  ConsumerGroupFence
} derive(Debug, Eq)

///|
/// The strip materialization is shared only for selected-row input. Result
/// redistribution always borrows consumer-group-local storage. Publication
/// and reader release remain separate even when a backend spells them alike.
pub struct ProjectionMapEffectPlan {
  priv shared_input : Bool
} derive(Debug, Eq)

///|
pub fn ProjectionResultDemand::map_effect_plan(
  self : ProjectionResultDemand,
) -> ProjectionMapEffectPlan {
  { shared_input: self == QueryRowEnds, }
}

///|
pub fn ProjectionMapEffectPlan::at(
  self : ProjectionMapEffectPlan,
  boundary : ProjectionMapBoundary,
) -> ProjectionMapFence? {
  match boundary {
    InputWritten | InputRead =>
      if self.shared_input {
        Some(WorkgroupFence)
      } else {
        None
      }
    ResultWritten | ResultRead => Some(ConsumerGroupFence)
  }
}
