///|
pub(all) enum ProjectionMapBoundary {
  InputWritten
  InputRead
  ResultWritten
  ResultRead
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
) -> ProjectionStorageFence? {
  match boundary {
    InputWritten =>
      if self.shared_input {
        Some(
          WorkgroupStorage.materialization_effects(reused=true).publication(),
        )
      } else {
        None
      }
    InputRead =>
      if self.shared_input {
        WorkgroupStorage.materialization_effects(reused=true).release()
      } else {
        None
      }
    ResultWritten =>
      Some(
        ConsumerGroupStorage.materialization_effects(reused=true).publication(),
      )
    ResultRead =>
      ConsumerGroupStorage.materialization_effects(reused=true).release()
  }
}
